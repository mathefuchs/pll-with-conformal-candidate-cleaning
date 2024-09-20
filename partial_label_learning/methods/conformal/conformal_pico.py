"""
Module for ConformalPiCO.

Code adapted from Haobo Wang <https://github.com/hbzju/PiCO/blob/main/resnet.py>.
License: Apache-2.0
"""

from collections import namedtuple
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.classifier_base import ClassifierBase
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult

Args = namedtuple("Args", [
    "num_class", "low_dim", "moco_queue", "moco_m", "proto_m"])


class PiCOModule(nn.Module):
    """ PiCO module. """

    def __init__(self, base_encoder, num_class: int):
        super().__init__()
        self.args = Args(
            num_class=num_class, low_dim=128,
            moco_queue=8192, moco_m=0.999, proto_m=0.99,
        )
        self.encoder_q = deepcopy(base_encoder)
        self.encoder_k = deepcopy(base_encoder)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # create the queue
        self.register_buffer("queue", torch.randn(
            self.args.moco_queue, self.args.low_dim))
        self.register_buffer("queue_pseudo", torch.randn(self.args.moco_queue))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("prototypes", torch.zeros(
            self.args.num_class, self.args.low_dim))
        self.queue = F.normalize(self.queue, dim=0)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """ Update momentum encoder. """

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.args.moco_m + \
                param_q.data * (1. - self.args.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.args.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_pseudo[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.args.moco_queue  # move pointer
        self.queue_ptr[0] = ptr

    def forward(
        self, original_input, corrupted_input, partial_labels,
        epoch, num_val, non_conformities_val, eval_only=False,
    ):
        """ Forward pass. """

        # Predict
        probs, output, q = self.encoder_q(original_input)
        if eval_only:
            return output

        # Conformal purification
        if epoch >= 10:
            # Use mean weights on non-candidates as a proxy
            # for the likelihood of misprediction
            alpha = 0.05
            beta = torch.mean(
                torch.sum(probs * (1 - partial_labels), dim=1))
            eps = 0.5 ** (epoch - 9)

            # Get non-conformity per class
            new_nonconf = 1 - probs * (1 - eps)
            ranks = num_val - torch.searchsorted(
                non_conformities_val, new_nonconf)
            p_vals = (ranks + 1) / (num_val + 1)
            conformal_pred = torch.where(
                p_vals > alpha + beta, 1.0, 0.0)
        else:
            conformal_pred = 1
        common_filter = conformal_pred * partial_labels
        w_filter = torch.where(
            common_filter.sum(dim=1, keepdim=True) >= 1.0,
            common_filter, partial_labels,
        )

        # Using partial labels to filter out negative labels
        predicted_scores = probs * w_filter
        _, pseudo_labels_b = torch.max(predicted_scores, dim=1)

        # Compute protoypical logits
        prototypes = self.prototypes.clone().detach()
        logits_prot = torch.mm(q, prototypes.t())
        score_prot = torch.softmax(logits_prot, dim=1)

        # Update momentum prototypes with pseudo labels
        for feat, label in zip(q, pseudo_labels_b):
            self.prototypes[label] = (
                self.prototypes[label] * self.args.proto_m +
                (1 - self.args.proto_m) * feat
            )

        # Normalize prototypes
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1).detach()

        with torch.no_grad():
            self._momentum_update_key_encoder()
            _, _, k = self.encoder_k(corrupted_input)

        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        pseudo_labels = torch.cat(
            (pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)
        self._dequeue_and_enqueue(k, pseudo_labels_b)

        return output, features, pseudo_labels, score_prot


class PartialLoss(nn.Module):
    """ Partial-label loss. """

    def __init__(self, confidence, conf_ema_m=0.99):
        super().__init__()
        self.confidence = confidence
        self.init_conf = confidence.detach()
        self.conf_ema_m = conf_ema_m

    def set_conf_ema_m(self, epoch, num_epochs):
        """ Set the moving average parameter. """

        start, end = 0.95, 0.8
        self.conf_ema_m = epoch / num_epochs * (end - start) + start

    def forward(self, outputs, index):
        """ Forward pass. """

        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - ((final_outputs).sum(dim=1)).mean()
        return average_loss

    @torch.no_grad()
    def confidence_update(self, temp_un_conf, batch_index, y_batch):
        """ Confidence update. """

        _, prot_pred = (temp_un_conf * y_batch).max(dim=1)
        pseudo_label = F.one_hot(
            prot_pred, y_batch.shape[1]).float().cuda().detach()
        self.confidence[batch_index, :] = (
            self.conf_ema_m * self.confidence[batch_index, :] +
            (1 - self.conf_ema_m) * pseudo_label
        )


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: 
    https://arxiv.org/pdf/2004.11362.pdf.
    """

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, batch_size, mask):
        """ Forward pass of MoCo loss. """

        if mask is not None:
            # SupCon loss (Partial Label Mode)
            device = features.device
            mask = mask.float().detach().to(device)
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size], features.T),
                self.temperature)
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0,
            )
            mask = mask * logits_mask
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - \
                torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            loss = - (self.temperature / self.base_temperature) * \
                mean_log_prob_pos
            loss = loss.mean()
        else:
            # Unsupervised MoCo loss
            q = features[:batch_size]
            k = features[batch_size:batch_size*2]
            queue = features[batch_size*2:]
            l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
            l_neg = torch.einsum("nc,kc->nk", [q, queue])
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= self.temperature
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)
        return loss


class ConformalPiCO(PllBaseClassifier):
    """
    PiCO by Wang et al. (2022),
    "PiCO: Contrastive Label Disambiguation for Partial Label Learning"
    """

    def __init__(
        self, rng: np.random.Generator, debug: bool,
        model: ClassifierBase, device: torch.device,
        is_small_scale_dataset: bool,
    ) -> None:
        super().__init__(
            rng, debug, model, device, is_small_scale_dataset)
        self.mu: np.ndarray = np.array([])
        self.std: np.ndarray = np.array([])
        self.pico_model: Optional[PiCOModule] = None

    def fit(
        self, inputs: np.ndarray, partial_targets: np.ndarray,
    ) -> SplitResult:
        """ Fits the model to the given inputs.

        Args:
            inputs (np.ndarray): The inputs.
            partial_targets (np.ndarray): The partial targets.

        Returns:
            SplitResult: The disambiguated targets.
        """

        self.pico_model = PiCOModule(
            self.model, partial_targets.shape[1])
        self.pico_model.to(self.device)

        # Separate validation set
        num_val = max(100, int(inputs.shape[0] * 0.2))
        all_indices = np.arange(inputs.shape[0])
        sup_mask = partial_targets.sum(axis=1) == 1
        sup_ind = np.arange(inputs.shape[0])[sup_mask]
        self.rng.shuffle(all_indices)
        self.rng.shuffle(sup_ind)
        if inputs.shape[0] / partial_targets.shape[1] < 1000:
            train_idx = all_indices.copy()  # Use all data if too few samples
        else:
            train_idx = all_indices[:-num_val]
        if sup_ind.shape[0] >= num_val:
            val_idx = sup_ind.copy()  # Use supervised samples if enough
        else:
            val_idx = all_indices[-num_val:]
        num_val = val_idx.shape[0]

        # Create tensors and dataloader
        x_train = torch.tensor(inputs[train_idx], dtype=torch.float32)
        x_train_noise = torch.tensor(inputs[train_idx], dtype=torch.float32)
        x_train_noise = x_train_noise + 0.1 * torch.randn_like(x_train_noise)
        y_train = torch.tensor(partial_targets[train_idx], dtype=torch.float32)
        train_indices = torch.arange(x_train.shape[0], dtype=torch.int32)
        data_loader = DataLoader(
            TensorDataset(train_indices, x_train, x_train_noise, y_train),
            batch_size=self.batch_size, shuffle=True, drop_last=True,
        )

        # Validation set
        val_batch_size = self.batch_size
        x_val = torch.tensor(inputs[val_idx], dtype=torch.float32)
        y_val = torch.tensor(partial_targets[val_idx], dtype=torch.float32)
        val_dataloader = DataLoader(
            TensorDataset(x_val, y_val),
            batch_size=val_batch_size, shuffle=False,
        )

        # Optimizer
        self.pico_model.train()
        optimizer = torch.optim.Adam(
            self.pico_model.parameters(), lr=self.max_lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.max_lr, epochs=self.num_epoch,
            steps_per_epoch=len(data_loader),
        )

        # Losses
        loss_weights = torch.tensor(partial_targets, dtype=torch.float32)
        loss_weights /= loss_weights.sum(dim=1, keepdim=True)
        loss_weights = loss_weights.to(self.device)
        loss_fn = PartialLoss(loss_weights)
        loss_cont_fn = SupConLoss()

        # Training loop
        non_conformities_val: torch.Tensor = torch.zeros(
            x_val.shape[0], dtype=torch.float32, device=self.device)
        for epoch in self.loop_wrapper(range(self.num_epoch)):
            # Compute non-conformities on validation set
            self.model.eval()
            with torch.no_grad():
                for batch_idx, (x_i, y_i) in enumerate(val_dataloader):
                    x_i = x_i.to(self.device)
                    y_i = y_i.to(self.device)
                    pred = self.model(x_i)[0]

                    # Take the minimum non-conformity
                    # among all candidates as reference
                    non_conformity = 1 - pred
                    non_conformity = torch.where(
                        y_i == 1, non_conformity, torch.inf)
                    non_conformity = torch.min(non_conformity, dim=1).values
                    non_conformities_val[
                        (batch_idx * val_batch_size):((batch_idx + 1) * val_batch_size)
                    ] = non_conformity

                # Sort non-conformities for ranking
                non_conformities_val[:] = non_conformities_val.sort().values

            # Train and purify train set
            self.model.train()
            for idx, inputs_i, inputs_noise_i, partial_targets_i in data_loader:
                # Move to device
                idx = idx.to(self.device)
                inputs_i = inputs_i.to(self.device)
                inputs_noise_i = inputs_noise_i.to(self.device)
                partial_targets_i = partial_targets_i.to(self.device)

                # Forward pass
                cls_out, features_cont, pseudo_target_cont, score_prot = \
                    self.pico_model(inputs_i, inputs_noise_i,
                                    partial_targets_i, epoch, num_val, non_conformities_val)
                batch_size = cls_out.shape[0]
                pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1)

                # Update prototypes
                if epoch >= 1:
                    loss_fn.confidence_update(
                        temp_un_conf=score_prot, batch_index=idx,
                        y_batch=partial_targets_i,
                    )
                    mask = torch.eq(
                        pseudo_target_cont[:batch_size],
                        pseudo_target_cont.T,
                    ).float().cuda()
                else:
                    mask = None

                # Compute loss
                loss_cont = loss_cont_fn(
                    features=features_cont, batch_size=batch_size, mask=mask)
                loss_cls = loss_fn(cls_out, idx)
                loss = loss_cls + 0.5 * loss_cont

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Update loss parameters
            loss_fn.set_conf_ema_m(epoch, self.num_epoch)

        # Get predictions
        inference_dataloader = DataLoader(
            TensorDataset(
                torch.tensor(inputs, dtype=torch.float32),
                torch.tensor(partial_targets, dtype=torch.float32),
            ),
            batch_size=self.batch_size, shuffle=False,
        )
        self.pico_model.eval()
        with torch.no_grad():
            all_res = []
            for x_i, s_i in inference_dataloader:
                x_i = x_i.to(self.device)
                s_i = s_i.to(self.device)
                outputs = s_i * F.softmax(self.pico_model(
                    original_input=x_i, corrupted_input=None,
                    partial_labels=None, epoch=None, num_val=None,
                    non_conformities_val=None, eval_only=True,
                ), dim=1)
                all_res.append(outputs.cpu().numpy())
            train_probs = np.vstack(all_res)

        # Return results
        return SplitResult.from_scores(self.rng, train_probs)

    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        if self.pico_model is None:
            raise ValueError()

        x_test = torch.tensor(inputs, dtype=torch.float32)
        inference_loader = DataLoader(
            TensorDataset(x_test),
            batch_size=self.batch_size, shuffle=False,
        )

        # Switch to eval mode
        self.pico_model.eval()
        all_results = []
        with torch.no_grad():
            for x_batch in inference_loader:
                x_batch = x_batch[0].to(self.device)
                all_results.append(F.softmax(self.pico_model(
                    original_input=x_batch, corrupted_input=None,
                    partial_labels=None, epoch=None, num_val=None,
                    non_conformities_val=None, eval_only=True,
                ), dim=1).cpu().numpy())
            test_probs = np.vstack(all_results)
        return SplitResult.from_scores(self.rng, test_probs)
