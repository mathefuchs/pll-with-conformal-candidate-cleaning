"""
    Module for ConformalValen.
    Code adapted from Congyu Qiao <https://github.com/palm-ml/valen>.
    License: Mulan PSL v2
"""

from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import lil_array
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult


class VAEBernulliDecoder(nn.Module):
    """ Decoder for VALEN. """

    def __init__(self, n_in, n_hidden, n_out) -> None:
        super().__init__()
        self.layer1 = nn.Linear(n_in, n_hidden)
        self.layer2 = nn.Linear(n_hidden, n_out)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)

    def forward(self, inputs):
        """ Forward pass. """

        h0 = self.layer1(inputs)
        h0 = F.relu(h0)
        x_hat = self.layer2(h0)
        return x_hat


class ConformalValen(PllBaseClassifier):
    """
    Valen by Xu et al.,
    "Instance-Dependent Partial Label Learning"
    """

    def _warmup(
        self, data_loader: DataLoader, loss_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Optimizer
        if self.model is None:
            raise ValueError()
        optimizer = torch.optim.Adam(self.model.parameters())

        # Training loop
        self.model.train()
        for _ in range(10):
            for idx, inputs_i, partial_targets_i, w_ij, _ in data_loader:
                # Move to device
                inputs_i = inputs_i.to(self.device)
                partial_targets_i = partial_targets_i.to(self.device)
                w_ij = w_ij.to(self.device)

                # Forward-backward pass
                probs, _, phi = self.model(inputs_i)
                loss, new_rel = self._partial_loss(
                    probs, w_ij, partial_targets_i, 0, 0, None)
                loss_weights[idx] = new_rel.cpu()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Extract features
        self.model.eval()
        feature_extracted = torch.zeros(
            (loss_weights.shape[0], phi.shape[-1]), dtype=torch.float32)
        with torch.no_grad():
            for idx, inputs_i, _, _, _ in data_loader:
                inputs_i = inputs_i.to(self.device)
                feature_extracted[idx, :] = self.model(inputs_i)[2].cpu()

        return feature_extracted

    def _partial_loss(
        self, output1: torch.Tensor, target: torch.Tensor,
        partial_targets_i, epoch, num_val, non_conformities_val,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = F.softmax(output1, dim=1)

        if epoch >= 10:
            # Use mean weights on non-candidates as a proxy
            # for the likelihood of misprediction
            c_alpha = 0.05
            beta = torch.mean(
                torch.sum(output * (1 - partial_targets_i), dim=1))
            eps = 0.5 ** (epoch - 9)

            # Get non-conformity per class
            new_nonconf = 1 - output * (1 - eps)
            ranks = num_val - torch.searchsorted(
                non_conformities_val, new_nonconf)
            p_vals = (ranks + 1) / (num_val + 1)
            conformal_pred = torch.where(
                p_vals > c_alpha + beta, 1.0, 0.0)
        else:
            conformal_pred = 1

        common_filter = target * conformal_pred * partial_targets_i
        new_targets = torch.where(
            common_filter.sum(dim=1, keepdim=True) > 1e-10,
            common_filter, target,
        )
        new_targets /= torch.sum(new_targets, dim=1, keepdim=True)

        ce_loss = new_targets * -torch.log(output + 1e-10)
        loss = torch.sum(ce_loss) / ce_loss.size(0)
        return loss, self._revised_target(output, new_targets)

    def _revised_target(
        self, output: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        revised_y = target.clone().detach()
        revised_y[revised_y > 0] = 1
        revised_y = revised_y * (output.clone().detach() + 1e-10)
        revised_y = revised_y / revised_y.sum(dim=1, keepdim=True)
        return revised_y

    def _alpha_loss(
        self, alpha: torch.Tensor, prior_alpha: torch.Tensor,
    ) -> torch.Tensor:
        kld = (
            torch.mvlgamma(alpha.sum(1), p=1) -
            torch.mvlgamma(alpha, p=1).sum(1) -
            torch.mvlgamma(prior_alpha.sum(1), p=1) +
            torch.mvlgamma(prior_alpha, p=1).sum(1) + (
                (alpha - prior_alpha) * (
                    torch.digamma(alpha) -
                    torch.digamma(
                        alpha.sum(dim=1, keepdim=True).expand_as(alpha))
                )
            ).sum(1)
        )
        return kld.mean()

    def _dot_product_decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.matmul(z, z.T))

    def _subselect_lilarray(
        self, sparse: lil_array, idx: torch.Tensor,
    ) -> torch.Tensor:
        idx_arr = idx.cpu().numpy()
        dense_subsample = sparse[
            idx_arr, :][:, idx_arr].todense().copy()  # type: ignore
        return torch.tensor(
            dense_subsample, dtype=torch.float32, device=self.device)

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

        # Create model
        num_classes = partial_targets.shape[1]
        num_feat = inputs.reshape((inputs.shape[0], -1)).shape[-1]
        encoder = deepcopy(self.model)
        decoder = VAEBernulliDecoder(num_classes, num_feat, num_feat)
        self.model.to(self.device)
        encoder.to(self.device)
        decoder.to(self.device)

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

        # Data preparation
        x_train = torch.tensor(inputs[train_idx], dtype=torch.float32)
        y_train = torch.tensor(partial_targets[train_idx], dtype=torch.float32)
        train_indices = torch.arange(x_train.shape[0], dtype=torch.int32)
        loss_weights = torch.tensor(
            partial_targets[train_idx], dtype=torch.float32)
        loss_weights /= loss_weights.sum(dim=1, keepdim=True)
        d_array = loss_weights.clone().detach()
        prior_alpha = torch.ones(
            (1, num_classes), dtype=torch.float32, device=self.device)
        data_loader = DataLoader(
            TensorDataset(
                train_indices, x_train, y_train, loss_weights, d_array,
            ),
            batch_size=self.batch_size, shuffle=True,
        )

        # Validation set
        val_batch_size = self.batch_size
        x_val = torch.tensor(inputs[val_idx], dtype=torch.float32)
        y_val = torch.tensor(partial_targets[val_idx], dtype=torch.float32)
        val_dataloader = DataLoader(
            TensorDataset(x_val, y_val),
            batch_size=val_batch_size, shuffle=False,
        )

        # Warm-up
        self.model.train()
        feature_extracted = self._warmup(data_loader, loss_weights)

        # Fit k-NN
        knn = NearestNeighbors(n_neighbors=3)
        knn.fit(feature_extracted.numpy())
        nn_indices = knn.kneighbors(return_distance=False)
        adj_matrix = lil_array((inputs.shape[0], inputs.shape[0]), dtype=float)
        for i, nn_row in enumerate(nn_indices):
            adj_matrix[i, i] = 1
            for neigh in nn_row:
                adj_matrix[neigh, i] = 1
        sum_adj = adj_matrix.sum(axis=1).reshape(-1, 1)
        sum_adj = np.where(sum_adj > 1e-10, sum_adj, 1.)
        adj_matrix = adj_matrix / sum_adj
        adj_matrix = adj_matrix.tolil()
        assert isinstance(adj_matrix, lil_array)

        # Optimizer
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) +
            list(encoder.parameters()) +
            list(decoder.parameters()),
            lr=self.max_lr, weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.max_lr, epochs=self.num_epoch,
            steps_per_epoch=len(data_loader),
        )

        # Training loop
        self.model.train()
        encoder.train()
        decoder.train()
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
            for idx, inputs_i, partial_targets_i, w_ij, d_ij in data_loader:
                # Move to device
                inputs_i = inputs_i.to(self.device)
                partial_targets_i = partial_targets_i.to(self.device)
                w_ij = w_ij.to(self.device)
                d_ij = d_ij.to(self.device)

                # Forward pass
                _, outputs, _ = self.model(inputs_i)
                _, alpha, _ = encoder(inputs_i)

                # Model and encoder loss
                loss_d, new_d = self._partial_loss(
                    alpha, w_ij, partial_targets_i,
                    epoch, num_val, non_conformities_val)
                loss_obj, new_o = self._partial_loss(
                    outputs, d_ij, partial_targets_i,
                    epoch, num_val, non_conformities_val)

                # Alpha loss
                s_alpha = F.softmax(alpha, dim=1)
                revised_alpha = torch.zeros_like(w_ij, device=self.device)
                revised_alpha[w_ij > 0] = 1.0
                s_alpha = (s_alpha + 1e-6) * revised_alpha
                s_alpha_sum = s_alpha.clone().detach().sum(dim=1, keepdim=True)
                s_alpha_sum = torch.where(s_alpha_sum < 1e-10, 1., s_alpha_sum)
                s_alpha = s_alpha / s_alpha_sum + 1e-2
                alpha = torch.exp(alpha / 4)
                alpha = F.hardtanh(alpha, min_val=1e-2, max_val=30)
                loss_alpha = self._alpha_loss(alpha, prior_alpha)

                # Reconstruction loss
                dirichlet_sample_machine = \
                    torch.distributions.dirichlet.Dirichlet(s_alpha)
                d: torch.Tensor = dirichlet_sample_machine.rsample()  # type: ignore
                x_hat: torch.Tensor = decoder(d)
                x_hat = x_hat.view(inputs_i.shape)
                a_hat = F.softmax(self._dot_product_decode(d), dim=1)
                loss_rec_x = F.mse_loss(x_hat, inputs_i)
                loss_rec_y = 0.01 * F.binary_cross_entropy_with_logits(
                    d, partial_targets_i)
                loss_rec_a = F.mse_loss(
                    a_hat, self._subselect_lilarray(adj_matrix, idx))
                loss_rec = loss_rec_x + loss_rec_y + loss_rec_a

                # Backward pass on final loss
                final_loss = loss_d + loss_obj + loss_alpha + loss_rec

                optimizer.zero_grad()
                final_loss.backward()
                optimizer.step()
                scheduler.step()

                # Update weights
                new_d = self._revised_target(d, new_d)
                d_array[idx, :] = new_d.clone().detach().cpu()
                loss_weights[idx, :] = new_o.clone().detach().cpu()

        # Inference
        self.model.eval()
        inference_loader = DataLoader(
            TensorDataset(
                torch.tensor(inputs, dtype=torch.float32),
                torch.tensor(partial_targets, dtype=torch.float32),
            ),
            batch_size=self.batch_size, shuffle=False,
        )
        with torch.no_grad():
            all_results = []
            for x_batch, y_batch in inference_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                probs = self.model(x_batch)[0] * y_batch
                all_results.append(probs.cpu().numpy())
            train_probs = np.vstack(all_results)

        # Return results
        return SplitResult.from_scores(self.rng, train_probs)

    def predict(self, inputs: np.ndarray) -> SplitResult:
        """ Predict the labels.

        Args:
            inputs (np.ndarray): The inputs.

        Returns:
            SplitResult: The predictions.
        """

        inference_loader = DataLoader(
            TensorDataset(torch.tensor(
                inputs, dtype=torch.float32)),
            batch_size=256, shuffle=False,
        )

        # Switch to eval mode
        self.model.eval()
        all_results = []
        with torch.no_grad():
            for x_batch in inference_loader:
                x_batch = x_batch[0].to(self.device)
                all_results.append(
                    F.softmax(self.model(x_batch)[0], dim=1).cpu().numpy())
            test_probs = np.vstack(all_results)
        return SplitResult.from_scores(self.rng, test_probs)
