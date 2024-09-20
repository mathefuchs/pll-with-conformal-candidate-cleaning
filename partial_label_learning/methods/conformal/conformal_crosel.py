""" Module for ConformalCroSel. """

import itertools
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.classifier_base import ClassifierBase
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import SplitResult


class ConformalCroSel(PllBaseClassifier):
    """
    CroSel by Tian et al. (2024),
    "CroSel: Cross Selection of Confident Pseudo Labels for Partial-Label Learning".
    """

    def __init__(
        self, rng: np.random.Generator, debug: bool,
        model: ClassifierBase, device: torch.device,
        is_small_scale_dataset: bool,
    ) -> None:
        super().__init__(
            rng, debug, model, device, is_small_scale_dataset)
        self.model2 = deepcopy(model)
        self.model2.to(device)

    def _warm_up_with_cc(
        self, model: nn.Module, num_inst: int, num_classes: int,
        data_loader: DataLoader,
    ) -> torch.Tensor:
        # Optimizer
        model.train()
        optimizer = torch.optim.Adam(model.parameters())

        # Memory Bank
        memory_bank = torch.zeros(
            (1000, num_inst, num_classes), dtype=torch.float32)

        # Training loop
        for t in range(10):
            for idx_batch, x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Forward-backward pass
                model_out = model(x_batch)[0]
                probs = y_batch * model_out
                loss = torch.mean(-torch.log(
                    torch.sum(probs, dim=1) + 1e-10
                ))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Store to memory bank
                memory_bank[t, idx_batch, :] = model_out.cpu().clone().detach()

        return memory_bank

    @torch.no_grad()
    def _get_selection_mask_from_memory_bank(
        self, memory_bank: torch.Tensor, y_train: torch.Tensor,
        curr_epoch: int,
    ) -> torch.Tensor:
        thresh = 0.9

        beta_1 = torch.count_nonzero(torch.isclose(
            memory_bank[curr_epoch - 1],
            torch.max(memory_bank[curr_epoch - 1],
                      dim=-1, keepdim=True).values,
        ) * y_train, dim=-1) > 0  # N
        beta_2 = (
            torch.argmax(memory_bank[curr_epoch - 1], dim=-1) ==
            torch.argmax(memory_bank[curr_epoch - 2], dim=-1)
        ) * (
            torch.argmax(memory_bank[curr_epoch - 2], dim=-1) ==
            torch.argmax(memory_bank[curr_epoch - 3], dim=-1)
        )  # N
        beta_3 = (
            torch.max(torch.mean(
                memory_bank[curr_epoch - 3:curr_epoch], dim=0,
            ), dim=-1).values > thresh
        )  # N

        # Return mask of selected instances
        return beta_1 * beta_2 * beta_3  # N

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
        data_loader = DataLoader(
            TensorDataset(train_indices, x_train, y_train),
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

        # Optimizer
        opt1 = torch.optim.Adam(
            self.model.parameters(), lr=self.max_lr,
            weight_decay=self.weight_decay,
        )
        scheduler1 = torch.optim.lr_scheduler.OneCycleLR(
            opt1, max_lr=self.max_lr, epochs=self.num_epoch,
            steps_per_epoch=len(data_loader),
        )
        opt2 = torch.optim.Adam(
            self.model2.parameters(), lr=self.max_lr,
            weight_decay=self.weight_decay,
        )
        scheduler2 = torch.optim.lr_scheduler.OneCycleLR(
            opt2, max_lr=self.max_lr, epochs=self.num_epoch,
            steps_per_epoch=len(data_loader),
        )

        # Warm-up with CC
        warm_up_epochs = 10
        memory_bank1 = self._warm_up_with_cc(
            self.model, x_train.shape[0], partial_targets.shape[1], data_loader)
        memory_bank2 = self._warm_up_with_cc(
            self.model2, x_train.shape[0], partial_targets.shape[1], data_loader)
        lambda_cr = 2.0

        # Train loop
        criterion = nn.CrossEntropyLoss()
        non_conformities_val: torch.Tensor = torch.zeros(
            x_val.shape[0], dtype=torch.float32, device=self.device)
        for epoch in self.loop_wrapper(range(self.num_epoch)):
            # Compute non-conformities on validation set
            self.model.eval()
            self.model2.eval()
            with torch.no_grad():
                for batch_idx, (x_i, y_i) in enumerate(val_dataloader):
                    x_i = x_i.to(self.device)
                    y_i = y_i.to(self.device)
                    pred1 = self.model(x_i)[0]
                    pred2 = self.model2(x_i)[0]
                    pred = 0.5 * (pred1 + pred2)

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
            self.model2.train()
            mask1 = self._get_selection_mask_from_memory_bank(
                memory_bank1, y_train, epoch + warm_up_epochs)
            mask2 = self._get_selection_mask_from_memory_bank(
                memory_bank2, y_train, epoch + warm_up_epochs)
            count1 = int(torch.count_nonzero(mask1).item())
            count2 = int(torch.count_nonzero(mask2).item())
            frac_labeled1 = count1 / mask1.shape[0]
            frac_labeled2 = count2 / mask1.shape[0]
            lambda_d1 = (1 - frac_labeled1) * lambda_cr
            lambda_d2 = (1 - frac_labeled2) * lambda_cr
            y_hat1 = torch.argmax(
                memory_bank1[warm_up_epochs + epoch - 1], dim=-1)
            y_hat2 = torch.argmax(
                memory_bank2[warm_up_epochs + epoch - 1], dim=-1)
            drop_last1 = count1 % self.batch_size in [1, 2, 3, 4, 5]
            drop_last2 = count2 % self.batch_size in [1, 2, 3, 4, 5]

            if count1 >= 2:
                selected_data_loader1 = DataLoader(
                    TensorDataset(x_train[mask1], y_hat1[mask1]),
                    batch_size=self.batch_size, shuffle=True,
                    drop_last=drop_last1,
                )
            else:
                selected_data_loader1 = [(None, None)]
            if count2 >= 2:
                selected_data_loader2 = DataLoader(
                    TensorDataset(x_train[mask2], y_hat2[mask2]),
                    batch_size=self.batch_size, shuffle=True,
                    drop_last=drop_last2,
                )
            else:
                selected_data_loader2 = [(None, None)]

            # Model 1
            for (
                (idx_batch, x_batch, y_batch), (x_batch_sel, y_hat_sel),
            ) in zip(data_loader, itertools.cycle(selected_data_loader2)):
                # Ground-truth part
                if frac_labeled2 > 0.05 and count2 >= 2:
                    x_batch_sel = x_batch_sel.to(self.device)
                    y_hat_sel = y_hat_sel.to(self.device)
                    probs, _, _ = self.model(x_batch_sel)
                    loss_hat = criterion(probs, y_hat_sel)
                    enough_samples = True
                else:
                    loss_hat = 0.0
                    enough_samples = False

                # PLL part
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                probs_pll, _, _ = self.model(x_batch)
                weights_batch = memory_bank2[
                    warm_up_epochs + epoch - 1, idx_batch, :].clone().detach().to(self.device)

                # Conformal purification
                with torch.no_grad():
                    if enough_samples and epoch >= 10:
                        probs_pll1, _, _ = self.model(x_batch)
                        probs_pll2, _, _ = self.model2(x_batch)
                        c_probs = 0.5 * (probs_pll1 + probs_pll2)

                        # Use mean weights on non-candidates as a proxy
                        # for the likelihood of misprediction
                        alpha = 0.05
                        beta = torch.mean(
                            torch.sum(c_probs * (1 - y_batch), dim=1))
                        eps = 0.5 ** (epoch - 9)

                        # Get non-conformity per class
                        new_nonconf = 1 - c_probs * (1 - eps)
                        ranks = num_val - torch.searchsorted(
                            non_conformities_val, new_nonconf)
                        p_vals = (ranks + 1) / (num_val + 1)
                        conformal_pred = torch.where(
                            p_vals > alpha + beta, 1.0, 0.0)
                    else:
                        conformal_pred = 1
                    common_filter = weights_batch * conformal_pred * y_batch
                    weights_batch = torch.where(
                        common_filter.sum(dim=1, keepdim=True) > 1e-10,
                        common_filter, (weights_batch + 1e-3) * y_batch,
                    )
                    weights_batch /= torch.sum(
                        weights_batch, dim=1, keepdim=True)

                # Loss
                pll_loss = torch.mean(torch.sum(
                    weights_batch * -torch.log(probs_pll + 1e-10), dim=-1,
                ))
                loss = loss_hat + lambda_d1 * pll_loss

                opt1.zero_grad()
                loss.backward()
                opt1.step()
                scheduler1.step()

                memory_bank2[
                    warm_up_epochs + epoch, idx_batch, :,
                ] = probs_pll.cpu().clone().detach()

            # Model 2
            for (
                (idx_batch, x_batch, y_batch), (x_batch_sel, y_hat_sel),
            ) in zip(data_loader, itertools.cycle(selected_data_loader1)):
                # Ground-truth part
                if frac_labeled1 > 0.05 and count1 >= 2:
                    x_batch_sel = x_batch_sel.to(self.device)
                    y_hat_sel = y_hat_sel.to(self.device)
                    probs, _, _ = self.model2(x_batch_sel)
                    loss_hat = criterion(probs, y_hat_sel)
                    enough_samples = True
                else:
                    loss_hat = 0.0
                    enough_samples = False

                # PLL part
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                probs_pll, _, _ = self.model2(x_batch)
                weights_batch = memory_bank1[
                    warm_up_epochs + epoch - 1, idx_batch, :].clone().detach().to(self.device)

                # Conformal purification
                with torch.no_grad():
                    if enough_samples and epoch >= 10:
                        probs_pll1, _, _ = self.model(x_batch)
                        probs_pll2, _, _ = self.model2(x_batch)
                        c_probs = 0.5 * (probs_pll1 + probs_pll2)

                        # Use mean weights on non-candidates as a proxy
                        # for the likelihood of misprediction
                        alpha = 0.05
                        beta = torch.mean(
                            torch.sum(c_probs * (1 - y_batch), dim=1))
                        eps = 0.5 ** (epoch - 9)

                        # Get non-conformity per class
                        new_nonconf = 1 - c_probs * (1 - eps)
                        ranks = num_val - torch.searchsorted(
                            non_conformities_val, new_nonconf)
                        p_vals = (ranks + 1) / (num_val + 1)
                        conformal_pred = torch.where(
                            p_vals > alpha + beta, 1.0, 0.0)
                    else:
                        conformal_pred = 1
                    common_filter = weights_batch * conformal_pred * y_batch
                    weights_batch = torch.where(
                        common_filter.sum(dim=1, keepdim=True) > 1e-10,
                        common_filter, (weights_batch + 1e-3) * y_batch,
                    )
                    weights_batch /= torch.sum(
                        weights_batch, dim=1, keepdim=True)

                # Loss
                pll_loss = torch.mean(torch.sum(
                    weights_batch * -torch.log(probs_pll + 1e-10), dim=-1,
                ))
                loss = loss_hat + lambda_d2 * pll_loss

                opt2.zero_grad()
                loss.backward()
                opt2.step()
                scheduler2.step()

                memory_bank1[
                    warm_up_epochs + epoch, idx_batch, :,
                ] = probs_pll.cpu().clone().detach()

        # Inference
        self.model.eval()
        self.model2.eval()
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
                probs1 = self.model(x_batch)[0] * y_batch
                probs2 = self.model2(x_batch)[0] * y_batch
                probs = 0.5 * (probs1 + probs2)
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
            batch_size=self.batch_size, shuffle=False,
        )

        # Switch to eval mode
        self.model.eval()
        self.model2.eval()
        all_results = []
        with torch.no_grad():
            for x_batch in inference_loader:
                x_batch = x_batch[0].to(self.device)
                out1 = self.model(x_batch)[0].cpu().numpy()
                out2 = self.model2(x_batch)[0].cpu().numpy()
                all_results.append(0.5 * (out1 + out2))
            test_probs = np.vstack(all_results)
        return SplitResult.from_scores(self.rng, test_probs)
