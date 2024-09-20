""" Module for bundling algorithm results. """

import math
from typing import List

import numpy as np
from scipy.stats import entropy


class SplitResult:
    """ Results on either the train or test set. """

    def __init__(
        self,
        pred: np.ndarray,
        conf: np.ndarray,
        is_guessing: np.ndarray,
        entr: np.ndarray,
    ) -> None:
        """ Init results.

        Args:
            pred (np.ndarray): A prediction for each instance.
            conf (np.ndarray): The confidence in each prediction.
            is_guessing (np.ndarray): Whether the prediction has been guessed.
            entr (np.ndarray): The normalized entropy of the probabilities.
        """

        self.pred = pred
        self.conf = np.clip(conf, 0, 1)
        self.is_guessing = is_guessing
        self.entr = np.clip(entr, 0, 1)

    @classmethod
    def from_logits(
        cls,
        rng: np.random.Generator,
        conf_logits: np.ndarray,
    ):
        """ Create results from logits with random tie-breaking. """

        # Extract predictions
        pred_list: List[int] = []
        guessing: List[bool] = []
        for score_row in conf_logits:
            max_idx = np.flatnonzero(np.isclose(score_row, np.max(score_row)))
            if max_idx.shape[0] == 1:
                pred_list.append(max_idx[0])
                guessing.append(False)
            elif max_idx.shape[0] > 1:
                pred_list.append(int(rng.choice(max_idx)))
                guessing.append(True)
            else:
                print("> Warning: Empty selection")
                pred_list.append(-1)
                guessing.append(True)

        # Compute probs for reference
        conf_logits = np.clip(conf_logits, -np.inf, 1e10)
        conf_logits -= np.max(conf_logits, axis=1, keepdims=True)
        logits_exp = np.exp(conf_logits)
        sum_exp = np.sum(logits_exp, axis=1, keepdims=True)
        sum_exp = np.where(sum_exp < 1e-10, 1.0, sum_exp)
        conf_probs = logits_exp / sum_exp
        conf_probs = np.clip(conf_probs, 0, 1)
        conf = np.max(conf_probs, axis=1)
        entr = entropy(conf_probs, axis=1) / math.log(conf_probs.shape[1])
        entr = np.where(np.isnan(entr), 1.0, entr)

        return cls(
            pred=np.array(pred_list), conf=conf,
            is_guessing=np.array(guessing), entr=entr,
        )

    @classmethod
    def from_scores(
        cls,
        rng: np.random.Generator,
        conf_scores: np.ndarray,
    ):
        """ Create results from non-negative scores with random tie-breaking. """

        # Extract prediction
        prob_sum = np.sum(conf_scores, axis=1, keepdims=True)
        prob_sum = np.where(prob_sum < 1e-10, 1.0, prob_sum)
        conf_probs = conf_scores / prob_sum
        conf_probs = np.clip(conf_probs, 0, 1)
        pred_list: List[int] = []
        guessing: List[bool] = []
        for score_row in conf_probs:
            max_idx = np.flatnonzero(np.isclose(score_row, np.max(score_row)))
            if max_idx.shape[0] == 1:
                pred_list.append(max_idx[0])
                guessing.append(False)
            elif max_idx.shape[0] > 1:
                pred_list.append(int(rng.choice(max_idx)))
                guessing.append(True)
            else:
                print("> Warning: Empty selection")
                pred_list.append(-1)
                guessing.append(True)

        conf = np.max(conf_probs, axis=1)
        entr = entropy(conf_probs, axis=1) / math.log(conf_probs.shape[1])
        entr = np.where(np.isnan(entr), 1.0, entr)
        return cls(
            pred=np.array(pred_list), conf=conf,
            is_guessing=np.array(guessing), entr=entr,
        )

    def frac_guessing(self) -> float:
        """ Returns the fraction of guessing. """

        return np.count_nonzero(self.is_guessing) / self.is_guessing.shape[0]


class Result:
    """ Results on train and test set. """

    def __init__(
        self,
        train_result: SplitResult,
        test_result: SplitResult,
    ) -> None:
        self.train_result = train_result
        self.test_result = test_result
