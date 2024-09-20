""" Script to run all experiments. """

import io
import os
from glob import glob
from typing import Dict, List, Tuple, Type

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from models.model_util import create_model, get_model_arch
from partial_label_learning.config import SELECTED_DATASETS
from partial_label_learning.data import Experiment
from partial_label_learning.methods.cavl_2021 import Cavl
from partial_label_learning.methods.cc_2020 import CC
from partial_label_learning.methods.conformal.conformal_cavl import \
    ConformalCavl
from partial_label_learning.methods.conformal.conformal_cc import ConformalCC
from partial_label_learning.methods.conformal.conformal_crosel import \
    ConformalCroSel
from partial_label_learning.methods.conformal.conformal_pico import \
    ConformalPiCO
from partial_label_learning.methods.conformal.conformal_pop import ConformalPop
from partial_label_learning.methods.conformal.conformal_proden import \
    ConformalProden
from partial_label_learning.methods.conformal.conformal_proden_ablation import \
    ConformalProdenAblation
from partial_label_learning.methods.conformal.conformal_valen import \
    ConformalValen
from partial_label_learning.methods.crosel_2024 import CroSel
from partial_label_learning.methods.pico_2022 import PiCO
from partial_label_learning.methods.pop_2023 import Pop
from partial_label_learning.methods.proden_2020 import Proden
from partial_label_learning.methods.valen_2021 import Valen
from partial_label_learning.pll_classifier_base import PllBaseClassifier
from partial_label_learning.result import Result

DEBUG = False
ALGOS: Dict[str, Tuple[int, Type[PllBaseClassifier]]] = {
    # Related work without conformal purification
    "proden-2020": (0, Proden),
    "cc-2020": (1, CC),
    "valen-2021": (2, Valen),
    "cavl-2021": (3, Cavl),
    # "pico-2022": (4, PiCO),
    "pop-2023": (5, Pop),
    "crosel-2024": (6, CroSel),
    # Related work with conformal purification
    "conformal-proden": (7, ConformalProden),
    "conformal-cc": (8, ConformalCC),
    "conformal-valen": (9, ConformalValen),
    "conformal-cavl": (10, ConformalCavl),
    # "conformal-pico": (11, ConformalPiCO),
    "conformal-pop": (12, ConformalPop),
    "conformal-crosel": (13, ConformalCroSel),
    # Ablation experiments
    "conformal-proden-abl": (14, ConformalProdenAblation),
}


def fts(number: float, max_digits: int = 6) -> str:
    """ Float to string. """

    return f"{float(number):.{max_digits}f}".rstrip("0").rstrip(".")


def get_header() -> str:
    """ Builds the header. """

    return "dataset,algo,seed,split,truelabel,predlabel\n"


def append_output(
    output: List[str], algo_name: str, exp: Experiment,
    result: Result, split: int,
) -> None:
    """ Create output from result. """

    if split == 0:
        res = result.train_result
        true_label_list = exp.datasplit.y_true_train
    elif split == 1:
        res = result.test_result
        true_label_list = exp.datasplit.y_true_test
    else:
        raise ValueError()

    for true_label, pred in zip(true_label_list, res.pred):
        output.append(f"{int(SELECTED_DATASETS[exp.dataset_name][0])}")
        output.append(f",{int(ALGOS[algo_name][0])},{int(exp.seed)}")
        output.append(f",{int(split)},{int(true_label)},{int(pred)}\n")


def print_debug_msg(
    algo_name: str, exp: Experiment, result: Result,
) -> None:
    """ Print debug message. """

    train_acc = accuracy_score(
        exp.datasplit.y_true_train, result.train_result.pred)
    test_acc = accuracy_score(
        exp.datasplit.y_true_test, result.test_result.pred)
    print(", ".join([
        f"{exp.dataset_name: >20}", f"{algo_name: >20}",
        f"{exp.seed}", f"{train_acc:.3f}", f"{test_acc:.3f}",
    ]))


def run_experiment(fname: str, algo_name: str, algo_type: Type[PllBaseClassifier]) -> None:
    """ Runs the given experiment. """

    # Skip experiment if results already exist
    res_fname = fname.split(os.sep)[-1].split(".")[0].strip()
    res_path = f"results/{algo_name}_{res_fname}.parquet.gz"
    if not DEBUG and os.path.isfile(res_path):
        return

    # Run experiment
    exp: Experiment = torch.load(fname, weights_only=False)
    rng = np.random.Generator(np.random.PCG64(exp.seed))
    torch.manual_seed(exp.seed)
    arch = get_model_arch(exp.dataset_name, algo_name)
    model, device = create_model(
        arch, exp.datasplit.y_train.shape[1],
        exp.datasplit.x_train.shape[1:],
    )
    algo = algo_type(
        rng, DEBUG, model, device, exp.datasplit.x_train.shape[0] < 5000)
    result = Result(
        train_result=algo.fit(
            exp.datasplit.x_train,
            exp.datasplit.y_train,
        ),
        test_result=algo.predict(exp.datasplit.x_test),
    )

    # Print debug message
    if DEBUG:
        print_debug_msg(algo_name, exp, result)

    # Store predictions
    output = [get_header()]
    append_output(output, algo_name, exp, result, split=0)
    append_output(output, algo_name, exp, result, split=1)
    if not DEBUG:
        csv_df = pd.read_csv(io.StringIO("".join(output)))
        csv_df.to_parquet(res_path, compression="gzip")


if __name__ == "__main__":
    if not DEBUG:
        # Run all experimental settings
        Parallel(n_jobs=1)(
            delayed(run_experiment)(fname, algo_name, algo_type)
            for fname in tqdm(list(sorted(glob("experiments/*.pt"))))
            for algo_name, (_, algo_type) in ALGOS.items()
        )
    else:
        # Run single experiments for debugging
        algo_n, (_, algo_t) = list(ALGOS.items())[0]
        run_experiment("experiments/lost_0.pt", algo_n, algo_t)
        algo_n, (_, algo_t) = list(ALGOS.items())[7]
        run_experiment("experiments/lost_0.pt", algo_n, algo_t)
