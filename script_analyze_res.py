""" Module for analyzing the results. """

from glob import glob
from typing import List

import numpy as np
import pandas as pd
import scipy.stats

from partial_label_learning.config import SELECTED_DATASETS

# Index to name maps
IDX_TO_DATASET = {
    i: d for d, (i, _) in SELECTED_DATASETS.items()
}
IDX_TO_ALGO = {
    # Deep Learning methods
    0: "\\textsc{Proden}",
    1: "\\textsc{Cc}",
    2: "\\textsc{Valen}",
    3: "\\textsc{Cavl}",
    5: "\\textsc{Pop}",
    6: "\\textsc{CroSel}",
    # Our method
    7: "\\textsc{Conf+Proden}",
    8: "\\textsc{Conf+Cc}",
    9: "\\textsc{Conf+Valen}",
    10: "\\textsc{Conf+Cavl}",
    12: "\\textsc{Conf+Pop}",
    13: "\\textsc{Conf+CroSel}",
    # Ablation experiment
    14: "\\textsc{Conf+Proden} (no correction)",
}


def main() -> None:
    """ Main. """

    # Load all results
    all_res_list = []
    for file in sorted(glob("results/*.gz")):
        all_res_list.append(pd.read_parquet(file))
    all_res = pd.concat(all_res_list)
    all_res["correct"] = (all_res["truelabel"] ==
                          all_res["predlabel"]).astype(int)
    del all_res_list

    bugs = list(map(lambda t: (IDX_TO_DATASET[t[0]], IDX_TO_ALGO[t[1]]), list(set(map(
        lambda t: t[1:],
        list(all_res.query("split == 1 and predlabel == -1")[
            ["dataset", "algo"]].itertuples())
    )))))
    if bugs:
        print("Bugs found: ", bugs)

    # Groups of datasets
    rl_datasets = [0, 1, 2, 3, 4, 5]
    sup_datasets = [6, 7, 8, 9, 10, 11]
    algo_order = [0, 14, 7, 1, 8, 2, 9, 3, 10, 5, 12, 6, 13]

    # Print info
    all_int_res = []
    for group in [rl_datasets, sup_datasets]:
        # Get mean (std) performance
        merged_df = None
        for ds in group:
            intermediate_df = all_res.query(
                f"dataset == {ds} and split == 1").groupby(
                    by=["algo", "seed"], as_index=False)[
                        ["algo", "seed", "correct"]].mean()
            all_int_res.append(intermediate_df)
            df = intermediate_df.groupby(by="algo")["correct"].agg(
                ["mean", "std"]).iloc[algo_order, :]
            df.rename(columns={
                "mean": f"{IDX_TO_DATASET[ds]}_mean",
                "std": f"{IDX_TO_DATASET[ds]}_std",
            }, inplace=True)

            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(
                    merged_df, df, left_index=True, right_index=True)

        assert merged_df is not None
        merged_df.index = merged_df.index.map(IDX_TO_ALGO)
        merged_df = merged_df.reset_index()

        print()
        print("\\toprule")
        print(
            "Method  &  " +
            "  &  ".join([
                "\\emph{" + s.split("_")[0] + "}" for s in list(merged_df.columns)[1::2]]) +
            "  \\\\"
        )
        print("\\midrule")
        for tup in merged_df.itertuples():
            if int(tup[0]) in (3, 5, 7, 9, 11, 13):
                print("\\midrule")
            row = "\\mbox{" + str(tup[1]) + "}"
            for j in range(2, 14, 2):
                row += "  &  \\mbox{"
                row += f"{100 * float(tup[j]):.2f}"
                row += f" ($\\pm$ {100 * float(tup[j + 1]):.2f})"
                row += "}"
            row += "  \\\\"
            print(row)
        print("\\bottomrule")
        print()

    def get_algo_ds(ds: int, algo: int) -> List[float]:
        return list(all_int_res[ds][all_int_res[ds]["algo"] == algo]["correct"])

    # Get significant differences
    sig_res = {}
    sig_res_detail = {}
    for algo1 in sorted(list(IDX_TO_ALGO.keys())):
        for algo2 in [0, 1, 2, 3, 5, 6]:
            # Get wins/ties/losses for (algo1, algo2) pair
            if algo1 == algo2:
                wtl_counts = np.array([0, len(IDX_TO_DATASET), 0])
                for ds in range(len(IDX_TO_DATASET)):
                    sig_res_detail[(algo1, algo2, ds)] = np.array([0, 1, 0])
            else:
                wtl_counts = np.array([0, 0, 0])
                for ds in range(len(IDX_TO_DATASET)):
                    acc1 = get_algo_ds(ds, algo1)
                    acc2 = get_algo_ds(ds, algo2)
                    test = scipy.stats.ttest_rel(acc1, acc2)
                    if sum(acc1) > sum(acc2) and test.pvalue < 0.05:
                        wtl_counts[0] += 1
                        sig_res_detail[(algo1, algo2, ds)
                                       ] = np.array([1, 0, 0])
                    elif test.pvalue >= 0.05:
                        wtl_counts[1] += 1
                        sig_res_detail[(algo1, algo2, ds)
                                       ] = np.array([0, 1, 0])
                    elif sum(acc1) < sum(acc2) and test.pvalue < 0.05:
                        wtl_counts[2] += 1
                        sig_res_detail[(algo1, algo2, ds)
                                       ] = np.array([0, 0, 1])
                    else:
                        raise ValueError()
            sig_res[(algo1, algo2)] = wtl_counts

    # Print table
    print("\\toprule")
    print("Comparison vs. all others  &  Wins  &  Ties  &  Losses  \\\\")
    print("\\midrule")
    for i, algo1 in enumerate(algo_order):
        if i in (3, 5, 7, 9, 11, 13):
            print("\\midrule")
        temp = np.array([0, 0, 0])
        a1 = IDX_TO_ALGO[algo1]
        for algo2 in [0, 1, 2, 3, 5, 6]:
            temp += sig_res[(algo1, algo2)]
        assert np.sum(temp) == 72
        s0 = f"{int(temp[0])}" if int(
            temp[0]) >= 10 else "\\phantom{0}" + f"{int(temp[0])}"
        s1 = f"{int(temp[1])}" if int(
            temp[1]) >= 10 else "\\phantom{0}" + f"{int(temp[1])}"
        s2 = f"{int(temp[2])}" if int(
            temp[2]) >= 10 else "\\phantom{0}" + f"{int(temp[2])}"
        print(f"{a1}  &  {s0}  &  {s1}  &  {s2}  \\\\")
    print("\\bottomrule")
    print()


if __name__ == "__main__":
    main()
