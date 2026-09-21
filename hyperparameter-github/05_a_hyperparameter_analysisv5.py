"""
05_a_hyperparameter_analysis.py

MSDS-Net v4.0 ProtocolV2.1 hyperparameter sensitivity analysis.

This script inherits:
    MSDSNet_v4.0_hyperparameterWrapper_Backbone.py

"""

import os
import json
import copy
import argparse
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

import importlib.util


FULL_FILE = "MSDSNet_v4.0_hyperparameterWrapper_Backbone.py"


spec = importlib.util.spec_from_file_location(
    "base",
    FULL_FILE
)

base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


#SEEDS = [42]
SEEDS = [42, 2024, 2025, 2026, 2027,
         2028, 2029, 2030, 2031, 2032]


DEFAULT_CFG = {
    "a_residual_mix": 0.80,
    "graph_score_weight": 0.031,
    "threshold_quantile": 0.990
}


SEARCH_SPACE = {


    # --------------------------------------------------------
    # Residual consistency coefficient
    # FULL: 0.80 excluded
    # --------------------------------------------------------
    "a_residual_mix": [
        0.00,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.90,
        1.00
    ],





    # --------------------------------------------------------
    # Graph score weight
    # FULL: 0.031 excluded
    # --------------------------------------------------------
    "graph_score_weight": [
        0.000,
        0.010,
        0.020,
        0.040,
        0.050,
        0.060
    ],



    # --------------------------------------------------------
    # Threshold quantile
    # FULL: 0.990 excluded
    # --------------------------------------------------------
    # "threshold_q": [
    #     0.90,
    #     0.93,
    #     0.95,
    #     0.995
    # ],



    # --------------------------------------------------------
    # Loss sensitivity
    # --------------------------------------------------------

    # lambda_trend
    # FULL: 1.75 excluded
    "lambda_trend": [
        0.50,
        1.00,
        2.50,
        3.50
    ],


    # lambda_dist
    # FULL: 0.10 excluded
    "lambda_dist": [
        0.01,
        0.05,
        0.20,
        0.50
    ],


    # lambda_res
    # FULL: 0.0008 excluded
    "lambda_res": [
        0.0001,
        0.0020,
        0.0030,
        0.0040
    ]

}


def set_cfg_value(cfg, key, value):

    cfg = copy.deepcopy(cfg)

    if key == "a_residual_mix":
        cfg.a_residual_mix = float(value)

    elif key == "graph_score_weight":
        cfg.graph_score_weight = float(value)

    # elif key == "threshold_q":
    #     cfg.threshold_quantile = float(value)

    elif key == "lambda_trend":
        cfg.lambda_trend = float(value)

    elif key == "lambda_dist":
        cfg.lambda_dist = float(value)

    elif key == "lambda_res":
        cfg.lambda_res = float(value)

    return cfg


def run_one(setting_name: str,
            parameter: str,
            value: float,
            seed: int,
            output_dir: str):

    cfg = set_cfg_value(base.CFG, parameter, value)

    cfg.hyperparameter_mode = True

    cfg.seed = int(seed)

    print(
        "USED CFG:",
        parameter,
        value,
        cfg.a_residual_mix,
        cfg.graph_score_weight,
        cfg.threshold_quantile,
        cfg.lambda_trend,
        cfg.lambda_dist,
        cfg.lambda_res
    )




    # 每个敏感性配置独立目录
    exp_dir = Path(output_dir) / (
        f"{parameter}_{value}"
    ) / (
        f"seed_{seed}"
    )

    exp_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    cfg.output_dir = str(exp_dir)


    result = base.run_full_experiment(cfg)


    result["SensitivityParameter"] = parameter
    result["SensitivityValue"] = float(value)
    result["Seed"] = int(seed)
    result["Setting"] = setting_name


    return result

def summarize(results):

    groups = {}

    for r in results:
        key = (
            r["SensitivityParameter"],
            r["SensitivityValue"]
        )

        groups.setdefault(key, []).append(r)

    summary = []

    for key, vals in groups.items():

        row = {
            "Parameter": key[0],
            "Value": key[1],
            "Runs": len(vals)
        }

        for metric in [
            "PointPrecision",
            "PointRecall",
            "PointF1",

            "ROC_AUC",
            "PR_AUC_AP",

            "FullFAR",
            "MDR",

            "AdjPrecision",
            "AdjRecall",
            "AdjF1",

            "EventPrecision",
            "EventRecall",
            "EventF1",

            "DelayMeanMin",
            "DelayMedianMin",

            "Merge120EventPrecision",
            "Merge120EventRecall",
            "Merge120EventF1",
            "Merge120DelayMeanMin"
        ]:
            data = [
                x[metric] for x in vals
                if metric in x and np.isfinite(x[metric])
            ]

            if len(data):
                row[metric + "_mean"] = float(np.mean(data))
                row[metric + "_std"] = float(np.std(data, ddof=1))

        summary.append(row)

    return summary

def save_checkpoint(all_results, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # 1. 原始逐 seed JSON
    # =========================================================
    raw_json = output_dir / "hyperparameter_raw_results.json"

    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump(
            all_results,
            f,
            ensure_ascii=False,
            indent=2
        )

    # =========================================================
    # 2. 原始逐 seed CSV
    # =========================================================
    raw_df = pd.DataFrame(all_results)

    # 把超参数、取值、seed放前面，方便后面查看
    first_cols = [
        "SensitivityParameter",
        "SensitivityValue",
        "Seed",
        "Setting"
    ]

    first_cols = [
        c for c in first_cols
        if c in raw_df.columns
    ]

    other_cols = [
        c for c in raw_df.columns
        if c not in first_cols
    ]

    raw_df = raw_df[first_cols + other_cols]

    raw_df.to_csv(
        output_dir / "hyperparameter_seedwise_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # =========================================================
    # 3. mean ± std
    # =========================================================
    summary = summarize(all_results)

    # JSON
    with open(
        output_dir / "hyperparameter_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    # CSV
    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        output_dir / "hyperparameter_mean_std.csv",
        index=False,
        encoding="utf-8-sig"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="05_hyperparameter_results"
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    raw_file = Path(args.output) / "hyperparameter_raw_results.json"

    if raw_file.exists():

        with open(
                raw_file,
                "r",
                encoding="utf-8"
        ) as f:
            all_results = json.load(f)

        print(
            f"Resume mode: loaded {len(all_results)} existing results"
        )

    else:

        all_results = []

    experiments = [

        ("a_residual_mix",
         "a_residual_mix",
         SEARCH_SPACE["a_residual_mix"]),

        ("graph_score_weight",
         "graph_score_weight",
         SEARCH_SPACE["graph_score_weight"]),

        # ("threshold_q",
        #  "threshold_q",
        #  SEARCH_SPACE["threshold_q"]),

        ("lambda_trend",
         "lambda_trend",
         SEARCH_SPACE["lambda_trend"]),

        ("lambda_dist",
         "lambda_dist",
         SEARCH_SPACE["lambda_dist"]),

        ("lambda_res",
         "lambda_res",
         SEARCH_SPACE["lambda_res"])
    ]


    for name, param, values in experiments:

        for v in values:

            print(
                f"\nRunning {name} = {v}"
            )

            for seed in SEEDS:

                exp_key = {
                    "SensitivityParameter": param,
                    "SensitivityValue": float(v),
                    "Seed": int(seed)
                }

                already_done = any(
                    r.get("SensitivityParameter") == exp_key["SensitivityParameter"]
                    and float(r.get("SensitivityValue")) == exp_key["SensitivityValue"]
                    and int(r.get("Seed")) == exp_key["Seed"]
                    for r in all_results
                )

                if already_done:
                    print(
                        f"Skip finished: {param}={v}, seed={seed}"
                    )

                    continue

                exp_dir = (
                        Path(args.output)
                        / f"{param}_{v}"
                        / f"seed_{seed}"
                )

                result_file = exp_dir / "seed_result.json"

                if result_file.exists():
                    print(
                        f"Skip finished: {param}={v}, seed={seed}"
                    )

                    with open(
                            result_file,
                            "r",
                            encoding="utf-8"
                    ) as f:
                        result = json.load(f)

                    all_results.append(result)
                    save_checkpoint(
                        all_results,
                        args.output
                    )

                    # immediate checkpoint save
                    with open(
                            raw_file,
                            "w",
                            encoding="utf-8"
                    ) as f:
                        json.dump(
                            all_results,
                            f,
                            indent=2,
                            ensure_ascii=False
                        )

                    # update summary immediately
                    summary = summarize(all_results)

                    with open(
                            Path(args.output) /
                            "hyperparameter_summary.json",
                            "w",
                            encoding="utf-8"
                    ) as f:
                        json.dump(
                            summary,
                            f,
                            indent=2,
                            ensure_ascii=False
                        )
                    continue

                print(
                    f"seed={seed}"
                )

                result = run_one(
                    name,
                    param,
                    v,
                    seed,
                    args.output
                )

                result_file.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                with open(
                        result_file,
                        "w",
                        encoding="utf-8"
                ) as f:
                    json.dump(
                        result,
                        f,
                        indent=2,
                        ensure_ascii=False
                    )

                all_results.append(result)


    summary = summarize(all_results)


    with open(
        Path(args.output) /
        "hyperparameter_raw_results.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            all_results,
            f,
            indent=2,
            ensure_ascii=False
        )


    with open(
        Path(args.output) /
        "hyperparameter_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )


    print("\nFinished.")
    print(
        Path(args.output) /
        "hyperparameter_summary.json"
    )

    # ============================================================
    # CSV export
    # ============================================================

    import pandas as pd

    # seed-wise results
    seed_df = pd.DataFrame(all_results)

    seed_df.to_csv(
        Path(args.output) /
        "hyperparameter_seedwise_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # mean-std results
    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        Path(args.output) /
        "hyperparameter_mean_std.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "Saved:",
        Path(args.output) /
        "hyperparameter_seedwise_results.csv"
    )

    print(
        "Saved:",
        Path(args.output) /
        "hyperparameter_mean_std.csv"
    )

if __name__ == "__main__":
    main()
