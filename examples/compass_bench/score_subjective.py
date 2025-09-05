import argparse
import json
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
from loguru import logger
from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--abbr", type=str, default="compass-bench-v1.3-code")
    parser.add_argument("--base_dir", type=str, default="./outputs")
    return parser.parse_args()


def read_judge_stats(results_path: str, abbr: str):
    judges = set()
    models = set()
    judge_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for subdir in os.listdir(results_path):
        subdir_path = os.path.join(results_path, subdir)
        if not os.path.isdir(subdir_path):
            continue
        judge = subdir_path.split("judged-by--")[-1]
        judges.add(judge)
        json_path = os.path.join(subdir_path, f"{abbr}.json")
        if not os.path.isfile(json_path):
            continue
        try:
            with open(json_path) as f:
                results = json.load(f)
                for detail in results["details"]:
                    idx = detail["id"]
                    m1 = detail["answer1_source"]
                    m2 = detail["answer2_source"]
                    choice = detail["choice"]
                    judge_stats[idx][judge][m1][m2] = choice
                    models.update([m1, m2])
        except Exception as e:
            logger.error(f"Failed to load {json_path}: {e}")
    return judges, models, judge_stats


def calc_align_rates(judge_stats, models, strict_map, loose_map):
    judge_strict_aligns = defaultdict(int)
    judge_loose_aligns = defaultdict(int)
    judge_total_aligns = defaultdict(int)
    for idx, results in judge_stats.items():
        for judge, details in results.items():
            for m1, m2 in combinations(models, 2):
                choice1 = details.get(m1, {}).get(m2)
                choice2 = details.get(m2, {}).get(m1)
                missing = False
                if not choice1:
                    logger.warning(
                        f"data missing, m1: {m1}, m2: {m2}, judge: {judge}, idx: {idx}"
                    )
                    missing = True
                if not choice2:
                    logger.warning(
                        f"data missing, m1: {m2}, m2: {m1}, judge: {judge}, idx: {idx}"
                    )
                    missing = True
                if not missing:
                    if choice2 == strict_map.get(choice1, ""):
                        judge_strict_aligns[judge] += 1
                    if choice2 in loose_map.get(choice1, []):
                        judge_loose_aligns[judge] += 1
                judge_total_aligns[judge] += 1
    return judge_strict_aligns, judge_loose_aligns, judge_total_aligns


def print_judge_summary(
    judges,
    judge_strict_aligns,
    judge_loose_aligns,
    judge_total_aligns,
    strict_weights,
    loose_weights,
):
    table = []
    for judge in sorted(judges):
        strict = judge_strict_aligns[judge]
        loose = judge_loose_aligns[judge]
        total = judge_total_aligns[judge]
        strict_ratio = strict / total if total else 0
        loose_ratio = loose / total if total else 0
        s_weight = strict_weights.get(judge, 0)
        l_weight = loose_weights.get(judge, 0)
        table.append(
            [
                judge,
                f"{strict_ratio:.2%}",
                f"{loose_ratio:.2%}",
                total,
                f"{s_weight:.3f}",
                f"{l_weight:.3f}",
            ]
        )
    print("\nJudge Alignment Summary:")
    print(
        tabulate(
            table,
            headers=[
                "Judge",
                "Strict Align",
                "Loose Align",
                "Total",
                "Strict Weight",
                "Loose Weight",
            ],
            tablefmt="github",
        )
    )


def calc_weights(judge_aligns: dict, total: float) -> dict:
    return {k: v / total if total else 0 for k, v in judge_aligns.items()}


def calc_weighted_scores(
    judge_stats, models, strict_weights, loose_weights, scores_map
):
    strict_weighted_scores = defaultdict(lambda: defaultdict(float))
    loose_weighted_scores = defaultdict(lambda: defaultdict(float))
    for idx, results in judge_stats.items():
        for judge, details in results.items():
            strict_weight = strict_weights.get(judge, 0)
            loose_weight = loose_weights.get(judge, 0)
            for m1, m2 in combinations(models, 2):
                for a, b in [(m1, m2), (m2, m1)]:
                    choice = details.get(a, {}).get(b)
                    if choice:
                        score = scores_map.get(choice, 0)
                        strict_weighted_scores[a][b] += score * strict_weight
                        loose_weighted_scores[a][b] += score * loose_weight
    return strict_weighted_scores, loose_weighted_scores


def print_scores(title: str, weighted_scores, models: list[str]):
    table = []
    for m1 in models:
        for m2 in models:
            if m1 != m2:
                score = weighted_scores[m1][m2]
                if score != 0:
                    table.append([m1, m2, f"{score:.2f}"])
    if table:
        print(f"\n{title}")
        print(
            tabulate(table, headers=["Model A", "Model B", "Score"], tablefmt="github")
        )
    else:
        print(f"\n{title}\n(No nonzero scores)")


def calc_strength(weighted_scores, model_list):
    strength = {}
    for m1 in model_list:
        scores = [weighted_scores[m1][m2] for m2 in model_list if m1 != m2]
        strength[m1] = np.mean(scores) if scores else 0
    return strength


def normalize_strength(strength_dict):
    vals = np.array(list(strength_dict.values()))
    min_v, max_v = vals.min(), vals.max()
    if max_v == min_v:
        return {k: 0.5 for k in strength_dict}
    return {k: (v - min_v) / (max_v - min_v) for k, v in strength_dict.items()}


def print_strengths(title: str, strength_dict: dict, percent: bool = False):
    table = [
        [m, f"{(s * 100 if percent else s):.2f}"]
        for m, s in sorted(strength_dict.items(), key=lambda x: -x[1])
    ]
    print(f"\n{title}")
    print(tabulate(table, headers=["Model", "Score"], tablefmt="github"))


def main():
    try:
        args = parse_args()
        abbr = args.abbr
        base_dir = os.path.join(args.base_dir, abbr)

        # 获取最新输出目录
        outputs_dirs = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]
        if not outputs_dirs:
            print(f"No output dirs found in {base_dir}")
            return
        latest_output_dir = sorted(outputs_dirs)[-1]
        print("Latest output dir:", latest_output_dir)
        latest_output_dir = os.path.join(base_dir, latest_output_dir)
        results_path = os.path.join(latest_output_dir, "results")

        # 读取数据
        judges, models, judge_stats = read_judge_stats(results_path, abbr)
        model_list = sorted(models)

        STRICT_ALIGN_MAP = {
            "A++": "B++",
            "A+": "B+",
            "A=B": "A=B",
            "B+": "A+",
            "B++": "A++",
        }
        LOOSE_ALIGN_MAP = {
            "A++": ["B++", "B+"],
            "A+": ["B+", "B++"],
            "A=B": ["A=B"],
            "B+": ["A+", "A++"],
            "B++": ["A++", "A+"],
        }
        SCORES_MAP = {
            "A++": 1.0,
            "A+": 0.5,
            "A=B": 0.0,
            "B+": -0.5,
            "B++": -1,
        }

        # 统计对齐率
        judge_strict_aligns, judge_loose_aligns, judge_total_aligns = calc_align_rates(
            judge_stats, models, STRICT_ALIGN_MAP, LOOSE_ALIGN_MAP
        )
        strict_total = sum(judge_strict_aligns.values())
        loose_total = sum(judge_loose_aligns.values())
        strict_weights = calc_weights(judge_strict_aligns, strict_total)
        loose_weights = calc_weights(judge_loose_aligns, loose_total)
        print_judge_summary(
            judges,
            judge_strict_aligns,
            judge_loose_aligns,
            judge_total_aligns,
            strict_weights,
            loose_weights,
        )

        # 计算加权得分
        strict_weighted_scores, loose_weighted_scores = calc_weighted_scores(
            judge_stats, models, strict_weights, loose_weights, SCORES_MAP
        )

        print_scores("Strict weighted scores:", strict_weighted_scores, model_list)
        print_scores("Loose weighted scores:", loose_weighted_scores, model_list)

        # 能力分数与归一化
        strict_strength = calc_strength(strict_weighted_scores, model_list)
        loose_strength = calc_strength(loose_weighted_scores, model_list)
        norm_strict_strength = normalize_strength(strict_strength)
        norm_loose_strength = normalize_strength(loose_strength)

        print_strengths("Model Strengths (strict weighted):", strict_strength)
        print_strengths("Model Strengths (loose weighted):", loose_strength)
        print_strengths(
            "Normalized Model Strengths (strict weighted, 0-100):",
            norm_strict_strength,
            percent=True,
        )
        print_strengths(
            "Normalized Model Strengths (loose weighted, 0-100):",
            norm_loose_strength,
            percent=True,
        )
    except Exception as e:
        logger.error(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
