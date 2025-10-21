import json
from collections import defaultdict

from math_verify import verify

# last - part1
# OUTPUTS_DIR = "outputs/compass-academic-202510-simple/20251011_004222"
# DATASETS = [
#     "aime-2025-n32",
#     "lcb-code-generation-n6",
# ]
# MODELS = [
#     "deepseek-v3.1",
#     "deepseek-v3-exp91",
#     "deepseek-v3-exp93",
#     "deepseek-v3-exp103",
#     "deepseek-v3-exp108",
#     "deepseek-v3-exp110",
#     "deepseek-v3-exp111",
#     "deepseek-v3-exp260",
#     "deepseek-v3-exp262",
# ]

# last - part2
# OUTPUTS_DIR = "outputs/deepseek-v3.1/20251015_145512"
# DATASETS = [
#     "aime-2024-n16",
# ]
# MODELS = [
#     "deepseek-v3.1",
#     "deepseek-v3-exp91",
#     "deepseek-v3-exp93",
#     "deepseek-v3-exp103",
#     "deepseek-v3-exp108",
#     "deepseek-v3-exp110",
#     "deepseek-v3-exp111",
# ]

# exp1
# OUTPUTS_DIR = "outputs/compass-academic-202510-fast-diff/20251017_213035"
# DATASETS = [
#     "aime-2024-n16",
#     "aime-2025-n16",
#     "lcb-code-generation-lite_2408_2505",
# ]
# MODELS = [
#     "deepseek-v3.1",
#     "deepseek-v3-0324",
# ]

# exp2
# OUTPUTS_DIR = "outputs/compass-academic-202510-fast-diff/20251018_001100"
# DATASETS = [
#     "aime-2024-n32",
#     "aime-2025-n32",
#     "lcb-code-generation-lite_2408_2505-n6",
# ]
# MODELS = [
#     "deepseek-v3.1",
#     "deepseek-v3-0324",
# ]

# exp3
# OUTPUTS_DIR = "outputs/compass-academic-202510-fast-diff/20251018_120415"
# DATASETS = [
#     "aime-2024-n32",
#     "aime-2025-n32",
# ]
# MODELS = [
#     "deepseek-v3.1",
#     "deepseek-v3-0324",
# ]

# exp4
# OUTPUTS_DIR = "outputs/compass-academic-202510-fast-diff/20251018_215349"
# DATASETS = [
#     "aime-2024-n32",
#     "aime-2025-n32",
# ]
# MODELS = [
#     "deepseek-v3.1",
#     "deepseek-v3-0324",
# ]

# exp5
# OUTPUTS_DIR = "outputs/compass-academic-202510-fast-diff/20251019_144712"
# DATASETS = [
#     "aime-2024-n32",
#     "aime-2025-n32",
# ]
# MODELS = [
#     "deepseek-v3.1",
#     "deepseek-v3-0324",
# ]

# exp6 - deepseek-v3.1-non-thinking
# OUTPUTS_DIR = "outputs/compass-academic-202510-fast-diff/20251019_160122"
# DATASETS = [
#     "aime-2024-n64",
#     "aime-2025-n64",
#     "lcb-code-generation-lite_2408_2505-n10",
# ]
# MODELS = [
#     "deepseek-v3.1",
#     "deepseek-v3-0324",
# ]

# exp6 - deepseek-v3.1-thinking
# OUTPUTS_DIR = "outputs/compass-academic-202510-fast-diff/20251020_114308"
# DATASETS = [
#     "aime-2024-n64",
#     "aime-2025-n64",
#     "lcb-code-generation-lite_2408_2505-n10",
# ]
# MODELS = [
#     "deepseek-v3.1-thinking",
# ]

# exp6 - deepseek-v3.1-terminus-thinking
# OUTPUTS_DIR = "outputs/compass-academic-202510-fast-diff/20251021_134930"
# DATASETS = [
#     "aime-2024-n64",
#     "aime-2025-n64",
#     "lcb-code-generation-lite_2408_2505-n10",
# ]
# MODELS = [
#     "deepseek-v3.1-terminus-thinking",
# ]

# exp6 - deepseek-v3.2-exp-thinking - vllm
# OUTPUTS_DIR = "outputs/compass-academic-202510-fast-diff/20251026_154321"
# DATASETS = [
#     "aime-2024-n64",
#     "aime-2025-n64",
#     "lcb-code-generation-lite_2408_2505-n10",
# ]
# MODELS = [
#     "deepseek-v3.2-exp-thinking",
# ]

# exp6 - deepseek-v3.2-exp-thinking - sgl
OUTPUTS_DIR = "./outputs/compass-academic-202510-fast-diff-sgl/20251027_164529"
DATASETS = [
    "aime-2024-n64",
    "aime-2025-n64",
    "lcb-code-generation-lite_2408_2505-n10",
]
MODELS = [
    "deepseek-v3.2-exp-thinking",
]

PREDS_DIR = f"{OUTPUTS_DIR}/predictions"
RESULTS_DIR = f"{OUTPUTS_DIR}/results"

for model in MODELS:
    print("------")
    print(model)
    print("------")
    for dataset_name in DATASETS:
        print(dataset_name)
        # res
        with open(f"{RESULTS_DIR}/{model}/{dataset_name}.json") as fh:
            obj = json.load(fh)
            details = obj["details"]

            # pass@k
            total = len(details)
            n = 0

            correct_each_question = [0] * total
            correct_each_run = []
            correct_each_question_major = [False] * total  # for cons@n
            for i, v in enumerate(details):
                correct_this_run = v["correct"]
                n = len(correct_this_run)  # assume each time has the same n

                correct_each_question[i] += sum(correct_this_run)

                if not correct_each_run:
                    correct_each_run = correct_this_run
                else:
                    for j, c in enumerate(correct_this_run):
                        correct_each_run[j] += c

                # for math datasets
                if "parsed_prediction" in v and "parsed_answer" in v:
                    parsed_predictions = v["parsed_prediction"]
                    predictions_counter = defaultdict(int)
                    for p in parsed_predictions:
                        predictions_counter[p] += 1
                    major_prediction = max(
                        predictions_counter, key=predictions_counter.get
                    )  # random select when tie, since py dict is not ordered
                    answer = v["parsed_answer"][0]  # assume each answer is the same
                    correct_each_question_major[i] = verify(major_prediction, answer)

            # print(correct_each_test)
            correct_ratio_each_test = [c / n for c in correct_each_question]
            print(f"pass@1: {sum(correct_ratio_each_test) / total * 100:.2f}%")
            print(
                f"pass@{n}: {sum([1 - (1 - c/ n) ** n for c in correct_each_question]) / total * 100:.2f}%"
            )

            # print(correct_each_time)
            correct_ratio_each_time = [c / total for c in correct_each_run]
            print(f"max_acc: {max(correct_ratio_each_time) * 100:.2f}%")
            print(f"avg_acc: {sum(correct_ratio_each_time) / n * 100:.2f}%")
            print(f"min_acc: {min(correct_ratio_each_time) * 100:.2f}%")

            # print(correct_each_test_major)
            print(f"cons@{n}: {sum(correct_each_question_major) / total * 100:.2f}%")
            print()
