import re

from datasets import Dataset, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path


# adapted from https://github.com/OpenStellarTeam/ChineseSimpleQA/blob/8b1aa9ebc5b91092356eb61f20f355c5993e6ce5/judge/chinese_simpleqa_easy.py
@LOAD_DATASET.register_module()
class CSimpleQADataset(BaseDataset):
    @staticmethod
    def load(
        path: str, n_repeats: int = 1, num_examples: int | None = None, seed: int = 0
    ) -> Dataset:
        path = get_data_path(path)
        dataset = load_dataset(path, split="train")

        # restrict to a subset of the data for debugging
        if num_examples is not None:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            shuffled_dataset = dataset.shuffle(seed)
            dataset = shuffled_dataset.select(
                range(min(num_examples, len(shuffled_dataset)))
            )

        # repeat examples
        if n_repeats > 1:
            original_indices = list(range(len(dataset)))
            repeated_indices = original_indices * n_repeats
            dataset = dataset.select(repeated_indices)

        return dataset


def c_simpleqa_llmjudge_postprocess(output: dict, output_path: str) -> dict:
    details = []
    judged_answers = []

    for k, v in output.items():
        # 'prediction' here is the judge's raw output
        judgement = v.get("prediction", "")
        match = re.search(r"(A|B|C)", judgement)
        grade_letter = match.group(0) if match else "C"
        judged_answers.append(grade_letter)

        details.append(
            {
                "judgement": judgement,
                "answer": v.get("gold", ""),
                "grade": grade_letter,
            }
        )

    count = len(judged_answers)
    if count == 0:
        return {
            "accuracy": 0,
            "accuracy_given_attempted": 0,
            "f1": 0,
            "details": [],
        }

    is_correct_count = judged_answers.count("A")
    is_incorrect_count = judged_answers.count("B")
    is_not_attempted = judged_answers.count("C")

    is_correct_frac = is_correct_count / count
    is_incorrect_frac = is_incorrect_count / count
    is_not_attempted_frac = is_not_attempted / count

    is_given_attempted_frac = is_correct_frac + is_incorrect_frac
    accuracy_given_attempted = (
        is_correct_frac / is_given_attempted_frac if is_given_attempted_frac > 0 else 0
    )

    f1 = (
        2
        * accuracy_given_attempted
        * is_correct_frac
        / (accuracy_given_attempted + is_correct_frac)
        if (accuracy_given_attempted + is_correct_frac) > 0
        else 0
    )

    aggregate_metrics = {
        "is_correct": is_correct_frac,
        "is_incorrect": is_incorrect_frac,
        "is_not_attempted": is_not_attempted_frac,
        "is_given_attempted": is_given_attempted_frac,
        "accuracy_given_attempted": accuracy_given_attempted,
    }
    print("AGGREGATE METRICS")
    print(aggregate_metrics)
    print("##################")
    print(f"Accuracy Given Attempted: {accuracy_given_attempted:.3f}")
    print(f"F1 Score: {f1:.3f}")

    return {
        "accuracy": is_correct_frac * 100,
        "accuracy_given_attempted": accuracy_given_attempted * 100,
        "f1": f1 * 100,
        "details": details,
    }
