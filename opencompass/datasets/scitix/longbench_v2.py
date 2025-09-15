from datasets import Dataset, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path
from opencompass.utils.text_postprocessors import first_option_postprocess


@LOAD_DATASET.register_module()
class LongBenchV2Dataset(BaseDataset):
    @staticmethod
    def load(
        path: str, n_repeats: int = 1, num_examples: int | None = None, seed: int = 0
    ) -> Dataset:
        path = get_data_path(path)
        dataset = load_dataset(path, split="train")

        # restrict to a subset of the data for debugging
        if num_examples is not None:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            shuffled_dataset = dataset.shuffle(seed=seed)
            dataset = shuffled_dataset.select(
                range(min(num_examples, len(shuffled_dataset)))
            )

        # repeat examples
        if n_repeats > 1:
            original_indices = list(range(len(dataset)))
            repeated_indices = original_indices * n_repeats
            dataset = dataset.select(repeated_indices)

        return dataset


@TEXT_POSTPROCESSORS.register_module("longbench_v2")
def longbench_v2_postprocess(response_text: str) -> str:
    return first_option_postprocess(response_text, options="ABCD", cushion=True)


class LongBenchV2Evaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(pred_postprocessor={"type": "longbench_v2"})

    def score(self, predictions, references, test_set: Dataset) -> dict:
        if len(predictions) != len(references):
            return {"error": "Predictions and references must have the same length"}

        metrics = {
            "total": {"correct": 0, "total": 0},
            "difficulty": {
                "easy": {"correct": 0, "total": 0},
                "hard": {"correct": 0, "total": 0},
                "unknown": {"correct": 0, "total": 0},
            },
            "length": {
                "short": {"correct": 0, "total": 0},
                "medium": {"correct": 0, "total": 0},
                "long": {"correct": 0, "total": 0},
                "unknown": {"correct": 0, "total": 0},
            },
        }
        details = []
        for prediction, reference, sample in zip(predictions, references, test_set):
            is_correct = prediction == reference
            difficulty = sample.get("difficulty", "unknown")
            length = sample.get("length", "unknown")

            metrics["total"]["total"] += 1
            metrics["difficulty"][difficulty]["total"] += 1
            metrics["length"][length]["total"] += 1
            if is_correct:
                metrics["total"]["correct"] += 1
                metrics["difficulty"][difficulty]["correct"] += 1
                metrics["length"][length]["correct"] += 1

            details.append(
                {
                    "prediction": prediction,
                    "answer": reference,
                    "is_correct": is_correct,
                    "difficulty": sample.get("difficulty", "unknown"),
                    "length": sample.get("length", "unknown"),
                }
            )

        results = {
            "accuracy": metrics["total"]["correct"] / metrics["total"]["total"] * 100,
            "details": details,
        }
        for diff in ["easy", "hard"]:
            if metrics["difficulty"][diff]["total"] > 0:
                acc = (
                    metrics["difficulty"][diff]["correct"]
                    / metrics["difficulty"][diff]["total"]
                    * 100
                )
                results[f"accuracy_{diff}"] = acc
        for length in ["short", "medium", "long"]:
            if metrics["length"][length]["total"] > 0:
                acc = (
                    metrics["length"][length]["correct"]
                    / metrics["length"][length]["total"]
                    * 100
                )
                results[f"accuracy_{length}"] = acc
        return results
