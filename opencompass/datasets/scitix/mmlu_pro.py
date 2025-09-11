import os
from collections import defaultdict

from datasets import Dataset, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

CHOICES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
]


@LOAD_DATASET.register_module()
class MMLUProDataset(BaseDataset):
    @staticmethod
    def load(
        path: str, n_repeats: int = 1, num_examples: int | None = None, seed: int = 3407
    ) -> Dataset:
        path = get_data_path(path)
        dataset = load_dataset(path, split="test")

        def preprocess_sample(sample):
            options_str = ""
            for i, opt in enumerate(sample["options"]):
                if opt == "N/A":
                    continue
                option = "{}. {}\n".format(CHOICES[i], opt)
                options_str += option
            sample["options_str"] = options_str.strip()
            return sample

        dataset = dataset.map(preprocess_sample, num_proc=os.cpu_count())

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


class MMLUProEvaluator(BaseEvaluator):
    def score(self, predictions, references, test_set: Dataset) -> dict:
        if len(predictions) != len(references):
            return {"error": "Predictions and references must have the same length"}

        correct = 0
        details = []
        category_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
        for prediction, reference, sample in zip(predictions, references, test_set):
            category = sample.get("category", "unknown")

            is_correct = prediction == reference
            if is_correct:
                correct += 1
                category_metrics[category]["correct"] += 1
            category_metrics[category]["total"] += 1

            details.append(
                {
                    "prediction": prediction,
                    "answer": reference,
                    "correct": is_correct,
                    "category": category,
                }
            )

        score = 100 * correct / len(predictions) if predictions else 0.0
        results = {"score": score, "details": details}
        for category, metrics in category_metrics.items():
            category_score = (
                100 * metrics["correct"] / metrics["total"]
                if metrics["total"] > 0
                else 0.0
            )
            results[f"score_{category}"] = category_score
        return results
