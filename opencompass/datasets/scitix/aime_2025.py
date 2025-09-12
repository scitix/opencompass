import os
import re

from datasets import Dataset, concatenate_datasets, load_dataset
from math_verify import parse, verify

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from .math import strip_string
from .simple_evals import ANSWER_PATTERN


@LOAD_DATASET.register_module()
class AIME2025Dataset(BaseDataset):
    @staticmethod
    def load(
        path: str, n_repeats: int = 1, num_examples: int | None = None, seed: int = 3407
    ) -> Dataset:
        path = get_data_path(path)
        subsets_names = ["AIME2025-I", "AIME2025-II"]
        subsets = [
            load_dataset(path, name=name, split="test") for name in subsets_names
        ]
        dataset = concatenate_datasets(subsets)

        def strip_sample(sample):
            sample["question"] = strip_string(sample["question"])
            sample["answer"] = strip_string(sample["answer"])
            return sample

        dataset = dataset.map(strip_sample, num_proc=os.cpu_count())

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


@TEXT_POSTPROCESSORS.register_module("aime_2025")
def aime_2025_postprocess(response_text: str) -> str:
    match = re.search(ANSWER_PATTERN, response_text)
    return match.group(1).strip() if match else response_text


class AIME2025Evaluator(BaseEvaluator):
    def __init__(self, pred_postprocessor=None):
        if pred_postprocessor is None:
            pred_postprocessor = {"type": "aime_2025"}
        super().__init__(pred_postprocessor=pred_postprocessor)

    def score(self, predictions, references) -> dict:
        if len(predictions) != len(references):
            return {"error": "Predictions and the references must have the same length"}

        correct = 0
        details = []
        for prediction, reference in zip(predictions, references):
            # surround with `$` to ensure the latex parser works
            prediction_with_env = f"${prediction}$"
            reference_with_env = f"${reference}$"
            parsed_prediction = parse(prediction_with_env)
            parsed_reference = parse(reference_with_env)
            is_correct = verify(parsed_prediction, parsed_reference)
            if is_correct:
                correct += 1

            details.append(
                {
                    "prediction": prediction,
                    "answer": reference,
                    "parsed_prediction": str(parsed_prediction),
                    "parsed_answer": str(parsed_reference),
                    "correct": is_correct,
                }
            )

        score = 100 * correct / len(predictions) if predictions else 0.0
        return {"score": score, "details": details}
