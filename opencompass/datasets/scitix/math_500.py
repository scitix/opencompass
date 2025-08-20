import os
import re

from datasets import Dataset, load_dataset
from math_verify import parse, verify

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from .math import strip_string
from .simple_evals import ANSWER_PATTERN


@LOAD_DATASET.register_module()
class MATH500Dataset(BaseDataset):
    @staticmethod
    def load(path: str, num_examples: int | None = None) -> Dataset:
        path = get_data_path(path)
        dataset = load_dataset(path, split="test")

        def strip_sample(sample):
            sample["problem"] = strip_string(sample["problem"])
            sample["answer"] = strip_string(sample["answer"])
            return sample

        dataset = dataset.map(strip_sample, num_proc=os.cpu_count())

        # restrict to a subset of the data for debugging
        if num_examples is not None:
            shuffled_dataset = dataset.shuffle(seed=3407)
            dataset = shuffled_dataset.select(
                range(min(num_examples, len(shuffled_dataset)))
            )

        return dataset


@TEXT_POSTPROCESSORS.register_module("math_500")
def math_500_postprocess(response_text: str) -> str:
    match = re.search(ANSWER_PATTERN, response_text)
    return match.group(1).strip() if match else response_text


class MATH500Evaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(pred_postprocessor={"type": "math_500"})

    def score(self, predictions, references, test_set: Dataset) -> dict:
        if len(predictions) != len(references):
            return {"error": "Predictions and references must have the same length"}

        correct = 0
        details = []
        for prediction, reference, sample in zip(predictions, references, test_set):
            subject = sample.get("subject", "unknown")
            level = sample.get("level", 0)

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
                    "subject": subject,
                    "level": level,
                }
            )

        score = 100 * correct / len(predictions) if predictions else 0.0
        return {"score": score, "details": details}
