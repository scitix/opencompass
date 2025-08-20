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
class AIME2024Dataset(BaseDataset):
    @staticmethod
    def load(
        path: str, n_repeats: int = 1, num_examples: int | None = None, seed: int = 3407
    ) -> Dataset:
        path = get_data_path(path)
        dataset = load_dataset(path, split="train")

        def strip_sample(sample):
            sample["problem"] = strip_string(sample["problem"])
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


@TEXT_POSTPROCESSORS.register_module("aime_2024")
def aime_2024_postprocess(response_text: str) -> str:
    match = re.search(ANSWER_PATTERN, response_text)
    return match.group(1).strip() if match else response_text


class AIME2024Evaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(pred_postprocessor={"type": "aime_2024"})

    def score(self, predictions, references) -> dict:
        print(len(predictions), len(references))
        print(predictions)
        print(references)

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
