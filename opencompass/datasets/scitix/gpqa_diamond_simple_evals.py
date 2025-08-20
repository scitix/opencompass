import os
import random
import re

from datasets import Dataset, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from .simple_evals import ANSWER_PATTERN_MULTICHOICE


# adapted from https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/gpqa_eval.py
@LOAD_DATASET.register_module()
class GPQADiamondSimpleEvalsDataset(BaseDataset):
    @staticmethod
    def load(
        path: str, n_repeats: int = 4, num_examples: int | None = None, seed: int = 0
    ) -> Dataset:
        path = get_data_path(path)
        data_files = {"test": path}
        dataset = load_dataset("csv", data_files=data_files, split="test")

        rng = random.Random(seed)
        # restrict to a subset of the data for debugging
        if num_examples is not None:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            shuffled_dataset = dataset.shuffle(seed=rng.randint(0, 2**32 - 1))
            dataset = shuffled_dataset.select(
                range(min(num_examples, len(shuffled_dataset)))
            )

        # repeat examples
        if n_repeats > 1:
            original_indices = list(range(len(dataset)))
            repeated_indices = original_indices * n_repeats
            dataset = dataset.select(repeated_indices)

        def permute_and_format(example):
            choices_list = [
                example["Correct Answer"],
                example["Incorrect Answer 1"],
                example["Incorrect Answer 2"],
                example["Incorrect Answer 3"],
            ]
            permutation = rng.sample(range(4), 4)
            shuffled_choices = [choices_list[i] for i in permutation]
            correct_index = shuffled_choices.index(example["Correct Answer"])
            correct_answer_letter = "ABCD"[correct_index]
            return {
                "Question": example["Question"],
                "A": shuffled_choices[0],
                "B": shuffled_choices[1],
                "C": shuffled_choices[2],
                "D": shuffled_choices[3],
                "Answer": correct_answer_letter,
            }

        dataset = dataset.map(permute_and_format, num_proc=os.cpu_count())
        return dataset


@TEXT_POSTPROCESSORS.register_module("gpqa_diamond_simple_evals")
def gpqa_diamond_postprocess(response_text: str) -> str:
    match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
    return match.group(1) if match else ""


class GPQADiamondSimpleEvalsEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(pred_postprocessor={"type": "gpqa_diamond_simple_evals"})

    def score(self, predictions, references) -> dict:
        if len(predictions) != len(references):
            return {"error": "Predictions and references must have the same length"}

        details = []
        correct = 0
        for prediction, reference in zip(predictions, references):
            is_correct = prediction == reference
            if is_correct:
                correct += 1
            details.append(
                {
                    "prediction": prediction,
                    "answer": reference,
                    "correct": is_correct,
                    "chars": len(prediction),
                }
            )

        score = 100 * correct / len(predictions) if predictions else 0.0
        return {"score": score, "details": details}
