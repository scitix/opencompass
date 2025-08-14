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
    def load(path: str, n_repeats: int = 4, num_examples: int | None = None) -> Dataset:
        path = get_data_path(path)
        data_files = {"test": path}
        dataset = load_dataset("csv", data_files=data_files, split="test")

        rng = random.Random(0)
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
    extracted_answer = match.group(1) if match else ""
    return extracted_answer


class GPQADiamondSimpleEvalsEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(pred_postprocessor={"type": "gpqa_diamond_simple_evals"})

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {"error": "Predictions and references must have the same length"}

        details = []
        correct = 0
        for pred, ref in zip(predictions, references):
            is_correct = pred == ref
            if is_correct:
                correct += 1
            details.append(
                {
                    "prediction": pred,
                    "answer": ref,
                    "correct": is_correct,
                }
            )

        score = 100 * correct / len(predictions) if predictions else 0.0
        return {"score": score, "details": details}

    def evaluate(self, k: int, n: int, original_dataset: Dataset, **score_kwargs):
        base_results = super().evaluate(k, n, original_dataset, **score_kwargs)

        details = base_results.get("details", [])
        if not details:
            return base_results

        # `original_dataset` will repeat `n` times
        num_samples = len(details)
        if num_samples == 0:
            return base_results

        num_fully_correct = 0
        for i, detail in enumerate(details):
            is_all_n_correct = all(detail.get("correct", []))
            if is_all_n_correct:
                num_fully_correct += 1
            detail["all_n_correct"] = is_all_n_correct

        new_score = 100 * num_fully_correct / num_samples if num_samples > 0 else 0
        base_results["all_n_correct_score"] = new_score

        return base_results
