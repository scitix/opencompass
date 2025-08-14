import random
import re

from datasets import Dataset, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from .simple_evals import (
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
    normalize_extracted_answer,
    normalize_response,
)


# adapted from https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/mmlu_eval.py
@LOAD_DATASET.register_module()
class MMLUSimpleEvalsDataset(BaseDataset):
    @staticmethod
    def load(path: str, num_examples: int | None = None) -> Dataset:
        path = get_data_path(path)
        data_files = {"test": path}
        dataset = load_dataset("csv", data_files=data_files, split="test")
        # restrict to a subset of the data for debugging
        if num_examples is not None:
            rng = random.Random(0)
            indices = rng.sample(range(len(dataset)), min(num_examples, len(dataset)))
            dataset = dataset.select(indices)
        return dataset


@TEXT_POSTPROCESSORS.register_module("mmlu_simple_evals")
def mmlu_postprocess(response_text: str) -> str:
    response_text = normalize_response(response_text)
    extracted_answer = ""
    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        match = re.search(regex, response_text)
        if match:
            extracted_answer = normalize_extracted_answer(match.group(1))
            break
    return extracted_answer


class MMLUSimpleEvalsEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(pred_postprocessor={"type": "mmlu_simple_evals"})

        self.subject2category = {
            "abstract_algebra": "stem",
            "anatomy": "other",
            "astronomy": "stem",
            "business_ethics": "other",
            "clinical_knowledge": "other",
            "college_biology": "stem",
            "college_chemistry": "stem",
            "college_computer_science": "stem",
            "college_mathematics": "stem",
            "college_medicine": "other",
            "college_physics": "stem",
            "computer_security": "stem",
            "conceptual_physics": "stem",
            "econometrics": "social_sciences",
            "electrical_engineering": "stem",
            "elementary_mathematics": "stem",
            "formal_logic": "humanities",
            "global_facts": "other",
            "high_school_biology": "stem",
            "high_school_chemistry": "stem",
            "high_school_computer_science": "stem",
            "high_school_european_history": "humanities",
            "high_school_geography": "social_sciences",
            "high_school_government_and_politics": "social_sciences",
            "high_school_macroeconomics": "social_sciences",
            "high_school_mathematics": "stem",
            "high_school_microeconomics": "social_sciences",
            "high_school_physics": "stem",
            "high_school_psychology": "social_sciences",
            "high_school_statistics": "stem",
            "high_school_us_history": "humanities",
            "high_school_world_history": "humanities",
            "human_aging": "other",
            "human_sexuality": "social_sciences",
            "international_law": "humanities",
            "jurisprudence": "humanities",
            "logical_fallacies": "humanities",
            "machine_learning": "stem",
            "management": "other",
            "marketing": "other",
            "medical_genetics": "other",
            "miscellaneous": "other",
            "moral_disputes": "humanities",
            "moral_scenarios": "humanities",
            "nutrition": "other",
            "philosophy": "humanities",
            "prehistory": "humanities",
            "professional_accounting": "other",
            "professional_law": "humanities",
            "professional_medicine": "other",
            "professional_psychology": "social_sciences",
            "public_relations": "social_sciences",
            "security_studies": "social_sciences",
            "sociology": "social_sciences",
            "us_foreign_policy": "social_sciences",
            "virology": "other",
            "world_religions": "humanities",
        }

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
            # add subject and category to each detail
            original_sample = original_dataset[i]
            subject = original_sample.get("Subject", "")
            detail["subject"] = subject
            detail["category"] = self.subject2category.get(subject, "other")

            is_all_n_correct = all(detail.get("correct", []))
            if is_all_n_correct:
                num_fully_correct += 1
            detail["all_n_correct"] = is_all_n_correct

        new_score = 100 * num_fully_correct / num_samples if num_samples > 0 else 0
        base_results["all_n_correct_score"] = new_score

        return base_results
