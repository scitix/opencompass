import random
import re
from collections import defaultdict

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
    def load(
        path: str, n_repeats: int = 1, num_examples: int | None = None, seed: int = 0
    ) -> Dataset:
        path = get_data_path(path)
        data_files = {"test": path}
        dataset = load_dataset("csv", data_files=data_files, split="test")

        # restrict to a subset of the data for debugging
        if num_examples is not None:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            rng = random.Random(seed)
            indices = rng.sample(range(len(dataset)), min(num_examples, len(dataset)))
            dataset = dataset.select(indices)

        # repeat examples
        if n_repeats > 1:
            original_indices = list(range(len(dataset)))
            repeated_indices = original_indices * n_repeats
            dataset = dataset.select(repeated_indices)

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

    def score(self, predictions, references, test_set: Dataset) -> dict:
        if len(predictions) != len(references):
            return {"error": "Predictions and references must have the same length"}

        correct = 0
        details = []
        category_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
        for prediction, reference, sample in zip(predictions, references, test_set):
            subject = sample.get("Subject", "unknown")
            category = self.subject2category.get(subject, "other")

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
                    "subject": subject,
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
