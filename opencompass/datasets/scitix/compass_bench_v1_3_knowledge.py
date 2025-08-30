import os
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


@LOAD_DATASET.register_module()
class CompassBenchKnowledgeDataset(BaseDataset):
    @staticmethod
    def load(
        path: str, n_repeats: int = 1, num_examples: int | None = None, seed: int = 3407
    ) -> Dataset:
        data_dir = get_data_path(path)
        path = os.path.join(data_dir, "knowledge", "single_choice_cn.jsonl")
        data_files = {"test": path}
        dataset = load_dataset("json", data_files=data_files, split="test")

        def preprocess_sample(sample):
            options = sample["options"]
            assert len(options) == 4, "Expected 4 options per question"
            sample["A"] = options[0]
            sample["B"] = options[1]
            sample["C"] = options[2]
            sample["D"] = options[3]
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


@TEXT_POSTPROCESSORS.register_module("compass_bench_knowledge_v1_3")
def compass_bench_knowledge_postprocess(response_text: str) -> str:
    response_text = normalize_response(response_text)
    extracted_answer = ""
    for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        match = re.search(regex, response_text)
        if match:
            extracted_answer = normalize_extracted_answer(match.group(1))
            break
    return extracted_answer


class CompassBenchKnowledgeEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(pred_postprocessor={"type": "compass_bench_knowledge_v1_3"})

    def score(self, predictions, references) -> dict:
        if len(predictions) != len(references):
            return {"error": "Predictions and references must have the same length"}

        correct = 0
        details = []
        for prediction, reference in zip(predictions, references):
            is_correct = prediction == reference
            if is_correct:
                correct += 1

            details.append(
                {
                    "prediction": prediction,
                    "answer": reference,
                    "correct": is_correct,
                }
            )

        score = 100 * correct / len(predictions) if predictions else 0.0
        return {"score": score, "details": details}
