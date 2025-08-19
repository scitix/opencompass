from collections import defaultdict

from datasets import Dataset, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .instruction_following_eval.evaluation_lib import (
    InputExample,
    OutputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)


@LOAD_DATASET.register_module()
class IFEvalDataset(BaseDataset):
    @staticmethod
    def load(path: str, num_examples: int | None = None) -> Dataset:
        path = get_data_path(path)
        dataset = load_dataset(path, split="train")
        # restrict to a subset of the data for debugging
        if num_examples is not None:
            shuffled_dataset = dataset.shuffle(seed=3407)
            dataset = shuffled_dataset.select(
                range(min(num_examples, len(shuffled_dataset)))
            )
        return dataset


# adapted from https://github.com/google-research/google-research/blob/89dbaf2657e70797f2dea7a6acac5a808da8ff4d/instruction_following_eval/evaluation_lib.py
def get_report(outputs: list[OutputExample]) -> dict:
    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0

    tier0_total = defaultdict(int)
    tier0_correct = defaultdict(int)

    tier1_total = defaultdict(int)
    tier1_correct = defaultdict(int)

    for example in outputs:
        follow_instruction_list = example.follow_instruction_list
        instruction_id_list = example.instruction_id_list

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

        for instruction_id, followed_or_not in zip(
            instruction_id_list, follow_instruction_list
        ):
            instruction_id = instruction_id.split(":")[0]
            tier0_total[instruction_id] += 1
            if followed_or_not:
                tier0_correct[instruction_id] += 1

        for instruction_id, followed_or_not in zip(
            instruction_id_list, follow_instruction_list
        ):
            tier1_total[instruction_id] += 1
            if followed_or_not:
                tier1_correct[instruction_id] += 1

    print(f"prompt-level: {prompt_correct / prompt_total}")
    print(f"instruction-level: {instruction_correct / instruction_total}")
    print()
    for instruction_id in sorted(tier0_total.keys()):
        accuracy = tier0_correct[instruction_id] / tier0_total[instruction_id]
        print(f"{instruction_id} {accuracy}")
    print()
    for instruction_id in sorted(tier1_total.keys()):
        accuracy = tier1_correct[instruction_id] / tier1_total[instruction_id]
        print(f"{instruction_id} {accuracy}")
    return {
        "prompt-level": prompt_correct / prompt_total,
        "instruction-level": instruction_correct / instruction_total,
    }


class IFEvalEvaluator(BaseEvaluator):
    def score(self, predictions, references, test_set: Dataset) -> dict:
        if len(predictions) != len(test_set):
            return {"error": "Predictions and the test set must have the same length"}

        # avoid hf datasets underlying Arrow sparse struct problem
        def _clean_kwargs(kwargs):
            return [{k: v for k, v in d.items() if v is not None} for d in kwargs]

        inputs = [
            InputExample(
                key=sample["key"],
                instruction_id_list=sample["instruction_id_list"],
                prompt=sample["prompt"],
                kwargs=_clean_kwargs(sample["kwargs"]),
            )
            for sample in test_set
        ]
        prompt_to_response = {
            sample["prompt"]: pred for pred, sample in zip(predictions, test_set)
        }

        results = {}
        for func, grade in [
            (test_instruction_following_strict, "strict"),
            (test_instruction_following_loose, "loose"),
        ]:
            outputs = [func(inp, prompt_to_response) for inp in inputs]
            follow_all_instructions = [o.follow_all_instructions for o in outputs]
            accuracy = sum(follow_all_instructions) / len(outputs)
            results[f"{grade}_accuracy"] = accuracy * 100

            report = get_report(outputs)
            results[f"{grade}_prompt_level_accuracy"] = (
                report.get("prompt-level", 0.0) * 100
            )
            results[f"{grade}_instruction_level_accuracy"] = (
                report.get("instruction-level", 0.0) * 100
            )

        # details
        details = []
        for prediction, sample in zip(predictions, test_set):
            details.append(
                {
                    "prompt": sample["prompt"],
                    "prediction": prediction,
                }
            )
        results["details"] = details

        return results
