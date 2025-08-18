from datasets import Dataset, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.registry import ICL_PROMPT_TEMPLATES, LOAD_DATASET
from opencompass.utils import get_data_path

from .zero_eval import (
    extract_first_complete_json,
    extract_values_from_json,
    generate_choice_string,
)


# adapted from https://github.com/WildEval/ZeroEval/blob/8c1485edf12c6efb5f69135a562927c5ad484059/src/evaluation/mcqa_eval.py
@LOAD_DATASET.register_module()
class MMLUReduxZeroEvalDataset(BaseDataset):
    @staticmethod
    def load(path: str, num_examples: int | None = None) -> Dataset:
        path = get_data_path(path)
        dataset = load_dataset(path, name="mmlu-redux", split="test")
        # restrict to a subset of the data for debugging
        if num_examples is not None:
            shuffled_dataset = dataset.shuffle(seed=3407)
            dataset = shuffled_dataset.select(
                range(min(num_examples, len(shuffled_dataset)))
            )
        return dataset


@ICL_PROMPT_TEMPLATES.register_module()
class MMLUReduxZeroEvalPromptTemplate(PromptTemplate):
    def generate_item(self, item: dict, *args, **kwargs) -> dict:
        if "choices" in item and isinstance(item["choices"], list):
            # create a copy to avoid modifying the original item dict in-place
            item = item.copy()
            item["choices"] = generate_choice_string(item["choices"])
        return super().generate_item(item, *args, **kwargs)


class MMLUReduxZeroEvalEvaluator(BaseEvaluator):
    def score(self, predictions, references, test_set: Dataset) -> dict:
        if len(predictions) != len(references):
            return {"error": "Predictions and references must have the same length"}

        solved_examples = 0
        num_total_examples = len(predictions)
        no_answer = 0
        reason_lens = []
        details = []

        for prediction_str, reference, sample in zip(predictions, references, test_set):
            # Read and Parse the prediction from model output
            prediction_json = extract_first_complete_json(prediction_str)
            if prediction_json is None or "answer" not in prediction_json:
                prediction_json = extract_values_from_json(
                    prediction_str, allow_no_quotes=True
                )

            # Check if a valid answer can be extracted
            if (
                prediction_json is None
                or "answer" not in prediction_json
                or prediction_json.get("answer") is None
                or str(prediction_json.get("answer")).strip() == ""
            ):
                no_answer += 1
                details.append(
                    {
                        "prediction": prediction_str,
                        "reference": reference,
                        "correct": False,
                        "error": "No answer found",
                    }
                )
                continue

            # Extract details if an answer is found
            reason = prediction_json.get("reasoning", "")
            model_answer = prediction_json.get("answer")
            correct_answer = sample["correct_answer"]
            try:
                index_of_correct_answer = sample["choices"].index(correct_answer)
                label_of_correct_answer = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[
                    index_of_correct_answer
                ]
            except ValueError:
                details.append(
                    {
                        "prediction": prediction_str,
                        "reference": reference,
                        "correct": False,
                        "error": "Correct answer not in choices",
                    }
                )
                continue

            # Compare and determine correctness
            correct = (
                model_answer == label_of_correct_answer
                or f"{label_of_correct_answer})" in model_answer
            )
            if correct:
                solved_examples += 1

            # Append results for this item
            reason_lens.append(len(reason))
            details.append(
                {
                    "prediction": prediction_str,
                    "reference": reference,
                    "reasoning": reason,
                    "model_answer": model_answer,
                    "correct_answer": label_of_correct_answer,
                    "correct": correct,
                }
            )

        # Get final metrics
        acc = (
            solved_examples / num_total_examples * 100 if num_total_examples > 0 else 0
        )
        no_answer_rate = (
            no_answer / num_total_examples * 100 if num_total_examples > 0 else 0
        )
        reason_lens_avg = sum(reason_lens) / len(reason_lens) if reason_lens else 0.0
        print(f"Acc: {acc:.2f}")
        print(f"No answer: {no_answer_rate:.2f}")
        print(f"Total: {num_total_examples}")
        print(f"Reason Lens: {reason_lens_avg:.2f}")
        return {"accuracy": acc, "details": details}
