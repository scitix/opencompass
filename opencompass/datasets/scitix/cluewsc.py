import os

from datasets import Dataset, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path
from opencompass.utils.text_postprocessors import first_option_postprocess


@LOAD_DATASET.register_module()
class CLUEWSCDataset(BaseDataset):
    @staticmethod
    def load(
        path: str, n_repeats: int = 1, num_examples: int | None = None, seed: int = 3407
    ) -> Dataset:
        data_dir = get_data_path(path)
        path = os.path.join(data_dir, "dev.json")
        data_files = {"test": path}
        dataset = load_dataset("json", data_files=data_files, split="test")

        def _preprocess_sample(sample):
            target = sample["target"]
            sample["span1"] = target["span1_text"]
            sample["span2"] = target["span2_text"]
            if sample["label"] == "true":
                sample["answer"] = "A"
            elif sample["label"] == "false":
                sample["answer"] = "B"
            else:
                sample["answer"] = "C"
            return sample

        dataset = dataset.map(_preprocess_sample, num_proc=os.cpu_count())

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


@TEXT_POSTPROCESSORS.register_module("cluewsc")
def cluewsc_postprocess(response_text: str) -> str:
    return first_option_postprocess(response_text, options="ABC", cushion=True)


class CLUEWSCEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(pred_postprocessor={"type": "cluewsc"})

    def score(self, predictions, references) -> dict:
        if len(predictions) != len(references):
            return {"error": "Predictions and the references must have the same length"}

        correct = 0
        details = []
        for prediction, reference in zip(predictions, references):
            is_correct = prediction == reference
            if is_correct:
                correct += 1

            details.append(
                {
                    "prediction": prediction,
                    "reference": reference,
                    "correct": is_correct,
                }
            )

        score = 100 * correct / len(predictions) if predictions else 0.0
        return {"score": score, "details": details}
