import json
import os
from collections import defaultdict

import numpy as np
from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .t_eval import ResponseDataSample, format_load


# at calling - instruct
# after calling - review
@LOAD_DATASET.register_module()
class CompassBenchTEvalAfterCallingDataset(BaseDataset):
    @staticmethod
    def load(
        path: str,
        lang: str = "cn",
        n_repeats: int = 1,
        num_examples: int | None = None,
        seed: int = 3407,
        legacy_model: bool = False,
    ) -> Dataset:
        suffix = "_zh" if lang == "cn" else ""
        data_dir = get_data_path(path)
        path = os.path.join(data_dir, "data", f"review_str_v2{suffix}.json")

        preprocessed_data = []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for v in data.values():
            dialog = v.get("origin_prompt", [])
            if legacy_model:
                # for legacy models, replace `function` with `user`
                for msg in dialog:
                    if msg["role"] == "function":
                        msg["role"] = "user"
            dialog.append({"role": "assistant", "content": ""})
            preprocessed_data.append(
                {
                    "template": v.get("template", {}),
                    "meta_data": v.get("meta_data", {}),
                    "dialog": dialog,
                    "ground_truth": v.get("ground_truth", {}),
                }
            )
        dataset = Dataset.from_list(preprocessed_data)

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


class CompassBenchTEvalAfterCallingEvaluator(BaseEvaluator):
    def score(self, predictions, references, test_set: Dataset) -> dict:
        if len(predictions) != len(references):
            return {"error": "Predictions and references must have the same length"}

        details = []
        results_list = []
        for prediction, reference, sample in zip(predictions, references, test_set):
            resp_data_sample = self._process_response(
                {
                    "template": sample["template"],
                    "prediction": prediction,
                    "ground_truth": reference,
                    "meta_data": sample["meta_data"],
                }
            )
            metrics_result = self._evaluate(resp_data_sample)
            results_list.append(metrics_result)

            details.append(
                {
                    "prediction": prediction,
                    "answer": reference,
                    **metrics_result,
                }
            )
        return {**self._post_process(results_list), "details": details}

    def _process_response(self, datum: dict) -> ResponseDataSample:
        template = datum["template"]
        pred_data = datum["prediction"]
        gt_data = datum["ground_truth"]["answer"]
        meta_data = datum["meta_data"]

        if meta_data["response_format"] == "json":
            pred_data = self.json_format_parse(pred_data)
        else:
            pred_data = pred_data[pred_data.find(":") + 1 :]
            pred_data = pred_data.strip()
            if len(pred_data) > 0 and pred_data[0] in ["A", "B", "C", "D", "E"]:
                pred_data = pred_data[0]
            else:
                pred_data = None

        return ResponseDataSample(
            template=template, pred=pred_data, gt=gt_data, meta_data=meta_data
        )

    def _evaluate(self, data_sample: ResponseDataSample) -> dict:
        metrics_result = dict(
            parse_rate=0,
            review_quality=0,
        )

        pred_data = data_sample.pred
        if pred_data is not None:
            metrics_result["review_quality"] = (
                1.0 if pred_data == data_sample.gt else 0.0
            )
            metrics_result["parse_rate"] = 1.0
        return metrics_result

    def _json_format_parse(self, pred_data: str) -> dict | None:
        try:
            data = format_load(pred_data)
        except Exception:
            return None
        try:
            new_data = dict()
            new_data["review"] = data["is_finished"]
            assert new_data["review"] in [True, False]
        except Exception:
            return None
        return new_data

    def _post_process(self, results_list: list[dict]) -> dict:
        results_dict = defaultdict(list)
        {results_dict[key].append(sub[key]) for sub in results_list for key in sub}
        metric_list = ["parse_rate", "review_quality"]
        for metric in metric_list:
            score = np.mean(results_dict[metric]) * 100
            results_dict[metric] = np.round(score, decimals=2)
        return results_dict
