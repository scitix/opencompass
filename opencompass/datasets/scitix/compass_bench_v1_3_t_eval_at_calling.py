import ast
import json
import os
from collections import defaultdict

import numpy as np
from datasets import Dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .t_eval import ResponseDataSample, format_load, parse_string


# at calling - instruct
@LOAD_DATASET.register_module()
class CompassBenchTEvalAtCallingDataset(BaseDataset):
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
        path = os.path.join(data_dir, "data", f"instruct_v2{suffix}.json")

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
                    # convert to str to avoid pyarrow type error
                    "ground_truth": json.dumps(v.get("ground_truth", {})),
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


class CompassBenchTEvalAtCallingEvaluator(BaseEvaluator):
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
                    "ground_truth": json.loads(reference),
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
        # Dict with keyword-only arguments.
        template = datum["template"]
        # Generated response.
        pred_data = datum["prediction"]
        # Response of ground truth.
        gt_data = datum["ground_truth"]
        meta_data = datum["meta_data"]

        return ResponseDataSample(
            template=template, pred=pred_data, gt=gt_data, meta_data=meta_data
        )

    def _evaluate(self, data_sample: ResponseDataSample) -> dict:
        metrics_result = dict()
        response_format = data_sample.meta_data["response_format"]
        if response_format == "json":
            pred_data = self._json_format_parse(data_sample)
        else:
            pred_data = self._string_format_parse(data_sample)

        if pred_data is None:
            # directly set to 0 for all metrics
            metrics_result[f"{response_format}_format_metric"] = 0
            metrics_result[f"{response_format}_args_em_metric"] = 0
            return metrics_result

        # Exact matching
        metrics_result[f"{response_format}_format_metric"] = 1
        metrics_result[f"{response_format}_args_em_metric"] = (
            self._compute_args_em_metric(
                gt_action=data_sample.gt["action"],
                pred_action=pred_data["action"],
                gt_args=data_sample.gt["args"],
                pred_args=pred_data["args"],
            )
        )
        return metrics_result

    def _compute_args_em_metric(
        self, gt_action, pred_action, gt_args, pred_args
    ) -> float:
        cnt = 0.0
        if gt_action == pred_action:
            cnt += 1.0
        num_args = len(gt_args) + 1  # 1 means action name match
        for gt_key in gt_args:
            pred_val = pred_args.get(gt_key, "")
            if pred_val == gt_args[gt_key]:
                cnt += 1.0
        return cnt / num_args

    def _string_format_parse(self, data_sample: ResponseDataSample) -> dict | None:
        pred_data = data_sample.pred
        template = data_sample.template
        thought_start = template["thought_start"]
        thought_end = template["thought_end"]
        action_start = template["action_start"]
        action_end = template["action_end"]
        args_start = template["args_start"]
        args_end = template["args_end"]

        parse_template = (
            thought_start
            + "{thought}"
            + thought_end
            + action_start
            + "{action}"
            + action_end
            + args_start
            + "{args}"
            + args_end
        )
        res = parse_string(parse_template, pred_data, allow_newline=True)
        try:
            if res is not None:
                args = ast.literal_eval(res["args"].strip())
                res["args"] = args if isinstance(args, dict) else {}
                res["action"] = res["action"].strip()
            return res
        except Exception:
            return dict(
                thought=res["thought"], action=res["action"].strip(), args=dict()
            )

    def _json_format_parse(self, data_sample: ResponseDataSample) -> dict | None:
        try:
            pred_data = format_load(data_sample.pred)
            template = data_sample.template
            new_data = dict()
            new_data["thought"] = pred_data[template["thought"]]
            new_data["action"] = pred_data[template["action"]]
            args = pred_data[template["args"]]
            new_data["args"] = args if isinstance(args, dict) else {}
        except Exception:
            return None

        return new_data

    def _post_process(self, results_list: list[dict]) -> dict:
        # list of dict to dict of list
        results_dict = defaultdict(list)
        {results_dict[key].append(sub[key]) for sub in results_list for key in sub}
        metric_list = [
            "json_format_metric",
            "json_args_em_metric",
            "string_format_metric",
            "string_args_em_metric",
        ]
        for metric in metric_list:
            score = np.mean(results_dict[metric]) * 100
            results_dict[metric] = np.round(score, decimals=2)
        return results_dict
