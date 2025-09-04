import json
import os
import time

import numpy as np
from datasets import Dataset
from openai import OpenAI
from sentence_transformers import util

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .t_eval import EMB_PLACEHOLDER, ResponseDataSample, format_load


# before calling - reason, retrieve, understand
# adapted from https://github.com/open-compass/T-Eval/blob/58f22406404d7e2a4f36856a19c7f4dc28a0a5f0/teval/evaluators/reason_retrieve_understand_evaluator.py
@LOAD_DATASET.register_module()
class CompassBenchTEvalBeforeCallingDataset(BaseDataset):
    @staticmethod
    def load(
        path: str,
        lang: str = "cn",
        form: str = "json",
        n_repeats: int = 1,
        num_examples: int | None = None,
        seed: int = 3407,
        legacy_model: bool = False,
    ) -> Dataset:
        suffix = "_zh" if lang == "cn" else ""
        data_dir = get_data_path(path)
        path = os.path.join(
            data_dir, "data", f"reason_retrieve_understand_{form}_v2{suffix}.json"
        )

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


class CompassBenchTEvalBeforeCallingEvaluator(BaseEvaluator):
    def __init__(
        self,
        bert_score_model: str = "simaas-qwen3-embedding-0-6b-v1",
        default_prompt_type: str = "json",
        eval_type: str = "reason",  # reason, retrieve, understand; not used in json mode
    ):
        self.bert_api_client = OpenAI(
            base_url=os.getenv(
                "OC_EVAL_API_BASE", "https://console.siflow.cn/model-api"
            ),
            api_key=os.getenv("OC_EVAL_API_KEY", ""),
        )
        self.bert_score_model = bert_score_model
        self.default_prompt_type = default_prompt_type
        self.eval_type = eval_type

    def score(self, predictions, references, test_set: Dataset) -> dict:
        if len(predictions) != len(references):
            return {"error": "Predictions and references must have the same length"}

        details = []
        results_list = []
        for prediction, reference, sample in zip(predictions, references, test_set):
            resp_data_sample, _ = self._process_response(
                {
                    "template": sample["template"],
                    "prediction": prediction,
                    "ground_truth": json.loads(reference),
                    "meta_data": sample["meta_data"],
                }
            )
            print(resp_data_sample)
            metrics_result = self._evaluate(resp_data_sample)
            results_list.append(metrics_result)

            details.append(
                {
                    "prediction": prediction,
                    "answer": reference,
                    **metrics_result,
                }
            )

        result = self._post_process(results_list)
        print(result)
        return {**result, "details": details}

    def _format_load(self, data) -> dict:
        try:
            json_format = format_load(data, start_character="{", end_character="}")
        except Exception:
            return {}
        if not isinstance(json_format, dict):
            return {}
        prepared_json_format = dict()
        try:
            prepared_json_format["thought"] = str(json_format["thought"])
        except Exception:
            prepared_json_format["thought"] = ""
        try:
            prepared_json_format["name"] = str(json_format["name"])
        except Exception:
            prepared_json_format["name"] = ""

        if self.default_prompt_type == "json":
            try:
                if isinstance(json_format["args"], dict):
                    prepared_json_format["args"] = json_format["args"]
                else:
                    prepared_json_format["args"] = dict()
            except Exception:
                prepared_json_format["args"] = dict()
        else:
            try:
                prepared_json_format["args"] = str(json_format["args"])
            except Exception:
                prepared_json_format["args"] = ""

        return prepared_json_format

    def _process_response(self, datum: dict) -> tuple[ResponseDataSample, int]:
        # Generated response, which can be a string or list
        pred_data = datum["prediction"]
        # Response of ground truth, which can be a string or list
        gt_data = datum["ground_truth"]
        # prompt_type: The type of planning prompt, supporting "json" and "ReWOO"
        if "meta" in datum:
            prompt_type = datum["meta"].get("response_format", self.default_prompt_type)
        else:
            prompt_type = self.default_prompt_type

        error = 0
        gt = self._format_load(gt_data)
        if prompt_type == "json":
            pred = self._format_load(pred_data)
            if pred == {} or gt == {}:
                error = 1
        elif prompt_type == "str":
            # choose the first line
            pred = dict()
            if self.eval_type == "reason":
                pred["thought"] = pred_data
            if self.eval_type == "retrieve":
                pred["name"] = pred_data
            if self.eval_type == "understand":
                pred["args"] = pred_data
        else:
            raise NotImplementedError(
                f"Currently, we only support json and str format, but get {prompt_type}"
            )

        if error == 1:
            pred = dict()
        return ResponseDataSample(template="", pred=pred, gt=gt), error

    def _evaluate(self, data_sample: ResponseDataSample) -> dict:
        """Evaluate the response data sample."""
        metrics_result = {
            "thought": 0,
            "name": 0,
            "args_precision": 0,
            "args_recall": 0,
            "args_f1_score": 0,
            "parse_rate": 0,
        }
        if "thought" in data_sample.pred and "thought" in data_sample.gt:
            pred_thought = data_sample.pred["thought"] or EMB_PLACEHOLDER
            gt_thought = data_sample.gt["thought"] or EMB_PLACEHOLDER

            resp = self.bert_api_client.embeddings.create(
                input=[pred_thought, gt_thought],
                model=self.bert_score_model,
            )
            time.sleep(0.1)  # to avoid being rate limited
            all_embeddings = [emb.embedding for emb in resp.data]
            pred_emb, gt_emb = all_embeddings

            # ensure dtype is float64 to keep compatible with isinstance float check in icl_base_evaluator
            pred_emb = np.array(pred_emb, dtype=np.float64)
            gt_emb = np.array(gt_emb, dtype=np.float64)
            cosine_scores = np.maximum(util.cos_sim(pred_emb, gt_emb).cpu().numpy(), 0)
            metrics_result["thought"] = cosine_scores[0, 0]

        if "name" in data_sample.pred and "name" in data_sample.gt:
            if data_sample.pred["name"] == data_sample.gt["name"]:
                metrics_result["name"] = 1
            else:
                metrics_result["name"] = 0
        if "args" in data_sample.pred and "args" in data_sample.gt:
            gt_num_keys = len(data_sample.gt["args"].keys())
            pred_num_keys = len(data_sample.pred["args"].keys())
            if pred_num_keys == 0 and gt_num_keys == 0:
                metrics_result["args_precision"] = 1
                metrics_result["args_recall"] = 1
                metrics_result["args_f1_score"] = 1
            elif pred_num_keys == 0 or gt_num_keys == 0:
                metrics_result["args_precision"] = 0
                metrics_result["args_recall"] = 0
                metrics_result["args_f1_score"] = 0
            else:
                correct_count = 0
                for key in data_sample.gt["args"].keys():
                    if key in data_sample.pred["args"] and str(
                        data_sample.pred["args"][key]
                    ) == str(data_sample.gt["args"][key]):
                        correct_count += 1
                metrics_result["args_precision"] = correct_count / pred_num_keys
                metrics_result["args_recall"] = correct_count / gt_num_keys
                if (
                    metrics_result["args_precision"] + metrics_result["args_recall"]
                    == 0
                ):
                    metrics_result["args_f1_score"] = 0
                else:
                    metrics_result["args_f1_score"] = (
                        2
                        * metrics_result["args_precision"]
                        * metrics_result["args_recall"]
                        / (
                            metrics_result["args_precision"]
                            + metrics_result["args_recall"]
                        )
                    )

        if len(data_sample.pred.keys()) == 0:
            metrics_result["parse_rate"] = 0
        else:
            metrics_result["parse_rate"] = 1
        return metrics_result

    def _post_process(self, results_list: list[dict]) -> dict[str, float]:
        # list of dict to dict of list
        results = dict()
        if self.default_prompt_type == "json":
            metric_keys = [
                "thought",
                "name",
                "args_precision",
                "args_recall",
                "args_f1_score",
                "parse_rate",
            ]
        if self.default_prompt_type == "str":
            if self.eval_type == "reason":
                metric_keys = ["thought", "parse_rate"]
            if self.eval_type == "retrieve":
                metric_keys = ["name", "parse_rate"]
            if self.eval_type == "understand":
                metric_keys = [
                    "args_precision",
                    "args_recall",
                    "args_f1_score",
                    "parse_rate",
                ]
        for key in metric_keys:
            results[key] = np.mean([result[key] for result in results_list]) * 100
        return results
