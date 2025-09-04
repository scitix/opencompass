import copy
import itertools
import json
import os
import re
import time

import networkx as nx
import numpy as np
from datasets import Dataset
from openai import OpenAI
from sentence_transformers import util

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .t_eval import EMB_PLACEHOLDER, ResponseDataSample, format_load


# before all callings - plan
# adapted from https://github.com/open-compass/T-Eval/blob/58f22406404d7e2a4f36856a19c7f4dc28a0a5f0/teval/evaluators/planning_evaluator.py
@LOAD_DATASET.register_module()
class CompassBenchTEvalPlanDataset(BaseDataset):
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
        path = os.path.join(data_dir, "data", f"plan_{form}_v2{suffix}.json")

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
                    "ground_truth": json.dumps(v.get("ground_truth", [])),
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


class CompassBenchTEvalPlanEvaluator(BaseEvaluator):
    def __init__(
        self,
        name_weight=0.75,
        args_weight=0.25,
        match_threshold=0.8,
        match_strategy: str = "bertscore",  # "bertscore" or "permutation"
        bert_score_model: str = "simaas-qwen3-embedding-0-6b-v1",
        default_prompt_type: str = "json",  # "json" or "ReWOO"
    ):
        self.name_weight = name_weight
        self.args_weight = args_weight
        self.match_threshold = match_threshold
        self.match_strategy = match_strategy
        self.bert_api_client = OpenAI(
            base_url=os.getenv(
                "OC_EVAL_API_BASE", "https://console.siflow.cn/model-api"
            ),
            api_key=os.getenv("OC_EVAL_API_KEY", ""),
        )
        self.bert_score_model = bert_score_model
        self.default_prompt_type = default_prompt_type

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

    def _format_load(self, data: str) -> list:
        r"""
        ensure evaluator can work correctly under any data input
        """
        try:
            json_format = format_load(data, start_character="[", end_character="]")
        except Exception:
            return []
        if not isinstance(json_format, list):
            return []
        for i in range(len(json_format)):
            try:
                json_format[i] = {
                    "name": str(json_format[i]["name"]),
                    "id": int(json_format[i]["id"]),
                    "args": str(json_format[i]["args"]),
                }
                if self.match_strategy == "permutation":
                    json_format[i]["prev"] = list(json_format[i].get("prev", []))
            except Exception:
                return []
        return json_format

    def _process_response(self, datum: dict) -> tuple[ResponseDataSample, int]:
        # Generated response, which can be a string or list
        pred_data = datum["prediction"]
        # Response of ground truth, which can be a string or list
        gt_data = datum["ground_truth"]
        # prompt_type: The type of planning prompt, supporting "json" and "ReWOO"
        if "meta" in datum:
            prompt_type = datum["meta"].get("prompt_type", self.default_prompt_type)
        else:
            prompt_type = self.default_prompt_type

        error = 0
        pred = dict()
        gt = dict()
        gt["planning"] = self._format_load(gt_data)
        if prompt_type == "json":
            pred["planning"] = self._format_load(pred_data)
            if pred["planning"] == [] or gt["planning"] == []:
                error = 1

        elif prompt_type == "ReWOO":
            """
            This type is deprecated
            The planning prediction data should in this format:
                Plan 1: <str> description about the first action
                Dependency 1: <list[number]> the first action depends on which previous actions
                Action 1: #E1 = api_name1(args1)
                ...
            Which will be passed only if "number of plan lines == number of dependency lines == number of action lines"
            The passed data's format is:
                [
                    dict(
                        id = i,
                        name = curr_name,
                        args = args_str
                    )
                    ...
                ]

            The golden answer prediction is a json that is the same as the json format.
            """
            thoughts = re.findall(r"(Plan [0-9]+: .+)", pred_data)
            dependencies = re.findall(r"(Dependency [0-9]+: .+)", pred_data)
            action_units = re.findall(r"Action [0-9]+: (.+)", pred_data)

            if not (
                len(thoughts) == len(dependencies)
                and len(thoughts) == len(action_units)
            ):
                pred["planning"] = []
                gt["planning"] = []
                return ResponseDataSample(template="", pred=pred, gt=gt), 1

            plan_action = []
            for i in range(len(action_units)):
                dependency_list = re.findall(
                    r"Dependency [0-9]+: (.+)", dependencies[i]
                )
                if action_units[i][0] == "#":
                    # The action has a return #E
                    args_str_list = re.findall(
                        r"#E[0-9]+ = .+\((.+)\)", action_units[i]
                    )
                    name_list = re.findall(r"#E[0-9]+ = (.+)\(", action_units[i])
                else:
                    # The action does not have a return
                    args_str_list = re.findall(r".+\((.+)\)", action_units[i])
                    name_list = re.findall(r"(.+)\(", action_units[i])
                if len(name_list) > 0:
                    curr_name = name_list[0]
                else:
                    curr_name = ""
                if len(args_str_list) > 0:
                    args_str = "{" + args_str_list[0] + "}"
                else:
                    args_str = "{}"
                if len(dependency_list) > 0:
                    dependency_str = dependency_list[0]
                else:
                    dependency_str = ""
                dependency = re.findall("([0-9]+)", dependency_str)
                dependency = list(set([int(x) - 1 for x in dependency]))
                plan_action.append(
                    dict(id=i, name=curr_name, prev=dependency, args=args_str)
                )
            pred["planning"] = plan_action
            # Turn dict into args str
            for i in range(len(gt["planning"])):
                args_str = ""
                if isinstance(gt["planning"][i]["args"], str):
                    args_dict = eval(gt["planning"][i]["args"])
                else:
                    assert isinstance(gt["planning"][i]["args"], dict)
                    args_dict = gt["planning"][i]["args"]
                for it in args_dict:
                    if args_str == "":
                        args_str += f'{it}="{args_dict[it]}"'
                    else:
                        args_str += f', {it}="{args_dict[it]}"'
                gt["planning"][i]["args"] = "{" + args_str + "}"

        elif prompt_type == "str":
            pred_data_format = pred_data.replace(". ", "\n").split("\n")
            pred_actions = []
            for pred_step in pred_data_format:
                first_occur_time = 1e9
                pred_action = ""
                for api_name in datum["meta"]["API_list"]:
                    occur_time = pred_step.find(api_name)
                    if occur_time != -1 and occur_time < first_occur_time:
                        first_occur_time = occur_time
                        pred_action = api_name
                if pred_action != "":
                    pred_actions.append(
                        {
                            "id": len(pred_actions),
                            "name": pred_action,
                            "args": pred_step,
                        }
                    )
            pred["planning"] = pred_actions
            if len(pred["planning"]) == 0:
                error = 1
        else:
            raise NotImplementedError(
                f"Currently, we only support json and ReWOO format, but get {prompt_type}"
            )

        return ResponseDataSample(template="", pred=pred, gt=gt), error

    def _evaluate(self, data_sample) -> dict:
        if self.match_strategy == "bertscore":
            metrics_result = self._bertscore_match(
                data_sample.pred["planning"], data_sample.gt["planning"]
            )
        elif self.match_strategy == "permutation":
            metrics_result = self._permutation_match(
                data_sample.pred["planning"], data_sample.gt["planning"]
            )
        else:
            raise NotImplementedError
        if (
            len(data_sample.pred["planning"]) == 0
            or len(data_sample.gt["planning"]) == 0
        ):
            metrics_result["parse_rate"] = 0
        else:
            metrics_result["parse_rate"] = 1
        return metrics_result

    def _bertscore_match(self, pred_plan: list, gt_plan: list) -> dict:
        """
        Calculate the similarity between predicted plan and golden answer,
        A plan can be regarded a sequence of actions, and each action has a name and args.
        Firstly, use bertscore to calculate pointwise similarity by:
            similarity(u, v) = bertscore(u.name, v.name) * name_weight + bertscore(u.args, v.args) * args_weight;
        Secondly, use Hungarian matching to match the points;
        Finally, use LIS to calculate the number of matched nodes.
        """
        if len(pred_plan) == 0 or len(gt_plan) == 0:
            return {"precision": 0, "recall": 0, "f1_score": 0}

        pred_plan = copy.deepcopy(sorted(pred_plan, key=lambda x: x["id"]))
        gt_plan = copy.deepcopy(sorted(gt_plan, key=lambda x: x["id"]))

        # Add end action
        # Currently it is hard-code
        if pred_plan[-1]["name"] == "FinishAction":
            pred_plan = pred_plan[:-1]
        if gt_plan[-1]["name"] == "FinishAction":
            gt_plan = gt_plan[:-1]
        # The total counts of nodes and edges.
        len_pred = len(pred_plan)
        len_gt = len(gt_plan)

        bert_score_matrix = np.zeros((len_pred, len_gt))
        name_pred, args_pred = [], []
        name_gt, args_gt = [], []
        for i in range(len_pred):
            name_pred.append(pred_plan[i]["name"] or EMB_PLACEHOLDER)
            args_pred.append(str(pred_plan[i]["args"]) or EMB_PLACEHOLDER)
        for i in range(len_gt):
            name_gt.append(gt_plan[i]["name"] or EMB_PLACEHOLDER)
            args_gt.append(str(gt_plan[i]["args"]) or EMB_PLACEHOLDER)

        all_inputs = name_pred + name_gt + args_pred + args_gt
        resp = self.bert_api_client.embeddings.create(
            input=all_inputs,
            model=self.bert_score_model,
        )
        time.sleep(0.1)  # to avoid being rate limited
        all_embeddings = [emb.embedding for emb in resp.data]

        l_np, l_ng, l_ap = len(name_pred), len(name_gt), len(args_pred)
        start_ng = l_np
        start_ap = start_ng + l_ng
        start_ag = start_ap + l_ap
        # ensure dtype is float64 to keep compatible with isinstance float check in icl_base_evaluator
        name_pred_emb = np.array(all_embeddings[:start_ng], dtype=np.float64)
        name_gt_emb = np.array(all_embeddings[start_ng:start_ap], dtype=np.float64)
        args_pred_emb = np.array(all_embeddings[start_ap:start_ag], dtype=np.float64)
        args_gt_emb = np.array(all_embeddings[start_ag:], dtype=np.float64)

        name_cosine_scores = np.maximum(
            util.cos_sim(name_pred_emb, name_gt_emb).cpu().numpy(), 0
        )
        args_cosine_scores = np.maximum(
            util.cos_sim(args_pred_emb, args_gt_emb).cpu().numpy(), 0
        )
        for i in range(len_pred):
            for j in range(len_gt):
                bert_score_matrix[i][j] = (
                    name_cosine_scores[i][j] * self.name_weight
                    + args_cosine_scores[i][j] * self.args_weight
                )
        G = nx.Graph()
        for i in range(len_pred):
            for j in range(len_gt):
                if bert_score_matrix[i][j] > self.match_threshold:
                    G.add_edge(i, str(j), weight=bert_score_matrix[i][j])
        max_weight_matching = nx.max_weight_matching(G)

        pred_to_gt_mapping = dict()
        for key in max_weight_matching:
            if isinstance(key[0], int):
                pred_to_gt_mapping[int(key[0])] = int(key[1])
            else:
                pred_to_gt_mapping[int(key[1])] = int(key[0])

        # If a prediction node does not match any golden answer node, we mark the node as -1.
        for i in range(len_pred):
            if i not in pred_to_gt_mapping:
                pred_to_gt_mapping[i] = -1
        # Calculate how many nodes are matched by Longest Increasing Subsequence (LIS)
        dp = np.ones(len_pred)
        for i in range(len_pred):
            for j in range(i):
                if pred_to_gt_mapping[i] == -1 or pred_to_gt_mapping[j] == -1:
                    continue
                if pred_to_gt_mapping[i] > pred_to_gt_mapping[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        correct_count = int(max(dp))

        recall, precision = correct_count / len(gt_plan), correct_count / len(pred_plan)
        f1_score = 2 * recall * precision / (recall + precision)
        result = {"precision": precision, "recall": recall, "f1_score": f1_score}
        return result

    def _permutation_match(self, pred_plan: list, gt_plan: list) -> dict:
        """
        The function calculates all the permutation matches' score and selects the max f1_score;
        Since permutation is time consuming, we truncate the length of plans to 9
        """
        if pred_plan[-1]["name"] != "FinishAction":
            pred_plan.append(
                {
                    "id": len(pred_plan),
                    "prev": [],
                    "name": "FinishAction",
                    "args": r"\{\}",
                }
            )

        if gt_plan[-1]["name"] != "FinishAction":
            gt_plan.append(
                {
                    "id": len(gt_plan),
                    "prev": [],
                    "name": "FinishAction",
                    "args": r"\{\}",
                }
            )

        # truncate plans to 9 since it is too long for permutation.
        if len(pred_plan) > 9:
            pred_plan = pred_plan[:9]
        if len(gt_plan) > 9:
            gt_plan = pred_plan[:9]

        pred_plan = sorted(pred_plan, key=lambda x: x["id"])
        gt_plan = sorted(gt_plan, key=lambda x: x["id"])
        len_pred = len(pred_plan)
        len_gt = len(gt_plan)
        map_id_max = max(len_pred, len_gt)
        numbers = [i for i in range(map_id_max)]
        perms = itertools.permutations(numbers, len_pred)
        gt_prev_count, pred_prev_count = 0, 0
        for i in range(len_gt):
            gt_plan[i]["prev"].append(i)
            gt_prev_count += len(gt_plan[i]["prev"])
        for i in range(len_pred):
            pred_plan[i]["prev"].append(i)
            pred_prev_count += len(pred_plan[i]["prev"])
        if gt_prev_count == 0 or pred_prev_count == 0:
            return {"precision": 0, "recall": 0, "f1_score": 0}
        max_recall, max_precision, max_f1 = 0, 0, 0
        for perm in perms:
            correct_count = 0
            for i in range(len_pred):
                if perm[i] >= len_gt:
                    continue
                for j in pred_plan[i]["prev"]:
                    if perm[j] in gt_plan[perm[i]]["prev"]:
                        correct_count += 1
            now_recall, now_precision = (
                correct_count / gt_prev_count,
                correct_count / pred_prev_count,
            )
            if now_recall + now_precision == 0:
                continue
            now_f1 = 2 * now_recall * now_precision / (now_recall + now_precision)
            if now_f1 > max_f1:
                max_f1, max_recall, max_precision = now_f1, now_recall, now_precision
        return {"precision": max_precision, "recall": max_recall, "f1_score": max_f1}

    def _post_process(self, results_list: list[dict]) -> dict:
        # list of dict to dict of list
        results = dict()
        planning_metric_keys = ["precision", "recall", "f1_score", "parse_rate"]
        for key in planning_metric_keys:
            results[key] = np.mean([result[key] for result in results_list]) * 100
        return results
