import asyncio
import base64
import json
import os
import pickle
import zlib
from collections import defaultdict
from datetime import datetime

import httpx
from datasets import Dataset, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from .livecodebench.prompts.code_generation import (
    PromptConstants,
    get_generic_question_template_answer,
)
from .livecodebench.utils.extraction_utils import extract_code

VERSION_FILES = {
    "release_v0": [],  # placeholder for initial version
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
    ],
    "release_v6": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
}


@LOAD_DATASET.register_module()
class LCBCodeGenerationDataset(BaseDataset):
    @staticmethod
    def load(
        path: str,
        version_tag: str = "release_v1",  # 'release_vX' or 'vX_vY'
        start_date: str | None = None,
        end_date: str | None = None,
        cot: bool = False,
        n_repeats: int = 1,
        num_examples: int | None = None,
        seed: int = 3407,
    ) -> Dataset:
        path = get_data_path(path)
        if os.path.isdir(path):
            if version_tag in VERSION_FILES:
                data_files = {
                    "test": [os.path.join(path, f) for f in VERSION_FILES[version_tag]]
                }
            else:
                # assume `version_tag` is in the form of 'vX_vY'
                start_v, end_v = version_tag.split("_")
                # include all files in [start_v, end_v]
                start_int, end_int = (int(start_v[1:]) - 1, int(end_v[1:]))
                start_key, end_key = f"release_v{start_int}", f"release_v{end_int}"
                files = set(VERSION_FILES[end_key]) - set(VERSION_FILES[start_key])
                data_files = {"test": [os.path.join(path, f) for f in files]}
            dataset = load_dataset("json", data_files=data_files, split="test")
        else:
            dataset = load_dataset(
                path, split="test", version_tag=version_tag, trust_remote_code=True
            )

        # filter by date first
        if start_date is not None:
            p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
            dataset = dataset.filter(lambda e: p_start_date <= e["contest_date"])
        if end_date is not None:
            p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
            dataset = dataset.filter(lambda e: e["contest_date"] <= p_end_date)

        def _preprocess_sample(sample):
            # prompt
            question = {
                "question_content": sample["question_content"],
                "starter_code": sample["starter_code"],
            }
            prompt = get_generic_question_template_answer(question, cot)
            sample["prompt"] = prompt
            dialog = [
                {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""},
            ]
            sample["dialog"] = dialog

            # test cases
            public_test_cases = json.loads(sample["public_test_cases"])
            private_test_cases = sample["private_test_cases"]
            try:
                private_test_cases = json.loads(sample["private_test_cases"])
            except Exception:
                private_test_cases = json.loads(
                    pickle.loads(
                        zlib.decompress(
                            base64.b64decode(private_test_cases.encode("utf-8"))
                        )
                    )
                )
            metadata = json.loads(sample["metadata"])
            sample["evaluation_sample"] = json.dumps(
                {
                    "inputs": [
                        t["input"] for t in public_test_cases + private_test_cases
                    ],
                    "outputs": [
                        t["output"] for t in public_test_cases + private_test_cases
                    ],
                    "fn_name": metadata.get("func_name", None),
                }
            )
            return sample

        dataset = dataset.map(_preprocess_sample, num_proc=os.cpu_count())

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


@TEXT_POSTPROCESSORS.register_module("lcb_code_generation")
def lcb_code_generation_postprocess(response_text: str, lmstyle: str = "") -> str:
    return extract_code(response_text, lmstyle)


class LCBCodeGenerationEvaluator(BaseEvaluator):
    def __init__(
        self, api: str, num_workers: int = 4, timeout: float = 6.0, k: int = 1, **kwargs
    ):
        super().__init__(pred_postprocessor={"type": "lcb_code_generation"}, **kwargs)
        self.source = "livecodebench"
        self.api_url = api
        self.num_workers = num_workers
        self.timeout = timeout
        self.k = k

    def score(self, predictions, references, test_set: Dataset) -> dict:
        if len(predictions) != len(test_set):
            return {"error": "Predictions and the test set must have the same length"}

        if len(predictions) == 0:
            return {f"pass@{self.k}": 0.0, "details": []}

        results = asyncio.run(self._evaluate_all(predictions, test_set))

        grouped_results = defaultdict(list)
        for r in results:
            question_id = r["sample"]["question_id"]
            grouped_results[question_id].append(
                {
                    "prediction": r["prediction"],
                    "answer": r["sample"]["evaluation_sample"],
                    "msg": r["msg"],
                    "correct": r["status"],
                }
            )

        num_total_problems = len(grouped_results)
        if num_total_problems == 0:
            pass_at_k = 0.0
            num_timeout = 0
        else:
            num_correct = sum(
                1
                for attempts in grouped_results.values()
                if any(attempt["correct"] for attempt in attempts)
            )
            pass_at_k = num_correct * 100 / num_total_problems
            num_timeout = sum(
                1
                for attempts in grouped_results.values()
                if all("timeout" in attempt["msg"].lower() for attempt in attempts)
            )

        details = [
            attempt for attempts in grouped_results.values() for attempt in attempts
        ]
        return {
            f"pass@{self.k}": pass_at_k,
            "num_timeout": num_timeout,
            "details": details,
        }

    async def _evaluate_all(
        self, predictions: list[str], test_set: Dataset
    ) -> list[dict]:
        async with httpx.AsyncClient() as client:
            tasks = []
            for prediction, sample in zip(predictions, test_set):
                evaluation_sample = sample.get("evaluation_sample", "{}")
                evaluation_sample_obj = json.loads(evaluation_sample)
                inputs = evaluation_sample_obj.get("inputs", [])
                outputs = evaluation_sample_obj.get("outputs", [])
                fn_name = evaluation_sample_obj.get("fn_name", None)
                if len(inputs) != len(outputs):
                    continue
                tasks.append(
                    self._evaluate_sample(
                        client, prediction, inputs, outputs, fn_name, sample
                    )
                )

            semaphore = asyncio.Semaphore(self.num_workers)

            async def run_with_semaphore(task):
                async with semaphore:
                    return await task

            results = await asyncio.gather(
                *(run_with_semaphore(task) for task in tasks)
            )
            return results

    async def _evaluate_sample(
        self,
        client: httpx.AsyncClient,
        prediction: str,
        inputs: list[str],
        outputs: list[str],
        fn_name: str | None,
        sample: dict,
    ) -> dict:
        platform = sample.get("platform", "unknown")
        question_id = sample.get("question_id", "unknown")
        try:
            resp = await client.post(
                self.api_url,
                json={
                    "uuid": f"{platform}-{question_id}",
                    "source": self.source,
                    "code": prediction,
                    "test": {
                        "inputs": inputs,
                        "outputs": outputs,
                        "fn_name": fn_name,
                    },
                },
                # allow more time for more test cases
                timeout=self.timeout + len(inputs) * 2,
            )
            resp.raise_for_status()
            result = resp.json()
            return {
                "status": result.get("status", False),
                "msg": result.get("msg", ""),
                "prediction": prediction,
                "sample": sample,
            }
        except httpx.HTTPStatusError as e:
            return {
                "status": False,
                "msg": f"HTTP Error: {e.response.status_code} - {e.response.text}",
                "prediction": prediction,
                "sample": sample,
            }
        except Exception as e:
            return {
                "status": False,
                "msg": str(e),
                "prediction": prediction,
                "sample": sample,
            }
