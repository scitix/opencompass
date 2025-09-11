import asyncio
import json
import os
from collections import defaultdict

import httpx
from datasets import Dataset, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path

from .livecodebench.evaluation.utils_execute import BASE_IMPORTS
from .livecodebench.prompts.code_execution import format_prompt_execution
from .livecodebench.utils.extraction_utils import extract_execution_code


@LOAD_DATASET.register_module()
class LCBCodeExecutionDataset(BaseDataset):
    @staticmethod
    def load(
        path: str,
        cot: bool = False,
        n_repeats: int = 1,
        num_examples: int | None = None,
        seed: int = 3407,
    ) -> Dataset:
        path = get_data_path(path)
        dataset = load_dataset(path, split="test")

        def _preprocess_sample(sample):
            # prompt
            question = {
                "code": sample["code"],
                "input": sample["input"],
            }
            prompt = format_prompt_execution(question, cot)
            sample["prompt"] = prompt

            # test cases
            evaluation_sample = json.dumps(
                {
                    "code": sample["code"],
                    "input": sample["input"],
                    "output": sample["output"],
                }
            )
            sample["evaluation_sample"] = evaluation_sample
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


@TEXT_POSTPROCESSORS.register_module("lcb_code_execution")
def lcb_code_execution_postprocess(response_text: str, cot: bool = False) -> str:
    return extract_execution_code(response_text, cot)


class LCBCodeExecutionEvaluator(BaseEvaluator):
    def __init__(
        self,
        api: str,
        num_workers: int = 4,
        timeout: float = 5.0,
        k: int = 1,
        cot: bool = False,
        **kwargs,
    ):
        super().__init__(
            pred_postprocessor={"type": "lcb_code_execution", "cot": cot}, **kwargs
        )
        self.source = "livecodebench"
        self.api_url = api
        self.num_workers = num_workers
        self.timeout = timeout
        self.k = k

    def score(self, predictions, references, test_set: Dataset) -> dict:
        if len(predictions) != len(test_set):
            return {"error": "Predictions and the test set must have the same length"}

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
        else:
            num_correct = sum(
                1
                for attempts in grouped_results.values()
                if any(attempt["correct"] for attempt in attempts)
            )
            pass_at_k = num_correct * 100 / num_total_problems

        details = [
            attempt for attempts in grouped_results.values() for attempt in attempts
        ]
        return {f"pass@{self.k}": pass_at_k, "details": details}

    async def _evaluate_all(
        self, predictions: list[str], test_set: Dataset
    ) -> list[dict]:
        async with httpx.AsyncClient() as client:
            tasks = []
            for prediction, sample in zip(predictions, test_set):
                evaluation_sample = sample.get("evaluation_sample", "{}")
                evaluation_sample_obj = json.loads(evaluation_sample)
                code = evaluation_sample_obj.get("code", "")
                inp = evaluation_sample_obj.get("input", "")
                out = evaluation_sample_obj.get("output", "")
                tasks.append(
                    self._evaluate_sample(client, prediction, code, inp, out, sample)
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
        code: str,
        inp: str,
        out: str,
        sample: dict,
    ) -> dict:
        if inp in prediction:
            return {
                "status": False,
                "msg": "input leaked in prediction",
                "prediction": prediction,
                "sample": sample,
            }

        question_id = str(sample.get("question_id", "unknown"))
        check_program = f"{BASE_IMPORTS}\n{code}\nassert {out} == {prediction}"
        try:
            resp = await client.post(
                self.api_url,
                json={
                    "uuid": question_id,
                    "source": self.source,
                    "code": check_program,
                },
                timeout=self.timeout,
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
