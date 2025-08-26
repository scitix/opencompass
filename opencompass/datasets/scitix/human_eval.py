import asyncio
import re
from collections import defaultdict

import httpx
from datasets import Dataset, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path


@LOAD_DATASET.register_module()
class HumanEvalDataset(BaseDataset):
    @staticmethod
    def load(
        path: str, n_repeats: int = 1, num_examples: int | None = None, seed: int = 3407
    ) -> Dataset:
        path = get_data_path(path)
        dataset = load_dataset(path, split="test")

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


# adapted from https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/humaneval_eval.py
@TEXT_POSTPROCESSORS.register_module("human_eval")
def human_eval_postprocess(response_text: str) -> str:
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(response_text)
    extracted_answer = matches[0] if len(matches) >= 1 else response_text
    extracted_answer = extracted_answer[
        extracted_answer.find(":\n    ") + 2 :
    ]  # remove signature
    return extracted_answer


class HumanEvalEvaluator(BaseEvaluator):
    def __init__(
        self, api: str, num_workers: int = 4, timeout: float = 5.0, k: int = 1, **kwargs
    ):
        super().__init__(pred_postprocessor={"type": "human_eval"}, **kwargs)
        self.source = "human-eval"
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
            task_id = r["sample"]["task_id"]
            grouped_results[task_id].append(
                {
                    "prediction": r["prediction"],
                    "answer": r["sample"]["canonical_solution"],
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
                tasks.append(self._evaluate_sample(client, prediction, sample))

            semaphore = asyncio.Semaphore(self.num_workers)

            async def run_with_semaphore(task):
                async with semaphore:
                    return await task

            results = await asyncio.gather(
                *(run_with_semaphore(task) for task in tasks)
            )
            return results

    async def _evaluate_sample(
        self, client: httpx.AsyncClient, prediction: str, sample: dict
    ) -> dict:
        task_id = sample.get("task_id", "unknown")
        check_program = (
            sample["prompt"]
            + prediction
            + "\n"
            + sample["test"]
            + "\n"
            + f"check({sample['entry_point']})"
        )
        try:
            resp = await client.post(
                self.api_url,
                json={
                    "uuid": task_id,
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
