import asyncio
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TypedDict

from datasets import Dataset, load_dataset
from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage
from tqdm import tqdm

from opencompass.datasets.scitix.simple_evals import (
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
    QUERY_TEMPLATE_MULTICHOICE,
    normalize_extracted_answer,
    normalize_response,
)


class BaseDataset(ABC):
    def __init__(self, path: str):
        self._path = path

    @abstractmethod
    def _load_raw_data(self) -> Dataset:
        raise NotImplementedError

    def load(
        self, n_repeats: int = 1, num_examples: int | None = None, seed: int = 0
    ) -> Dataset:
        dataset = self._load_raw_data()

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


class BaseModel:
    def __init__(self, model: str, api_base: str, api_key: str, **kwargs):
        self._model = model
        self._client = AsyncOpenAI(base_url=api_base, api_key=api_key)
        self._kwargs = kwargs

    async def achat(self, messages: list[ChatCompletionMessage]) -> str:
        text = ""
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **self._kwargs,
            stream=True,
        )
        async for chunk in resp:
            text += chunk.choices[0].delta.content or ""
        return text


class BaseTask(ABC):
    def __init__(
        self,
        model: BaseModel,
        dataset: Dataset,
        batch_size: int = 32,
        timeout: float = 30.0,
    ):
        self._model = model
        self._dataset = dataset
        self._semaphore = asyncio.Semaphore(batch_size)
        self._timeout = timeout

    @abstractmethod
    def _preprocess(self, item: dict) -> list[ChatCompletionMessage]:
        raise NotImplementedError

    async def _infer(self, idx: int, item: dict) -> tuple[int, str]:
        async with self._semaphore:
            try:
                req = self._preprocess(item)
                texts = await self._model.achat(req)
                preds = self._postprocess(texts)
                return idx, preds
            except Exception as e:
                logger.error(f"Inference error on item {idx}: {e}")
                return idx, ""

    @abstractmethod
    def _postprocess(self, text: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def _eval(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    async def arun(self) -> dict[str, float]:
        raise NotImplementedError

    def run(self) -> dict[str, float]:
        return asyncio.run(self.arun())


class MMLUItem(TypedDict):
    Question: str
    A: str
    B: str
    C: str
    D: str
    Answer: str
    Subject: str


class MMLUEvalDetail(TypedDict):
    prediction: str
    reference: str
    correct: bool
    subject: str
    category: str


class MMLUDataset(BaseDataset):
    def _load_raw_data(self) -> Dataset:
        data_files = {"test": self._path}
        return load_dataset("csv", data_files=data_files, split="test")


class MMLUTask(BaseTask):
    subject2category = {
        "abstract_algebra": "stem",
        "anatomy": "other",
        "astronomy": "stem",
        "business_ethics": "other",
        "clinical_knowledge": "other",
        "college_biology": "stem",
        "college_chemistry": "stem",
        "college_computer_science": "stem",
        "college_mathematics": "stem",
        "college_medicine": "other",
        "college_physics": "stem",
        "computer_security": "stem",
        "conceptual_physics": "stem",
        "econometrics": "social_sciences",
        "electrical_engineering": "stem",
        "elementary_mathematics": "stem",
        "formal_logic": "humanities",
        "global_facts": "other",
        "high_school_biology": "stem",
        "high_school_chemistry": "stem",
        "high_school_computer_science": "stem",
        "high_school_european_history": "humanities",
        "high_school_geography": "social_sciences",
        "high_school_government_and_politics": "social_sciences",
        "high_school_macroeconomics": "social_sciences",
        "high_school_mathematics": "stem",
        "high_school_microeconomics": "social_sciences",
        "high_school_physics": "stem",
        "high_school_psychology": "social_sciences",
        "high_school_statistics": "stem",
        "high_school_us_history": "humanities",
        "high_school_world_history": "humanities",
        "human_aging": "other",
        "human_sexuality": "social_sciences",
        "international_law": "humanities",
        "jurisprudence": "humanities",
        "logical_fallacies": "humanities",
        "machine_learning": "stem",
        "management": "other",
        "marketing": "other",
        "medical_genetics": "other",
        "miscellaneous": "other",
        "moral_disputes": "humanities",
        "moral_scenarios": "humanities",
        "nutrition": "other",
        "philosophy": "humanities",
        "prehistory": "humanities",
        "professional_accounting": "other",
        "professional_law": "humanities",
        "professional_medicine": "other",
        "professional_psychology": "social_sciences",
        "public_relations": "social_sciences",
        "security_studies": "social_sciences",
        "sociology": "social_sciences",
        "us_foreign_policy": "social_sciences",
        "virology": "other",
        "world_religions": "humanities",
    }

    def _preprocess(self, item: MMLUItem) -> list[ChatCompletionMessage]:
        data = {
            "Question": item["Question"],
            "A": item["A"],
            "B": item["B"],
            "C": item["C"],
            "D": item["D"],
        }
        msgs = [
            {"role": "user", "content": QUERY_TEMPLATE_MULTICHOICE.format(**data)},
        ]
        return msgs

    def _postprocess(self, resp: str) -> str:
        response_text = normalize_response(resp)
        extracted_answer = ""
        for answer_regex in MULTILINGUAL_ANSWER_REGEXES:
            regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
            match = re.search(regex, response_text)
            if match:
                extracted_answer = normalize_extracted_answer(match.group(1))
                break
        return extracted_answer

    def _eval(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        correct = 0
        details: list[MMLUEvalDetail] = []
        category_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
        for prediction, reference, sample in zip(
            predictions, references, self._dataset
        ):
            subject = sample.get("Subject", "unknown")
            category = self.subject2category.get(subject, "other")

            is_correct = prediction == reference
            if is_correct:
                correct += 1
                category_metrics[category]["correct"] += 1
            category_metrics[category]["total"] += 1

            details.append(
                {
                    "prediction": prediction,
                    "reference": reference,
                    "correct": is_correct,
                    "subject": subject,
                    "category": category,
                }
            )

        score = 100 * correct / len(predictions) if predictions else 0.0
        results = {"score": score, "details": details}
        for category, metrics in category_metrics.items():
            category_score = (
                100 * metrics["correct"] / metrics["total"]
                if metrics["total"] > 0
                else 0.0
            )
            results[f"score_{category}"] = category_score
        return results

    async def arun(self) -> dict[str, float]:
        predictions = [""] * len(self._dataset)
        references = [item["Answer"] for item in self._dataset]
        num_failures = 0

        tasks = [
            asyncio.create_task(self._infer(idx, item))
            for idx, item in enumerate(self._dataset)
        ]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            try:
                idx, pred = await asyncio.wait_for(task, timeout=self._timeout)
                predictions[idx] = pred
            except asyncio.TimeoutError:
                logger.error("An inference task timed out.")
                num_failures += 1
            except Exception as e:
                logger.error(f"An error occurred awaiting a task: {e}")
                num_failures += 1

        metrics = self._eval(predictions, references)
        metrics["num_failures"] = num_failures
        return metrics


if __name__ == "__main__":
    mmlu_dataset_loader = MMLUDataset(path="./data/mmlu_simple-evals/mmlu.csv")
    mmlu_dataset = mmlu_dataset_loader.load(
        n_repeats=1,
        num_examples=None,
        seed=0,
    )

    qwen2_5_72b_instruct = BaseModel(
        model="/models/preset/Qwen/Qwen2.5-72B-Instruct/v1.0/",
        api_base="http://localhost:8000/v1",
        api_key="EMPTY",
        max_tokens=8192,
        temperature=0.0,
    )

    mmlu_task = MMLUTask(
        model=qwen2_5_72b_instruct,
        dataset=mmlu_dataset,
        batch_size=128,
        timeout=30.0,
    )
    metrics = mmlu_task.run()
    print(f"Score: {metrics['score']}")
