import asyncio
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from pprint import pprint
from typing import TypedDict

from datasets import Dataset, concatenate_datasets, load_dataset
from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage
from tqdm import tqdm


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
                print(texts)
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


class CEvalItem(TypedDict):
    question: str
    A: str
    B: str
    C: str
    D: str
    answer: str
    subject: str
    subject_zh: str
    category: str


class CEvalEvalDetail(TypedDict):
    prediction: str
    reference: str
    correct: bool
    subject: str
    category: str


class CEvalDataset(BaseDataset):
    ceval_subject_mapping = {
        "computer_network": ["Computer Network", "计算机网络", "STEM"],
        "operating_system": ["Operating System", "操作系统", "STEM"],
        "computer_architecture": ["Computer Architecture", "计算机组成", "STEM"],
        "college_programming": ["College Programming", "大学编程", "STEM"],
        "college_physics": ["College Physics", "大学物理", "STEM"],
        "college_chemistry": ["College Chemistry", "大学化学", "STEM"],
        "advanced_mathematics": ["Advanced Mathematics", "高等数学", "STEM"],
        "probability_and_statistics": [
            "Probability and Statistics",
            "概率统计",
            "STEM",
        ],
        "discrete_mathematics": ["Discrete Mathematics", "离散数学", "STEM"],
        "electrical_engineer": ["Electrical Engineer", "注册电气工程师", "STEM"],
        "metrology_engineer": ["Metrology Engineer", "注册计量师", "STEM"],
        "high_school_mathematics": ["High School Mathematics", "高中数学", "STEM"],
        "high_school_physics": ["High School Physics", "高中物理", "STEM"],
        "high_school_chemistry": ["High School Chemistry", "高中化学", "STEM"],
        "high_school_biology": ["High School Biology", "高中生物", "STEM"],
        "middle_school_mathematics": ["Middle School Mathematics", "初中数学", "STEM"],
        "middle_school_biology": ["Middle School Biology", "初中生物", "STEM"],
        "middle_school_physics": ["Middle School Physics", "初中物理", "STEM"],
        "middle_school_chemistry": ["Middle School Chemistry", "初中化学", "STEM"],
        "veterinary_medicine": ["Veterinary Medicine", "兽医学", "STEM"],
        "college_economics": ["College Economics", "大学经济学", "Social Science"],
        "business_administration": [
            "Business Administration",
            "工商管理",
            "Social Science",
        ],
        "marxism": ["Marxism", "马克思主义基本原理", "Social Science"],
        "mao_zedong_thought": [
            "Mao Zedong Thought",
            "毛泽东思想和中国特色社会主义理论体系概论",
            "Social Science",
        ],
        "education_science": ["Education Science", "教育学", "Social Science"],
        "teacher_qualification": [
            "Teacher Qualification",
            "教师资格",
            "Social Science",
        ],
        "high_school_politics": ["High School Politics", "高中政治", "Social Science"],
        "high_school_geography": [
            "High School Geography",
            "高中地理",
            "Social Science",
        ],
        "middle_school_politics": [
            "Middle School Politics",
            "初中政治",
            "Social Science",
        ],
        "middle_school_geography": [
            "Middle School Geography",
            "初中地理",
            "Social Science",
        ],
        "modern_chinese_history": [
            "Modern Chinese History",
            "近代史纲要",
            "Humanities",
        ],
        "ideological_and_moral_cultivation": [
            "Ideological and Moral Cultivation",
            "思想道德修养与法律基础",
            "Humanities",
        ],
        "logic": ["Logic", "逻辑学", "Humanities"],
        "law": ["Law", "法学", "Humanities"],
        "chinese_language_and_literature": [
            "Chinese Language and Literature",
            "中国语言文学",
            "Humanities",
        ],
        "art_studies": ["Art Studies", "艺术学", "Humanities"],
        "professional_tour_guide": [
            "Professional Tour Guide",
            "导游资格",
            "Humanities",
        ],
        "legal_professional": ["Legal Professional", "法律职业资格", "Humanities"],
        "high_school_chinese": ["High School Chinese", "高中语文", "Humanities"],
        "high_school_history": ["High School History", "高中历史", "Humanities"],
        "middle_school_history": ["Middle School History", "初中历史", "Humanities"],
        "civil_servant": ["Civil Servant", "公务员", "Other"],
        "sports_science": ["Sports Science", "体育学", "Other"],
        "plant_protection": ["Plant Protection", "植物保护", "Other"],
        "basic_medicine": ["Basic Medicine", "基础医学", "Other"],
        "clinical_medicine": ["Clinical Medicine", "临床医学", "Other"],
        "urban_and_rural_planner": [
            "Urban and Rural Planner",
            "注册城乡规划师",
            "Other",
        ],
        "accountant": ["Accountant", "注册会计师", "Other"],
        "fire_engineer": ["Fire Engineer", "注册消防工程师", "Other"],
        "environmental_impact_assessment_engineer": [
            "Environmental Impact Assessment Engineer",
            "环境影响评价工程师",
            "Other",
        ],
        "tax_accountant": ["Tax Accountant", "税务师", "Other"],
        "physician": ["Physician", "医师资格", "Other"],
    }

    def _load_raw_data(self) -> Dataset:
        subsets = []
        for subset_name, (
            subject,
            subject_zh,
            category,
        ) in self.ceval_subject_mapping.items():
            subset = load_dataset(self._path, subset_name, split="test")

            def _preprocess_sample(sample: CEvalItem) -> CEvalItem:
                sample["subject"] = subject
                sample["subject_zh"] = subject_zh
                sample["category"] = category
                return sample

            subset = subset.map(_preprocess_sample, num_proc=os.cpu_count())
            subsets.append(subset)
        return concatenate_datasets(subsets)


class CEvalTask(BaseTask):
    def _preprocess(self, item: CEvalItem) -> list[ChatCompletionMessage]:
        msgs = [
            {
                "role": "system",
                "content": f"你是一个中文人工智能助手，以下是中国关于{item['subject_zh']}考试的单项选择题，请选出其中的正确答案。",
            },
            {
                "role": "user",
                "content": f"""以下是中国关于{item['subject_zh']}考试的单项选择题，请选出其中的正确答案。

{item['question']}
A. {item['A']}
B. {item['B']}
C. {item['C']}
D. {item['D']}
答案：""",
            },
        ]
        return msgs

    def _postprocess(self, resp: str) -> str:
        if resp and resp[0] in "ABCD":
            return resp[0]
        pattern = [
            r"^选([A-D])",
            r"^选项([A-D])",
            r"答案是\s?选?项?\s?([A-D])",
            r"答案为\s?选?项?\s?([A-D])",
            r"答案应为\s?选?项?\s?([A-D])",
            r"答案选\s?选?项?\s?([A-D])",
            r"答案是:\s?选?项?\s?([A-D])",
            r"答案应该是:\s?选?项?\s?([A-D])",
            r"正确的一项是\s?([A-D])",
            r"答案为:\s?选?项?\s?([A-D])",
            r"答案应为:\s?选?项?\s?([A-D])",
            r"答案:\s?选?项?\s?([A-D])",
            r"答案是：\s?选?项?\s?([A-D])",
            r"答案应该是：\s?选?项?\s?([A-D])",
            r"答案为：\s?选?项?\s?([A-D])",
            r"答案应为：\s?选?项?\s?([A-D])",
            r"答案：\s?选?项?\s?([A-D])",
        ]
        for p in pattern:
            m = re.search(p, resp)
            if m:
                return m.group(1)
        return ""

    def _eval(self, predictions: list[str], references: list[str]) -> dict[str, float]:
        correct = 0
        details: list[CEvalEvalDetail] = []
        category_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
        for prediction, reference, sample in zip(
            predictions, references, self._dataset
        ):
            subject = sample.get("subject", "unknown")
            category = sample.get("category", "unknown")

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
        references = [item["answer"] for item in self._dataset]
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
    c_eval_dataset_loader = CEvalDataset(path="./data/c-eval/")
    c_eval_dataset = c_eval_dataset_loader.load(
        n_repeats=1,
        num_examples=10,
        seed=0,
    )

    qwen2_5_72b_instruct = BaseModel(
        model="/models/preset/Qwen/Qwen2.5-72B-Instruct/v1.0/",
        api_base="http://172.18.145.156:8000/v1",
        api_key="EMPTY",
        max_tokens=8192,
        temperature=0.0,
    )

    c_eval_task = CEvalTask(
        model=qwen2_5_72b_instruct,
        dataset=c_eval_dataset,
        batch_size=128,
        timeout=30.0,
    )
    metrics = c_eval_task.run()
    print(f"Score: {metrics['score']}")
    pprint(metrics)
