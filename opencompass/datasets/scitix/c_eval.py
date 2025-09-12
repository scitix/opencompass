import os
from collections import defaultdict

from datasets import Dataset, concatenate_datasets, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.utils import get_data_path
from opencompass.utils.text_postprocessors import first_option_postprocess

# adapted from https://github.com/hkust-nlp/ceval/blob/cba65ae93bcf189149ced9f66ae0c958201faed9/code/evaluator_series/evaluators/chatgpt.py
SUBJECT_MAPPING = {
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


@LOAD_DATASET.register_module()
class CEvalDataset(BaseDataset):
    @staticmethod
    def load(
        path: str,
        few_shot: bool = False,
        few_shot_k: int = -1,
        cot: bool = False,
        n_repeats: int = 1,
        num_examples: int | None = None,
        seed: int = 3407,
    ) -> Dataset:
        path = get_data_path(path)
        subsets = []
        for subset_name, (
            subject,
            subject_zh,
            category,
        ) in SUBJECT_MAPPING.items():
            subset = load_dataset(path, subset_name, split="test")
            dev_examples = None
            if few_shot:
                # dev set for in-context learning
                dev_subset = load_dataset(path, subset_name, split="dev")
                if few_shot_k == -1:
                    dev_examples = dev_subset
                else:
                    k = min(few_shot_k, len(dev_subset))
                    if k < len(dev_subset):
                        dev_examples = dev_subset.select(range(k))
                    else:
                        dev_examples = dev_subset

            def _preprocess_sample(sample):
                dialog = [
                    {
                        "role": "system",
                        "content": f"你是一个中文人工智能助手，以下是中国关于{subject_zh}考试的单项选择题，请选出其中的正确答案。",
                    }
                ]
                if few_shot:
                    for i, example in enumerate(dev_examples):
                        prefix = (
                            f"以下是中国关于{subject_zh}考试的单项选择题，请选出其中的正确答案。\n\n"
                            if i == 0
                            else ""
                        )
                        few_shot_q = (
                            f"{prefix}{example['question']}\n"
                            f"A. {example['A']}\n"
                            f"B. {example['B']}\n"
                            f"C. {example['C']}\n"
                            f"D. {example['D']}\n"
                            "答案："
                        )
                        if cot:
                            few_shot_a = (
                                "让我们一步一步思考，\n"
                                f"{example['explanation']}\n"
                                f"所以答案是{example['answer']}。"
                            )
                        else:
                            few_shot_a = example["answer"]
                        dialog += [
                            {"role": "user", "content": few_shot_q},
                            {"role": "assistant", "content": few_shot_a},
                        ]
                test_prefix = (
                    f"以下是中国关于{subject_zh}考试的单项选择题，请选出其中的正确答案。\n\n"
                    if not few_shot
                    else ""
                )
                test_q = (
                    f"{test_prefix}{sample['question']}\n"
                    f"A. {sample['A']}\n"
                    f"B. {sample['B']}\n"
                    f"C. {sample['C']}\n"
                    f"D. {sample['D']}\n"
                    "答案："
                )
                dialog += [
                    {"role": "user", "content": test_q},
                    {"role": "assistant", "content": ""},
                ]
                sample["dialog"] = dialog
                # extra info
                sample["subject"] = subject
                sample["subject_zh"] = subject_zh
                sample["category"] = category
                return sample

            subset = subset.map(_preprocess_sample, num_proc=os.cpu_count())
            subsets.append(subset)
        dataset = concatenate_datasets(subsets)

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


@TEXT_POSTPROCESSORS.register_module("c_eval")
def c_eval_postprocess(response_text: str) -> str:
    return first_option_postprocess(response_text, options="ABCD", cushion=True)


class CEvalEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(pred_postprocessor={"type": "c_eval"})

    def score(self, predictions, references, test_set: Dataset) -> dict:
        if len(predictions) != len(references):
            return {"error": "Predictions and the references must have the same length"}

        correct = 0
        details = []
        category_metrics = defaultdict(lambda: {"correct": 0, "total": 0})
        for prediction, reference, sample in zip(predictions, references, test_set):
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
