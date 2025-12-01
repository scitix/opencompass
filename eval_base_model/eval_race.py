#!/usr/bin/env python3
"""RACE Middle/High EM evaluation script (支持可配置的 shot_num)."""

from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer
import eval_utils
RACE_LEVELS = ("middle", "high")
SUPPORTED_SPLITS = ("validation", "test")


class ModelResponseError(RuntimeError):
    """Raised when the model response is invalid."""


@dataclass
class RaceExample:
    article: str
    question: str
    options: Dict[str, str]
    answer: str

    @classmethod
    def from_raw(cls, record: Dict[str, str]) -> "RaceExample":
        return cls(
            article=record.get("article", ""),
            question=record.get("question", ""),
            options={
                "A": record.get("A", ""),
                "B": record.get("B", ""),
                "C": record.get("C", ""),
                "D": record.get("D", ""),
            },
            answer=record.get("answer", "").strip().upper(),
        )


class RACEEvaluator:
    """Evaluate RACE subsets using configurable few-shot prompting and EM scoring."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        max_workers: int = 32,
        max_tokens: int = 16,
        shot_num: int = 5,
        data_root: str | None = None,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.shot_num = shot_num
        self.data_root = data_root or os.path.join(
            os.path.dirname(__file__), "datasets", "race"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B", trust_remote_code=True
        )

        self._few_shot_cache: Dict[str, List[RaceExample]] = {}
        self._lock = threading.Lock()
        self._level_stats: Dict[str, Dict[str, int]] = {}


    # PPL评估方法
    def extract_option_logprob(
        self, tokens: List[str], token_logprobs: List[Optional[float]], option_label: str
    ) -> Optional[float]:
        """从token列表中提取选项标签的logprob"""
        for i in range(len(tokens) - 1, max(0, len(tokens) - 15), -1):
            token = tokens[i]
            token_stripped = token.strip()
            if token_stripped == option_label or token == f" {option_label}":
                if i < len(token_logprobs) and token_logprobs[i] is not None:
                    return token_logprobs[i]
        return None

    def get_option_logprob_with_length(self, prompt: str, option_label: str, max_retries: int = 3):
        """获取选项的logprob和token长度"""
        for attempt in range(max_retries):
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=1,
                    logprobs=5,
                    echo=True,
                    temperature=0,
                )
                if not response.choices:
                    raise ModelResponseError('模型返回的choices为空')
                logprobs_obj = response.choices[0].logprobs
                if logprobs_obj is None:
                    raise ModelResponseError('模型返回的logprobs为空')
                tokens = logprobs_obj.tokens
                token_logprobs = logprobs_obj.token_logprobs
                if not tokens or not token_logprobs:
                    raise ModelResponseError('模型返回的tokens/logprobs为空')
                option_logprob = self.extract_option_logprob(tokens, token_logprobs, option_label)
                if option_logprob is not None:
                    token_length = len(tokens) - 1
                    return option_logprob, token_length
                raise ModelResponseError(f'无法提取选项 {option_label} 的logprob')
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise ModelResponseError(str(e))
        raise ModelResponseError('未知错误')

    # ---------------------------------------------------------------------
    # Dataset preparation helpers
    # ---------------------------------------------------------------------
    def ensure_local_files(self) -> None:
        """Ensure required RACE jsonl files exist locally, download if missing."""
        for level in RACE_LEVELS:
            for split in SUPPORTED_SPLITS:
                jsonl_path = self._jsonl_path(level, split)
                if os.path.exists(jsonl_path):
                    continue
                os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
                dataset = load_dataset("ehovy/race", level, split=split)
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for sample in dataset:
                        record = {
                            "article": sample.get("article", ""),
                            "question": sample.get("question", ""),
                            "options": sample.get("options", ["", "", "", ""]),
                            "answer": sample.get("answer", ""),
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _jsonl_path(self, level: str, split: str) -> str:
        return os.path.join(self.data_root, split, f"{level}.jsonl")

    def load_split(self, level: str, split: str) -> List[RaceExample]:
        jsonl_path = self._jsonl_path(level, split)
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"未找到 RACE {level} {split} 数据，请先下载到 {jsonl_path}"
            )

        examples: List[RaceExample] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                options = record.get("options", ["", "", "", ""])
                examples.append(
                    RaceExample(
                        article=record.get("article", ""),
                        question=record.get("question", ""),
                        options={
                            "A": options[0] if len(options) > 0 else "",
                            "B": options[1] if len(options) > 1 else "",
                            "C": options[2] if len(options) > 2 else "",
                            "D": options[3] if len(options) > 3 else "",
                        },
                        answer=record.get("answer", "").strip().upper(),
                    )
                )
        return examples

    # ------------------------------------------------------------------
    # Few-shot prompt construction
    # ------------------------------------------------------------------
    def _prepare_few_shot(self, level: str) -> List[RaceExample]:
        """准备 few-shot 示例"""
        if self.shot_num == 0:
            return []

        cache_key = f"{level}_{self.shot_num}"
        if cache_key in self._few_shot_cache:
            return self._few_shot_cache[cache_key]

        # 从 test split 加载 few-shot 示例（避免与 validation 重叠）
        test_examples = self.load_split(level, "test")

        # 使用前 shot_num 条
        if len(test_examples) < self.shot_num:
            raise ValueError(
                f"RACE-{level} test split 少于 {self.shot_num} 条样本，无法构建 {self.shot_num}-shot。"
            )
        few_shot_samples = test_examples[: self.shot_num]

        self._few_shot_cache[cache_key] = few_shot_samples
        return few_shot_samples

    def _format_example(self, example: RaceExample) -> str:
        """格式化 few-shot 示例，参考 OpenCompass 的格式"""
        parts = [
            "Read the article, and answer the question by replying A, B, C or D.",
            "",
            f"Article:\n{example.article.strip()}",
            "",
            f"Q: {example.question.strip()}",
            "",
            f"A. {example.options['A']}",
            f"B. {example.options['B']}",
            f"C. {example.options['C']}",
            f"D. {example.options['D']}",
            f"Answer: {example.answer}",
        ]
        return "\n".join(parts).strip() + "\n"

    def build_prompt(self, level: str, example: RaceExample, option_label: str = None) -> str:
        """构建 prompt，参考 OpenCompass 的格式（PPL方法需要option_label）"""
        few_shot_examples = self._prepare_few_shot(level)

        segments = []
        # 添加 few-shot examples
        for shot in few_shot_examples:
            segments.append(self._format_example(shot))
            segments.append("")

        # 添加测试样本
        if self.shot_num == 0:
            # 0-shot: 使用 OpenCompass 的 0-shot 格式
            prompt = (
                "Read the article, and answer the question by replying A, B, C or D.\n\n"
                f"{example.article.strip()}\n\n"
                f"Q: {example.question.strip()}\n\n"
                f"A. {example.options['A']}\n"
                f"B. {example.options['B']}\n"
                f"C. {example.options['C']}\n"
                f"D. {example.options['D']}\n"
                f"Answer: {option_label if option_label else ''}"
            )
        else:
            # Few-shot: 使用 OpenCompass 的 few-shot 格式
            segments.extend(
                [
                    "Read the article, and answer the question by replying A, B, C or D.",
                    "",
                    f"Article:\n{example.article.strip()}",
                    "",
                    f"Q: {example.question.strip()}",
                    "",
                    f"A. {example.options['A']}",
                    f"B. {example.options['B']}",
                    f"C. {example.options['C']}",
                    f"D. {example.options['D']}",
                    f"Answer: {option_label if option_label else ''}",
                ]
            )
            prompt = "\n".join(segments).strip()

        return prompt

    # ------------------------------------------------------------------
    # PPL evaluation
    # ------------------------------------------------------------------
    def _get_logprob_for_label(self, level: str, example: RaceExample, label: str) -> Tuple[str, float, int]:
        """为单个选项获取logprob和长度（用于并发调用）"""
        prompt = self.build_prompt(level, example, label)
        logprob, length = self.get_option_logprob_with_length(prompt, label)
        return label, logprob, length

    def evaluate_single(self, level: str, example: RaceExample) -> Dict[str, object]:
        """使用PPL方法+长度归一化评估单个样本"""
        import numpy as np
        
        option_logprobs = {}
        option_normalized_logprobs = {}
        option_ppls = {}
        
        # 并发获取所有选项的logprob
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._get_logprob_for_label, level, example, label): label
                for label in ['A', 'B', 'C', 'D']
            }
            
            for future in as_completed(futures):
                try:
                    label, logprob, length = future.result()
                    option_logprobs[label] = logprob
                    # 长度归一化
                    normalized_logprob = logprob / length if length > 0 else logprob
                    option_normalized_logprobs[label] = normalized_logprob
                    option_ppls[label] = np.exp(-normalized_logprob)
                except ModelResponseError:
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    raise
                except Exception as e:
                    label = futures[future]
                    print(f"\n获取选项 {label} 的logprob时出错: {e}")
                    option_logprobs[label] = -10.0
                    option_normalized_logprobs[label] = -10.0
                    option_ppls[label] = np.exp(10.0)
        
        # 确保所有选项都有结果
        for label in ['A', 'B', 'C', 'D']:
            if label not in option_normalized_logprobs:
                option_logprobs[label] = -10.0
                option_normalized_logprobs[label] = -10.0
                option_ppls[label] = np.exp(10.0)
        
        # 使用归一化后的logprob选择答案
        pred = max(option_normalized_logprobs, key=option_normalized_logprobs.get)
        is_correct = (pred == example.answer.strip().upper())

        with self._lock:
            stats = self._level_stats.setdefault(level, {"total": 0, "correct": 0})
            stats["total"] += 1
            stats["correct"] += int(is_correct)

        return {
            "question": example.question,
            "answer": example.answer,
            "pred": pred,
            "is_correct": is_correct,
            "logprobs": option_logprobs,
            "normalized_logprobs": option_normalized_logprobs,
            "ppls": option_ppls,
        }

    def evaluate_level(
        self, level: str, test_examples: List[RaceExample], max_samples: Optional[int] = None, seed: int = 42
    ) -> Dict[str, object]:
        if max_samples is not None:
            random.seed(seed)
            test_examples = random.sample(test_examples, min(max_samples, len(test_examples)))
            print(f"随机采样 {len(test_examples)} 条样本用于 RACE-{level} 评估（seed={seed}）。")

        # 预热检测：先测试3个样本
        print("\n执行预热检测（测试3个样本）...")
        warmup_samples = test_examples[:min(3, len(test_examples))]
        warmup_failed = False
        for i, sample in enumerate(warmup_samples):
            try:
                result = self.evaluate_single(level, sample)
                print(f"✓ 样本{i+1}: 测试通过")
                
                # 检测异常logprobs
                if 'normalized_logprobs' in result:
                    lps = result['normalized_logprobs']
                    if lps and all(v <= -10.0 or v is None for v in lps.values()):
                        print(f"⚠️  警告: 样本{i+1}的logprobs全部异常: {lps}")
                        warmup_failed = True
                        
            except Exception as e:
                print(f"⚠️  样本{i+1}失败: {e}")
                warmup_failed = True
        
        if warmup_failed:
            print(f"\n❌ 预热检测发现异常！")
            print(f"请检查：")
            print(f"  1. API endpoint是否正确")
            print(f"  2. 模型是否正常运行")
            print(f"  3. 网络连接是否正常")
            raise ModelResponseError("预热检测失败，中止评估")
        print("✓ 预热检测通过，开始正式评估...\n")

        # 重置计数器（预热检测已完成，不计入正式统计）
        stats = self._level_stats[level]
        stats["total"] = 0
        stats["correct"] = 0


        # 正式评估：评估所有样本（预热检测不跳过任何数据）
        eval_test_examples = test_examples
        print(f"正式评估 {len(eval_test_examples)} 条样本（包括预热样本）\n")
        
        results: List[Dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_example = {
                executor.submit(self.evaluate_single, level, example): example
                for example in eval_test_examples
            }
            with tqdm(total=len(eval_test_examples), desc=f"RACE-{level}", unit="题") as pbar:
                for future in as_completed(future_to_example):
                    result = future.result()
                    results.append(result)
                    stats = self._level_stats[level]
                    acc = stats["correct"] / max(1, stats["total"]) * 100
                    pbar.update(1)
                    pbar.set_postfix(acc=f"{acc:.2f}%", correct=stats["correct"])

        stats = self._level_stats[level]
        accuracy = stats["correct"] / max(1, stats["total"]) * 100
        return {
            "level": level,
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": accuracy,
            "shot_num": self.shot_num,
            "results": results,
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RACE 评估脚本")
    parser.add_argument(
        "--shot-num",
        type=int,
        default=5,
        help="Few-shot 数量，0 表示 0-shot（默认: 5）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("EVAL_MODEL_NAME", "qwen2-5-72b"),
        help="模型名称（默认: eval-qwen2-5-72b）",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("EVAL_BASE_URL", "http://localhost:8000/v1"),
        help="API base URL",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="数据集目录（默认: datasets/race）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="随机采样大小，None 表示评估所有数据（默认: None）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=128,
        help="最大并发数（默认: 32）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="最大生成 token 数（默认: 16）",
    )

    args = parser.parse_args()

    data_root = args.dataset_dir or os.path.join(
        os.path.dirname(__file__), "datasets", "race"
    )
    
    evaluator = RACEEvaluator(
        base_url=args.base_url,
        model=args.model,
        api_key="EMPTY",
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        shot_num=args.shot_num,
        data_root=data_root,
    )

    print(f"\n检查数据集: {data_root}")
    try:
        evaluator.ensure_local_files()
        print("✓ 数据集准备完成")
    except FileNotFoundError as e:
        print(f"⚠️  数据集文件不存在: {e}")
        print(f"请先运行以下命令下载数据集:")
        print(f"  python download_datasets.py race")
        print(f"或者手动将数据集放置在: {data_root}")
        return
    except Exception as e:
        print(f"✗ 数据集检查失败: {e}")
        import traceback

        traceback.print_exc()
        return

    log_dir = os.path.join(os.path.dirname(__file__), "logs", args.model)
    os.makedirs(log_dir, exist_ok=True)

    shot_desc = f"{args.shot_num}-shot" if args.shot_num > 0 else "0-shot"
    print("=" * 70)
    print(f"开始评估 RACE ({shot_desc}, EM)")
    print("=" * 70)

    summary = {}
    for level in RACE_LEVELS:
        print(f"\n评估 RACE-{level}...")
        test_examples = evaluator.load_split(level, "test")
        print(f"加载到 {len(test_examples)} 条测试样本")

        try:
            result = evaluator.evaluate_level(level, test_examples, max_samples=args.max_samples, seed=args.seed)
            summary[level] = {
                "total": result["total"],
                "correct": result["correct"],
                "accuracy": result["accuracy"],
            }

            # 生成输出文件名
            if args.shot_num == 0:
                output_filename = f"race_{level}_0shot.json"
            else:
                output_filename = f"race_{level}_{args.shot_num}shot.json"

            output_path = os.path.join(log_dir, output_filename)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"结果已保存至 {output_path}")
        except ModelResponseError as exc:
            print(f"✗ RACE-{level} 评估过程中检测到模型响应异常: {exc}")
            break

    print("=" * 70)
    print("整体总结：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
