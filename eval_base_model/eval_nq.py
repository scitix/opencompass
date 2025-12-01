#!/usr/bin/env python3
"""NaturalQuestions 5-shot EM evaluation script."""

from __future__ import annotations
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from opencompass.utils.text_postprocessors import general_postprocess
import eval_utils


EVAL_SPLIT = "validation"
FEW_SHOT_SPLIT = "train"


class ModelResponseError(RuntimeError):
    """Raised when the model response is invalid."""


@dataclass
class NQExample:
    question: str
    answers: List[str]

    @property
    def canonical_answer(self) -> str:
        return self.answers[0] if self.answers else ""


class NaturalQuestionsEvaluator:
    """Evaluate NaturalQuestions using N-shot prompting and EM scoring."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        max_workers: int = 32,
        max_tokens: int = 256,
        data_root: str | None = None,
        num_shots: int = 5,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.num_shots = num_shots
        self.data_root = data_root or os.path.join(
            os.path.dirname(__file__), "datasets", "nq"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B", trust_remote_code=True
        )

        self._few_shot_cache: Optional[List[NQExample]] = None
        self._lock = threading.Lock()
        self._stats = {"total": 0, "correct": 0}

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------
    def ensure_local_files(self) -> None:
        """Ensure NaturalQuestions jsonl files exist locally, download if missing."""
        for split in (EVAL_SPLIT, FEW_SHOT_SPLIT):
            jsonl_path = self._jsonl_path(split)
            if os.path.exists(jsonl_path):
                continue

            os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
            dataset = load_dataset("google-research-datasets/natural_questions", split=split)
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for sample in dataset:
                    answers = sample.get("answer", []) or []
                    if isinstance(answers, str):
                        answers = [answers]
                    answers = [ans for ans in answers if ans]
                    if not answers:
                        continue
                    record = {
                        "question": sample.get("question", ""),
                        "answers": answers,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _jsonl_path(self, split: str) -> str:
        return os.path.join(self.data_root, f"{split}.jsonl")

    def load_split(self, split: str) -> List[NQExample]:
        jsonl_path = self._jsonl_path(split)
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"未找到 NaturalQuestions {split} 数据，请先下载到 {jsonl_path}"
            )
        examples: List[NQExample] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                answers = record.get("answers", []) or []
                if not answers:
                    continue
                examples.append(
                    NQExample(
                        question=record.get("question", ""),
                        answers=answers,
                    )
                )
        return examples

    # ------------------------------------------------------------------
    # Few-shot prompt construction
    # ------------------------------------------------------------------
    def _prepare_few_shot(self) -> List[NQExample]:
        if self._few_shot_cache is not None:
            return self._few_shot_cache
        
        if self.num_shots == 0:
            self._few_shot_cache = []
            return self._few_shot_cache

        few_shot_pool = self.load_split(FEW_SHOT_SPLIT)
        if len(few_shot_pool) < self.num_shots:
            raise ValueError(f"NaturalQuestions few-shot源数据不足{self.num_shots}条，无法构建{self.num_shots}-shot示例。")

        # 过滤掉答案为纯数字的样本（避免模型过拟合到年份等数字）
        valid_examples = []
        for ex in few_shot_pool:
            # 检查答案是否为纯数字
            is_pure_number = all(ans.strip().replace(',', '').replace('.', '').isdigit() 
                                for ans in ex.answers if ans.strip())
            if not is_pure_number:
                valid_examples.append(ex)
            
            # 收集到足够的样本就停止
            if len(valid_examples) >= self.num_shots:
                break
        
        if len(valid_examples) < self.num_shots:
            raise ValueError(
                f"NaturalQuestions 过滤纯数字答案后，有效样本不足{self.num_shots}条"
                f"（仅{len(valid_examples)}条），无法构建{self.num_shots}-shot示例。"
            )

        self._few_shot_cache = valid_examples[:self.num_shots]
        return self._few_shot_cache

    def build_prompt(self, example: NQExample) -> str:
        few_shot_examples = self._prepare_few_shot()
        intro = (
            "Answer the question directly. Respond with the format 'The answer is <...>'."
        )

        segments = [intro, ""]
        for shot in few_shot_examples:
            segments.extend(
                [
                    f"Q: {shot.question.strip()}",
                    f"A: The answer is {shot.canonical_answer.strip()}.",
                    "",
                ]
            )

        segments.extend(
            [
                f"Q: {example.question.strip()}",
                "A: The answer is",
            ]
        )

        prompt = "\n".join(segments).strip()
        # tokenized = self._tokenizer(prompt, add_special_tokens=False)
        # print(
        #     f"[Prompt Stats][NaturalQuestions] chars={len(prompt)} tokens={len(tokenized['input_ids'])}"
        # )
        return prompt

    # ------------------------------------------------------------------
    # Model interaction & scoring
    # ------------------------------------------------------------------
    def _call_model(self, prompt: str, max_retries: int = 3) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                )
                if not response.choices:
                    raise ModelResponseError("模型返回的choices为空。")
                
                text = response.choices[0].text
                
                # 检测异常响应
                if not text or text.strip() == "":
                    raise ModelResponseError("模型返回空响应。")
                if text.strip() in ["<unk>", "<unk>.", "<UNK>", "<UNK>."]:
                    raise ModelResponseError(f"模型返回无效token: {text.strip()}")
                if len(text.strip()) < 2 and not text.strip().isalnum():
                    raise ModelResponseError(f"模型返回异常短响应: {text.strip()}")
                
                return text
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                message = str(exc)
                if "rate" in message.lower() or "limit" in message.lower():
                    wait_time = min(10 * (attempt + 1), 60)
                    time.sleep(wait_time)
                    continue
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    break
        if last_error is None:
            last_error = ModelResponseError("未知原因导致模型响应失败。")
        if not isinstance(last_error, ModelResponseError):
            last_error = ModelResponseError(str(last_error))
        raise last_error

    def _normalize(self, text: str) -> str:
        return general_postprocess(text.strip().lower())

    def _is_correct(self, prediction: str, answers: List[str]) -> bool:
        normalized_pred = self._normalize(prediction)
        normalized_answers = [self._normalize(ans) for ans in answers]
        return any(ans and ans in normalized_pred for ans in normalized_answers)

    def evaluate_single(self, example: NQExample) -> Dict[str, object]:
        try:
            prompt = self.build_prompt(example)
            raw_answer = self._call_model(prompt)
            pred_text = raw_answer.strip().split("\n", 1)[0]
            pred_text = pred_text.replace("The answer is", "").strip()

            is_correct = self._is_correct(pred_text, example.answers)

            with self._lock:
                self._stats["total"] += 1
                self._stats["correct"] += int(is_correct)

            return {
                "question": example.question,
                "answers": example.answers,
                "pred": pred_text,
                "raw_model_output": raw_answer.strip(),
                "is_correct": is_correct,
            }
        except ModelResponseError as e:
            # 模型返回异常，记录为错误但不中断评估
            with self._lock:
                self._stats["total"] += 1
                self._stats["correct"] += 0
            
            return {
                "question": example.question,
                "answers": example.answers,
                "pred": f"[ERROR: {str(e)}]",
                "raw_model_output": f"[ERROR: {str(e)}]",
                "is_correct": False,
            }

    def evaluate_dataset(
        self, examples: List[NQExample], max_samples: Optional[int] = None, seed: int = 42
    ) -> Dict[str, object]:
        if max_samples is not None:
            random.seed(seed)
            examples = random.sample(examples, min(max_samples, len(examples)))
            print(f"随机采样 {len(examples)} 条 NaturalQuestions 样本进行评估（seed={seed}）。")

        # 预热检测：先测试少量样本验证API和模型状态
        num_warmup = min(3, len(examples)) if not max_samples else min(3, max_samples, len(examples))
        print(f"\n执行预热检测（测试 {num_warmup} 个样本）...")
        warmup_samples = examples[:num_warmup]
        for i, example in enumerate(warmup_samples):
            try:
                result = self.evaluate_single(example)
                pred_text = result.get('pred', '')[:30]
                print(f"✓ 样本{i+1}: 问题='{example.question[:50]}...' 预测='{pred_text}...'")
                if not pred_text or pred_text.strip() in ["<unk>", "<unk>.", "*"]:
                    raise ModelResponseError(f"预热检测失败：模型返回异常响应 '{pred_text}'")
            except ModelResponseError as e:
                print(f"\n❌ 预热检测失败！")
                print(f"错误: {e}")
                print(f"请检查：")
                print(f"  1. API endpoint是否正确")
                print(f"  2. 模型是否正常运行")
                print(f"  3. max_tokens设置是否合理（当前: {self.max_tokens}）")
                raise
        print("✓ 预热检测通过，开始正式评估...\n")
        
        # 重置统计计数器（预热结果不计入最终统计）
        self._stats = {"correct": 0, "total": 0}

        # 正式评估：评估所有样本（包括预热的3个）
        eval_examples = examples
        
        results: List[Dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_example = {
                executor.submit(self.evaluate_single, example): example for example in eval_examples
            }
            with tqdm(total=len(eval_examples), desc="NaturalQuestions", unit="题") as pbar:
                for future in as_completed(future_to_example):
                    result = future.result()
                    results.append(result)
                    acc = self._stats["correct"] / max(1, self._stats["total"]) * 100
                    pbar.update(1)
                    pbar.set_postfix(acc=f"{acc:.2f}%", correct=self._stats["correct"])

        accuracy = self._stats["correct"] / max(1, self._stats["total"]) * 100
        return {
            "total": self._stats["total"],
            "correct": self._stats["correct"],
            "accuracy": accuracy,
            "shot_num": self.num_shots,
            "results": results,
        }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="NaturalQuestions 评估脚本")
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
        help="数据集目录（默认: datasets/nq）",
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
        help="最大并发数（默认: 128）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="最大生成 token 数（默认: 256）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="随机采样样本数量，None 表示评估全部数据（默认: None）",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=5,
        help="Few-shot 示例数量，0 表示 zero-shot（默认: 5）",
    )

    args = parser.parse_args()

    data_root = args.dataset_dir or os.path.join(
        os.path.dirname(__file__), "datasets", "nq"
    )
    
    evaluator = NaturalQuestionsEvaluator(
        base_url=args.base_url,
        model=args.model,
        api_key="EMPTY",
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        data_root=data_root,
        num_shots=args.num_shots,
    )

    print(f"\n检查数据集: {data_root}")
    try:
        evaluator.ensure_local_files()
        print("✓ 数据集准备完成")
    except FileNotFoundError as e:
        print(f"⚠️  数据集文件不存在: {e}")
        print(f"请先运行以下命令下载数据集:")
        print(f"  python download_datasets.py nq")
        print(f"或者手动将数据集放置在: {data_root}")
        return
    except Exception as e:
        print(f"✗ 数据集检查失败: {e}")
        import traceback

        traceback.print_exc()
        return

    log_dir = os.path.join(os.path.dirname(__file__), "logs", args.model)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 70)
    print(f"开始评估 NaturalQuestions ({args.num_shots}-shot, EM)")
    eval_examples = evaluator.load_split(EVAL_SPLIT)
    print(f"加载到 {len(eval_examples)} 条 {EVAL_SPLIT} 样本")
    if args.max_samples:
        print(f"将随机采样 {args.max_samples} 条样本进行评估（seed={args.seed}）")

    result: Optional[Dict[str, object]] = None
    try:
        result = evaluator.evaluate_dataset(eval_examples, max_samples=args.max_samples, seed=args.seed)
        
        # 生成输出文件名
        if args.max_samples:
            output_filename = f"nq_{args.num_shots}shot_{args.max_samples}samples.json"
        else:
            output_filename = f"nq_{args.num_shots}shot.json"
        
        output_path = os.path.join(log_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存至 {output_path}")
    except ModelResponseError as exc:
        print(f"✗ NaturalQuestions 评估过程中检测到模型响应异常: {exc}")
    except KeyboardInterrupt:
        print("\n\n⚠️  评估被用户中断")
        if evaluator._stats["total"] > 0:
            partial_results = {
                "accuracy": evaluator._stats["correct"] / evaluator._stats["total"] * 100,
                "correct": evaluator._stats["correct"],
                "total": evaluator._stats["total"],
                "shot_num": args.num_shots,
                "note": "部分结果（已中断）",
            }
            partial_path = os.path.join(log_dir, f"nq_{args.num_shots}shot_partial.json")
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(partial_results, f, ensure_ascii=False, indent=2)
            print(f"部分结果已保存到: {partial_path}")

    print("=" * 70)
    if result is not None:
        print(
            json.dumps(
                {
                    "accuracy": result["accuracy"],
                    "total": result["total"],
                    "correct": result["correct"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print("未生成有效评估结果。")


if __name__ == "__main__":
    main()
