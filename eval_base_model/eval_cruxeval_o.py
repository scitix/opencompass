#!/usr/bin/env python3
"""CRUXEval-O 2-shot EM evaluation script.
Given function and input, find output。
"""

from __future__ import annotations
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import eval_utils


class ModelResponseError(RuntimeError):
    """Raised when the model response is invalid."""


@dataclass
class CruxExample:
    code: str
    input: str
    output: str
    id: str


class CRUXEvalOEvaluator:
    """Evaluate CRUXEval-O using 2-shot prompting and EM scoring.
    Given function code and input, find output。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        max_workers: int = 32,
        max_tokens: int = 512,
        shot_num: int = 1,
        data_root: str | None = None,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.shot_num = shot_num
        self.data_root = data_root or os.path.join(
            os.path.dirname(__file__), "datasets", "cruxeval"
        )

        self._lock = threading.Lock()
        self._stats = {"total": 0, "correct": 0}

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------
    def ensure_local_files(self) -> None:
        """Ensure CRUXEval jsonl files exist locally, download if missing."""
        jsonl_path = self._jsonl_path("test")
        if os.path.exists(jsonl_path):
            return

        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        dataset = load_dataset("cruxeval-org/cruxeval", split="test")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for sample in dataset:
                record = {
                    "code": sample.get("code", ""),
                    "input": sample.get("input", ""),
                    "output": sample.get("output", ""),
                    "id": sample.get("id", ""),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _jsonl_path(self, split: str) -> str:
        return os.path.join(self.data_root, f"{split}.jsonl")

    def load_split(self, split: str) -> List[CruxExample]:
        jsonl_path = self._jsonl_path(split)
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"Not found CRUXEval {split} data, please download to {jsonl_path}"
            )
        examples: List[CruxExample] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                examples.append(
                    CruxExample(
                        code=record.get("code", ""),
                        input=record.get("input", ""),
                        output=record.get("output", ""),
                        id=record.get("id", ""),
                    )
                )
        return examples

    # ------------------------------------------------------------------
    # Few-shot prompt construction
    # ------------------------------------------------------------------
    def _get_few_shot_examples(self) -> List[str]:
        """Return fixed 1-shot examples (simplified for Base model)"""
        return [
            """[PYTHON]
def f(s):
    return s + "a"
assert f("x9j") == ??
[/PYTHON]
[ANSWER]
assert f("x9j") == "x9ja"
[/ANSWER]""",
        ]

    def build_prompt(self, example: CruxExample) -> str:
        """Build evaluation prompt"""
        # Extract function name
        func_name = "f"
        match = re.search(r"def\s+(\w+)\s*\(", example.code)
        if match:
            func_name = match.group(1)

        # Build prompt
        intro = (
            "You are given a Python function and an assertion containing an input to the "
            "function. Complete the assertion with a literal (no unsimplified expressions, "
            "no function calls) containing the output when executing the provided code on "
            "the given input, even if the function is incorrect or incomplete. Do NOT output "
            "any extra information. Provide the full assertion with the correct output in "
            "[ANSWER] and [/ANSWER] tags, following the examples.\n"
        )

        segments = [intro]
        # Add fixed few-shot examples
        for shot in self._get_few_shot_examples():
            segments.append(shot)
            segments.append("")

        # Add test sample
        segments.extend(
            [
                "[PYTHON]",
                example.code,
                f"assert {func_name}({example.input}) == ??",
                "[/PYTHON]",
                "[ANSWER]",
            ]
        )

        prompt = "\n".join(segments).strip() + "\n"
        return prompt

    # ------------------------------------------------------------------
    # Model interaction & scoring
    # ------------------------------------------------------------------
    def _call_model(self, prompt: str, max_retries: int = 3) -> str:
        """Call model with strong stop conditions to prevent Base model continuation"""
        last_error: Optional[Exception] = None

        # Strong stop conditions: prevent Base model from generating new problems
        stop_sequences = [
            "[/ANSWER]",          # Answer end
            "\n\n[PYTHON]",       # New problem start
            "\ndef f(",           # New function definition
            "\n\n\n",             # Multiple empty lines
        ]

        for attempt in range(max_retries):
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    stop=stop_sequences,
                )
                if not response.choices:
                    raise ModelResponseError("Model returned empty choices。")
                return response.choices[0].text
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
            last_error = ModelResponseError("Model response failed for unknown reason。")
        if not isinstance(last_error, ModelResponseError):
            last_error = ModelResponseError(str(last_error))
        raise last_error

    def _extract_answer(self, raw_output: str, func_name: str, given_input: str, current_code: str) -> str:
        """Extract answer from model output（输出）- Optimized Base model version

        Base 模型OptimizedStrategy：
        1. Only take first answer (ignore continuation)
        2. Simplify matching logic
        3. Extract assert f(input) == output output part from
        """
        # Strategy1：Direct extract first assert statement output (simplest)
        # Match assert func_name(given_input) == output
        # 需要Escape special characters in input
        escaped_input = re.escape(str(given_input))
        pattern = rf"assert\s+{re.escape(func_name)}\s*\(\s*{escaped_input}\s*\)\s*==\s*(.+?)(?:\s*$|\s*\n|\[)"
        match = re.search(pattern, raw_output, re.DOTALL)
        if match:
            output = match.group(1).strip()
            # Clean output (remove possible markers)
            output = output.split('[')[0].strip()  # Remove [PYTHON] and other markers
            output = output.rstrip(',;')  # Remove trailing punctuation
            return output

        # Strategy2：Extract any assert func_name(...) == output (relaxed)
        pattern = rf"assert\s+{re.escape(func_name)}\s*\([^)]*\)\s*==\s*(.+?)(?:\s*$|\s*\n|\[)"
        match = re.search(pattern, raw_output, re.DOTALL)
        if match:
            output = match.group(1).strip()
            output = output.split('[')[0].strip()
            output = output.rstrip(',;')
            # Filter obviously wrong extractions (containing ?? or empty)
            if output and '??' not in output:
                return output

        # Strategy3：Find first line containing == content
        lines = raw_output.split('\n')
        for line in lines:
            if '==' in line and func_name in line:
                # Extract == 后面的部分
                parts = line.split('==', 1)
                if len(parts) == 2:
                    output = parts[1].strip()
                    output = output.split('[')[0].strip()
                    output = output.rstrip(',;')
                    if output and '??' not in output:
                        return output

        return ""

    def _normalize_output(self, text: str) -> str:
        """Normalize output string for comparison"""
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text.strip())
        # Normalize quotes
        text = text.replace("'", '"')
        return text

    def _is_correct(self, prediction: str, reference: str) -> bool:
        """Check if prediction is correct（EM）"""
        pred_norm = self._normalize_output(prediction)
        ref_norm = self._normalize_output(reference)
        return pred_norm == ref_norm

    def evaluate_single(self, example: CruxExample) -> Dict[str, object]:
        prompt = self.build_prompt(example)
        raw_answer = self._call_model(prompt)

        # Extract function name
        func_name = "f"
        match = re.search(r"def\s+(\w+)\s*\(", example.code)
        if match:
            func_name = match.group(1)

        pred_output = self._extract_answer(raw_answer, func_name, example.input, example.code)
        is_correct = self._is_correct(pred_output, example.output)

        with self._lock:
            self._stats["total"] += 1
            self._stats["correct"] += int(is_correct)

        return {
            "id": example.id,
            "code": example.code,
            "given_input": example.input,
            "expected_output": example.output,
            "pred_output": pred_output,
            "raw_model_output": raw_answer.strip(),
            "is_correct": is_correct,
        }

    def evaluate_dataset(
        self, examples: List[CruxExample], max_samples: Optional[int] = None
    ) -> Dict[str, object]:
        if max_samples is not None:
            random.seed(42)
            examples = random.sample(examples, min(max_samples, len(examples)))
            print(f"随机采样 {len(examples)} 条 CRUXEval-O 样本进行评估。")

        results: List[Dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_example = {
                executor.submit(self.evaluate_single, example): example for example in examples
            }
            with tqdm(total=len(examples), desc="CRUXEval-O", unit="question") as pbar:
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
            "shot_num": self.shot_num,
            "results": results,
        }


def main() -> None:
    import argparse
    import os

    # Load .env 文件
    def load_env():
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key, value = key.strip(), value.strip().strip('"').strip("'")
                        if key not in os.environ:
                            os.environ[key] = value

    load_env()

    # Read default config from environment variables
    default_model = os.environ.get("EVAL_MODEL_NAME", "qwen2-5-72b")
    default_base_url = os.environ.get("EVAL_BASE_URL", "http://localhost:8000/v1")
    default_api_key = os.environ.get("EVAL_API_KEY", "EMPTY")

    parser = argparse.ArgumentParser(description="CRUXEval-O Evaluation script")

    parser.add_argument(
        "--shot-num",
        type=int,
        default=1,
        help=f"Few-shot Number of examples（default: 1）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help=f"Model name（default: {default_model}，can be set via EVAL_MODEL_NAME environment variable）",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=default_base_url,
        help=f"API base URL（can be set via EVAL_BASE_URL environment variable）",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=default_api_key,
        help=f"API API key（can be set via EVAL_API_KEY environment variable）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Max workers（default: 32）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens to generate token count (default: 512）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Random sample size，None means evaluate all data（default: None）",
    )

    args = parser.parse_args()

    evaluator = CRUXEvalOEvaluator(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        shot_num=args.shot_num,
    )

    evaluator.ensure_local_files()

    log_dir = os.path.join(os.path.dirname(__file__), "logs", args.model)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 70)
    print(f"{args.shot_num}-shot, EM)")
    eval_examples = evaluator.load_split("test")
    print(f"Loadloaded {len(eval_examples)} test samples")
    if args.max_samples:
        print(f"Will randomly sample {args.max_samples} samples for evaluation（seed=42）")

    result: Optional[Dict[str, object]] = None
    try:
        result = evaluator.evaluate_dataset(eval_examples, max_samples=args.max_samples)

        # Generate output filename
        if args.max_samples:
            output_filename = f"cruxeval_o_{args.shot_num}shot_{args.max_samples}samples.json"
        else:
            output_filename = f"cruxeval_o_{args.shot_num}shot.json"

        output_path = os.path.join(log_dir, output_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_path}")
    except ModelResponseError as exc:
        print(f"✗ CRUXEval-O Model response error detected during evaluation: {exc}")

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
        print("未生成有效Evaluation Results。")


if __name__ == "__main__":
    main()

