#!/usr/bin/env python3
"""MMLU evaluation script using PPL method."""

import json
import numpy as np
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
from openai import OpenAI
from tqdm import tqdm

DEFAULT_MMLU_DIR = os.path.join(os.path.dirname(__file__), "datasets", "cais_mmlu")


class ModelResponseError(RuntimeError):
    """Raised when the model response does not contain usable logprob data."""


class MMLUEvaluator:
    """Evaluate MMLU dataset using PPL method."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        max_workers: int = 128,
        max_workers_per_question: int = 4,
        dataset_dir: Optional[str] = None,
        shot_num: int = 5,
        max_tokens: int = 1,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_workers_per_question = min(max_workers_per_question, 4)
        self.dataset_dir = dataset_dir or DEFAULT_MMLU_DIR
        self.shot_num = shot_num
        self.max_tokens = max_tokens
        self._correct_count = 0
        self._total_count = 0
        self._failed_extractions = 0
        self._lock = threading.Lock()
        self._dataset_loaded = False
        self._fewshot_by_subject: Dict[str, List[Dict]] = {}
        self._few_shot_cache: Dict[str, str] = {}
        self._test_data: List[Dict] = []

    def _load_dataset(self):
        """Load local MMLU dataset and build index."""
        if self._dataset_loaded:
            return

        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(
                f"MMLU dataset not found at {self.dataset_dir}, please download cais/mmlu first."
            )

        dev_records = self._load_split("dev")
        if not dev_records:
            dev_records = self._load_split("validation")
        if not dev_records:
            raise FileNotFoundError(
                f"No 'dev' or 'validation' split found in {self.dataset_dir}, "
                "cannot build few-shot prompts."
            )

        test_records = self._load_split("test")
        if not test_records:
            raise FileNotFoundError(
                f"No 'test' split found in {self.dataset_dir}, cannot evaluate."
            )

        self._fewshot_by_subject = self._group_by_subject(dev_records)
        self._test_data = self._convert_split(test_records)

        for subject in list(self._fewshot_by_subject.keys()):
            self._few_shot_cache[subject] = self._build_few_shot_prompt(subject)

        if "miscellaneous" not in self._few_shot_cache:
            self._few_shot_cache["miscellaneous"] = self._build_few_shot_prompt("miscellaneous")

        self._dataset_loaded = True

    def _load_split(self, split_name: str) -> List[Dict]:
        split_dir = os.path.join(self.dataset_dir, split_name)
        if not os.path.isdir(split_dir):
            return []

        records: List[Dict] = []
        jsonl_files = [
            fname for fname in os.listdir(split_dir)
            if fname.endswith('.jsonl')
        ]

        if jsonl_files:
            for filename in sorted(jsonl_files):
                subject = os.path.splitext(filename)[0]
                file_path = os.path.join(split_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        sample = json.loads(line)
                        self._append_record(records, sample, subject)
        else:
            arrow_files = [fname for fname in os.listdir(split_dir) if fname.endswith(".arrow")]
            for filename in sorted(arrow_files):
                file_path = os.path.join(split_dir, filename)
                with pa.memory_map(file_path, "r") as source:
                    try:
                        reader = pa.ipc.open_file(source)
                    except pa.ArrowInvalid:
                        reader = pa.ipc.open_stream(source)
                    table = reader.read_all()
                for sample in table.to_pylist():
                    self._append_record(records, sample)
        return records

    def _append_record(
        self, records: List[Dict], sample: Dict, default_subject: Optional[str] = None
    ):
        question = sample.get("question") or sample.get("input")
        choices = sample.get("choices") or [
            sample.get("A"),
            sample.get("B"),
            sample.get("C"),
            sample.get("D"),
        ]
        if not question or not choices or len(choices) != 4:
            return
        choices = ["" if c is None else str(c) for c in choices]
        answer = sample.get("answer")
        if answer is None:
            answer = sample.get("target")
        if answer is None:
            return
        subject = sample.get("subject") or default_subject or "miscellaneous"

        answer_label = self._normalize_answer_label(answer)
        if answer_label is None:
            return

        records.append(
            {"question": str(question), "choices": choices, "answer": answer_label, "subject": subject}
        )

    @staticmethod
    def _normalize_answer_label(answer) -> Optional[str]:
        if isinstance(answer, int):
            if 0 <= answer < 4:
                return ["A", "B", "C", "D"][answer]
            return None
        if isinstance(answer, str):
            ans = answer.strip().upper()
            if ans in ["A", "B", "C", "D"]:
                return ans
            try:
                idx = int(ans)
                if 0 <= idx < 4:
                    return ["A", "B", "C", "D"][idx]
            except Exception:
                return None
        return None

    @staticmethod
    def _group_by_subject(records: List[Dict]) -> Dict[str, List[Dict]]:
        grouped: Dict[str, List[Dict]] = {}
        for row in records:
            subject = row.get("subject", "miscellaneous") or "miscellaneous"
            grouped.setdefault(subject, []).append(row)
        return grouped

    @staticmethod
    def _convert_split(records: List[Dict]) -> List[Dict]:
        converted: List[Dict] = []
        for row in records:
            choices = row["choices"]
            if len(choices) != 4:
                continue
            converted.append(
                {
                    "question": row["question"],
                    "A": choices[0],
                    "B": choices[1],
                    "C": choices[2],
                    "D": choices[3],
                    "answer": row["answer"],
                    "subject": row.get("subject", "miscellaneous"),
                }
            )
        return converted

    def _select_examples(self, subject: str) -> List[Dict]:
        subject_examples = list(self._fewshot_by_subject.get(subject, []))
        if len(subject_examples) >= self.shot_num:
            return subject_examples[: self.shot_num]

        supplemental: List[Dict] = []
        if subject != "miscellaneous":
            supplemental = self._fewshot_by_subject.get("miscellaneous", [])

        combined = subject_examples + [ex for ex in supplemental if ex not in subject_examples]
        return combined[: self.shot_num]

    def _build_few_shot_prompt(self, subject: str) -> str:
        examples = self._select_examples(subject)
        if not examples:
            return ""

        lines = [f"The following are multiple choice questions (with answers) about {subject}.\n"]
        option_labels = ["A", "B", "C", "D"]
        for example in examples:
            lines.append(example["question"])
            for idx, choice_text in enumerate(example["choices"]):
                lines.append(f"{option_labels[idx]}. {choice_text}")
            lines.append(f"Answer: {example['answer']}")
            lines.append("")

        return "\n".join(lines).strip() + "\n\n"

    def load_mmlu_data(self) -> List[Dict]:
        """Load test set and prepare few-shot examples."""
        self._load_dataset()
        return list(self._test_data)

    def get_few_shot_examples(self, subject: str = "miscellaneous") -> str:
        """Get few-shot examples from local dev/validation set, grouped by subject."""
        self._load_dataset()
        subject_key = subject or "miscellaneous"
        if subject_key not in self._few_shot_cache:
            self._few_shot_cache[subject_key] = self._build_few_shot_prompt(subject_key)
        return self._few_shot_cache.get(subject_key, "")

    def build_prompt(
        self,
        question: str,
        option_a: str,
        option_b: str,
        option_c: str,
        option_d: str,
        option_label: str,
        subject: str = "miscellaneous",
    ) -> str:
        """Build evaluation prompt with few-shot examples."""
        few_shot = self.get_few_shot_examples(subject)

        prompt = f"""{few_shot}{question}
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
Answer: {option_label}"""
        return prompt
    
    def extract_option_logprob(
        self, tokens: List[str], token_logprobs: List[Optional[float]], option_label: str
    ) -> Optional[float]:
        """Extract option label logprob from token list. Search backwards as answer is at the end."""
        for i in range(len(tokens) - 1, max(0, len(tokens) - 15), -1):
            token = tokens[i]
            token_stripped = token.strip()

            if token_stripped == option_label:
                if i < len(token_logprobs) and token_logprobs[i] is not None:
                    return token_logprobs[i]

            if token == f" {option_label}":
                if i < len(token_logprobs) and token_logprobs[i] is not None:
                    return token_logprobs[i]

            if option_label in token_stripped and len(token_stripped) <= 2:
                if i < len(token_logprobs) and token_logprobs[i] is not None:
                    return token_logprobs[i]

        return None

    def get_option_logprob(
        self, prompt: str, option_label: str, max_tokens: int = 1, max_retries: int = 3
    ) -> float:
        """Get option logprob using completions API with echo=True."""
        last_exception: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    logprobs=5,
                    echo=True,
                    temperature=0,
                )

                if not response.choices:
                    raise ModelResponseError("Model returned empty choices.")

                logprobs_obj = response.choices[0].logprobs
                if logprobs_obj is None:
                    raise ModelResponseError("Model returned empty logprobs.")

                tokens = logprobs_obj.tokens
                token_logprobs = logprobs_obj.token_logprobs
                if not tokens or not token_logprobs:
                    raise ModelResponseError("Model returned empty tokens/logprobs.")

                option_logprob = self.extract_option_logprob(tokens, token_logprobs, option_label)

                if option_logprob is not None:
                    return option_logprob

                raise ModelResponseError(f"Failed to extract logprob for option {option_label}.")

            except Exception as e:
                error_msg = str(e)

                if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                    wait_time = min(10 * (attempt + 1), 60)
                    time.sleep(wait_time)
                    continue

                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    last_exception = (
                        e if isinstance(e, ModelResponseError) else ModelResponseError(error_msg)
                    )
                    break

        if last_exception is None:
            last_exception = ModelResponseError("Unknown error in logprob calculation.")

        raise last_exception
    
    def _get_logprob_for_label(
        self,
        question: str,
        option_a: str,
        option_b: str,
        option_c: str,
        option_d: str,
        label: str,
        subject: str = "miscellaneous",
    ) -> Tuple[str, float]:
        """Get logprob for a single option (for concurrent calls)."""
        prompt = self.build_prompt(question, option_a, option_b, option_c, option_d, label, subject)
        logprob = self.get_option_logprob(prompt, label, self.max_tokens)
        return label, logprob

    def evaluate_single_question(self, question_data: Dict) -> Tuple[str, Dict, Dict]:
        """Evaluate a single question (concurrently get logprobs for 4 options)."""
        question = question_data["question"]
        subject = question_data.get("subject", "miscellaneous")

        option_logprobs = {}
        option_ppls = {}

        with ThreadPoolExecutor(max_workers=self.max_workers_per_question) as executor:
            futures = {
                executor.submit(
                    self._get_logprob_for_label,
                    question,
                    question_data["A"],
                    question_data["B"],
                    question_data["C"],
                    question_data["D"],
                    label,
                    subject,
                ): label
                for label in ["A", "B", "C", "D"]
            }

            for future in as_completed(futures):
                try:
                    label, logprob = future.result()
                    option_logprobs[label] = logprob
                    option_ppls[label] = np.exp(-logprob)
                except ModelResponseError:
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    raise
                except Exception as e:
                    label = futures[future]
                    print(f"\nError getting logprob for option {label}: {e}")
                    option_logprobs[label] = -10.0
                    option_ppls[label] = np.exp(10.0)
                    with self._lock:
                        self._failed_extractions += 1

        for label in ["A", "B", "C", "D"]:
            if label not in option_logprobs:
                option_logprobs[label] = -10.0
                option_ppls[label] = np.exp(10.0)

        predicted = max(option_logprobs, key=option_logprobs.get)

        return predicted, option_logprobs, option_ppls
    
    def _evaluate_single_question_with_result(self, question_data: Dict) -> Dict:
        """Evaluate a single question and return result dict."""
        try:
            predicted, option_logprobs, option_ppls = self.evaluate_single_question(question_data)
            correct_answer = question_data["answer"].strip().upper()
            is_correct = predicted == correct_answer

            with self._lock:
                if is_correct:
                    self._correct_count += 1
                self._total_count += 1

            return {
                "question": question_data["question"],
                "options": {
                    "A": question_data["A"],
                    "B": question_data["B"],
                    "C": question_data["C"],
                    "D": question_data["D"],
                },
                "predicted": predicted,
                "correct": correct_answer,
                "is_correct": is_correct,
                "logprobs": option_logprobs,
                "ppls": option_ppls,
                "subject": question_data["subject"],
            }
        except ModelResponseError:
            raise
        except Exception as e:
            print(f"\nError evaluating question: {e}")
            return {
                "question": question_data.get("question", ""),
                "predicted": "ERROR",
                "correct": question_data.get("answer", ""),
                "is_correct": False,
                "error": str(e),
            }

    def evaluate_dataset(self, data: List[Dict], max_samples: Optional[int] = None, seed: int = 42) -> Dict:
        """Evaluate entire dataset."""
        if max_samples:
            random.seed(seed)
            data = random.sample(data, min(max_samples, len(data)))

        total = len(data)
        results = []

        self._correct_count = 0
        self._total_count = 0
        self._failed_extractions = 0

        print(f"Starting evaluation of {total} questions...")
        print(
            f"Concurrency: {self.max_workers} questions, "
            f"{self.max_workers_per_question} options per question"
        )
        print("Method: PPL (completions API + echo=True)")

        pbar = tqdm(total=total, desc="Evaluating")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_data = {
                executor.submit(self._evaluate_single_question_with_result, q): q for q in data
            }

            for future in as_completed(future_to_data):
                try:
                    result = future.result()
                    results.append(result)
                    pbar.update(1)

                    with self._lock:
                        current_total = self._total_count
                        current_correct = self._correct_count
                        failed = self._failed_extractions

                    if current_total % 100 == 0 and current_total > 0:
                        current_acc = current_correct / current_total * 100
                        pbar.set_postfix(
                            {"accuracy": f"{current_acc:.2f}%", "correct": current_correct, "failed": failed}
                        )
                except ModelResponseError as e:
                    print(f"\nModel response error, evaluation aborted: {e}")
                    for f in future_to_data:
                        if not f.done():
                            f.cancel()
                    pbar.close()
                    raise
                except Exception as e:
                    print(f"\nEvaluation failed: {e}")
                    pbar.update(1)

        pbar.close()

        correct = self._correct_count
        accuracy = correct / total * 100 if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "failed_extractions": self._failed_extractions,
            "shot_num": self.shot_num,
            "seed": seed if max_samples else None,
            "max_samples": max_samples,
            "results": results,
        }

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MMLU evaluation script")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2-5-72b",
        help="Model name (default: qwen2-5-72b)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://172.18.178.129:8000/v1",
        help="API base URL",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=128,
        help="Maximum number of concurrent workers (default: 128)",
    )
    parser.add_argument(
        "--max-workers-per-question",
        type=int,
        default=4,
        help="Maximum number of concurrent workers per question (default: 4)",
    )
    parser.add_argument(
        "--shot-num",
        type=int,
        default=5,
        help="Number of few-shot examples (default: 5)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Random sample size, None means evaluate all data (default: None)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1,
        help="Maximum number of tokens to generate (default: 1)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help=f"Dataset directory (default: {DEFAULT_MMLU_DIR})",
    )

    args = parser.parse_args()

    log_dir = os.path.join(os.path.dirname(__file__), "logs", args.model)
    os.makedirs(log_dir, exist_ok=True)

    shot_desc = f"{args.shot_num}-shot" if args.shot_num > 0 else "0-shot"
    output_filename = f"mmlu_{args.shot_num}shot.json"
    if args.max_samples:
        output_filename = f"mmlu_{args.shot_num}shot_{args.max_samples}samples.json"
    output_path = os.path.join(log_dir, output_filename)

    print("=" * 70)
    print(f"MMLU Evaluation - {args.model} - {shot_desc} PPL Method")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"API: completions + echo=True")
    print(f"Method: {shot_desc} few-shot prompting + PPL")
    print("=" * 70)

    evaluator = MMLUEvaluator(
        base_url=args.base_url,
        model=args.model,
        api_key="EMPTY",
        max_workers=args.max_workers,
        max_workers_per_question=args.max_workers_per_question,
        dataset_dir=args.dataset_dir or DEFAULT_MMLU_DIR,
        shot_num=args.shot_num,
        max_tokens=args.max_tokens,
    )

    print(f"\nLoading dataset: {evaluator.dataset_dir}")
    try:
        data = evaluator.load_mmlu_data()
        print(f"✓ Successfully loaded {len(data)} samples")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return

    if args.max_samples:
        print(f"Will randomly sample {args.max_samples} samples (seed=42)")

    print("\nStarting evaluation...")

    try:
        results = evaluator.evaluate_dataset(data, max_samples=args.max_samples, seed=42)

        print(f"\nSaving results: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 70)
        print("Evaluation Results")
        print("=" * 70)
        print(f"Total questions: {results['total']}")
        print(f"Correct answers: {results['correct']}")
        print(f"Accuracy: {results['accuracy']:.2f}%")
        print(f"Shot configuration: {shot_desc}")

        if results.get("failed_extractions", 0) > 0:
            fail_rate = results["failed_extractions"] / (results["total"] * 4) * 100
            print(f"\nLogprob extraction failures: {results['failed_extractions']}")
            print(f"Failure rate: {fail_rate:.2f}%")
            if fail_rate > 5:
                print("⚠️  High failure rate, may affect accuracy")

        if results["results"]:
            subject_stats = {}
            for result in results["results"]:
                subject = result.get("subject", "unknown")
                if subject not in subject_stats:
                    subject_stats[subject] = {"correct": 0, "total": 0}
                subject_stats[subject]["total"] += 1
                if result.get("is_correct"):
                    subject_stats[subject]["correct"] += 1

            print("\nSubject statistics (Top 10):")
            sorted_subjects = sorted(
                subject_stats.items(),
                key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0,
                reverse=True,
            )
            for subject, stats in sorted_subjects[:10]:
                acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
                print(f"  {subject}: {stats['correct']}/{stats['total']} = {acc:.2f}%")

            if len(sorted_subjects) > 10:
                print(f"  ... {len(sorted_subjects) - 10} more subjects")

        print(f"\nDetailed results saved to: {output_path}")
        print(f"Log directory: {log_dir}")
        print("=" * 70)

    except ModelResponseError as e:
        print(f"\n✗ Model response error detected: {e}")
        print("Evaluation terminated, no result file generated.")
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        if evaluator._total_count > 0:
            partial_results = {
                "accuracy": evaluator._correct_count / evaluator._total_count * 100,
                "correct": evaluator._correct_count,
                "total": evaluator._total_count,
                "shot_num": args.shot_num,
                "note": "Partial results (interrupted)",
            }
            partial_path = output_path.replace(".json", "_partial.json")
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(partial_results, f, ensure_ascii=False, indent=2)
            print(f"Partial results saved to: {partial_path}")
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
