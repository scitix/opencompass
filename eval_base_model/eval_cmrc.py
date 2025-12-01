"""CMRC数据集评估脚本 - 1-shot生成方法（EM评估）。"""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from openai import OpenAI
from tqdm import tqdm
import eval_utils


class ModelResponseError(RuntimeError):
    """Raised when the model response is invalid."""


class CMRCEvaluator:
    """使用1-shot生成方法评估CMRC数据集"""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 max_workers: int = 32, max_tokens: int = 512, shot_num: int = 1):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.shot_num = shot_num
        self._correct_count = 0
        self._total_count = 0
        self._lock = threading.Lock()
        self._few_shot_example: Optional[Dict] = None

    def load_cmrc_data(self, test_file: str, dev_file: str = None) -> List[Dict]:
        """从JSON文件加载CMRC数据"""
        # 加载few-shot示例（从dev集选择1个）
        if dev_file and os.path.exists(dev_file):
            with open(dev_file, 'r', encoding='utf-8') as f:
                dev_data = json.load(f)
            
            # 提取第一个示例
            if isinstance(dev_data, dict) and 'data' in dev_data:
                for paragraphs in dev_data['data']:
                    for paragraph in paragraphs.get('paragraphs', []):
                        context = paragraph.get('context', '')
                        for qa in paragraph.get('qas', []):
                            question = qa.get('question', '')
                            answers = qa.get('answers', [])
                            if answers:
                                answer_text = answers[0].get('text', '')
                                self._few_shot_example = {
                                    'context': context,
                                    'question': question,
                                    'answer': answer_text,
                                }
                                break
                        if self._few_shot_example:
                            break
                    if self._few_shot_example:
                        break
        
        # 加载测试集
        data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        if isinstance(test_data, dict) and 'data' in test_data:
            for paragraphs in test_data['data']:
                for paragraph in paragraphs.get('paragraphs', []):
                    context = paragraph.get('context', '')
                    for qa in paragraph.get('qas', []):
                        question = qa.get('question', '')
                        answers = qa.get('answers', [])
                        unique_answers = list(set([a.get('text', '') for a in answers]))
                        
                        data.append({
                            'context': context,
                            'question': question,
                            'answers': unique_answers,
                        })
        
        return data

    def build_few_shot_prompt(self) -> str:
        """构建1-shot prompt"""
        if not self._few_shot_example:
            return ""
        
        example = self._few_shot_example
        prompt = f"""根据文章回答问题。你的答案应该尽可能简练，请以 '答案是' 开头的句式作答。
文章：{example['context']}
问：{example['question']}
答：答案是{example['answer']}

"""
        return prompt

    def build_prompt(self, context: str, question: str) -> str:
        """构建评估prompt（参考OpenCompass的CLUE_CMRC_gen_1bd3c8.py）"""
        few_shot = self.build_few_shot_prompt()
        prompt = f"""{few_shot}根据文章回答问题。你的答案应该尽可能简练，请以 '答案是' 开头的句式作答。
文章：{context}
问：{question}
答："""
        return prompt

    def postprocess_answer(self, text: str) -> str:
        """后处理答案（参考OpenCompass的cmrc_postprocess）"""
        # 提取"答案是"之后的内容
        if '答案是' in text:
            text = text.split('答案是')[1]
        
        # 清理文本
        text = text.strip()
        
        # 移除换行符后面的内容（这通常是重复的prompt）
        if '\n' in text:
            text = text.split('\n')[0]
        
        # 移除可能的标点符号
        text = re.sub(r'^[：:，,。.]+', '', text)
        text = re.sub(r'[：:，,。.]+$', '', text)
        
        return text.strip()

    def is_answer_correct(self, pred: str, ref_answers: List[str]) -> bool:
        """判断答案是否正确（EM评估）"""
        pred = pred.strip()
        for ref in ref_answers:
            ref = ref.strip()
            if pred == ref:
                return True
        return False

    def generate_answer(self, prompt: str, max_retries: int = 3) -> str:
        """生成答案"""
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    stop=None,
                )

                if not response.choices:
                    raise ModelResponseError('模型返回的choices为空。')

                generated_text = response.choices[0].text
                return self.postprocess_answer(generated_text)

            except Exception as e:
                error_msg = str(e)

                if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                    wait_time = min(10 * (attempt + 1), 60)
                    time.sleep(wait_time)
                    continue

                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    last_exception = e
                    break

        if last_exception is None:
            last_exception = ModelResponseError('未知原因导致生成失败。')

        raise last_exception

    def evaluate_single_question(self, question_data: Dict) -> Dict:
        """评估单个问题"""
        try:
            context = question_data['context']
            question = question_data['question']
            reference_answers = question_data['answers']

            prompt = self.build_prompt(context, question)
            predicted_answer = self.generate_answer(prompt)

            is_correct = self.is_answer_correct(predicted_answer, reference_answers)

            with self._lock:
                if is_correct:
                    self._correct_count += 1
                self._total_count += 1

            return {
                'context': context[:200] + '...' if len(context) > 200 else context,
                'question': question,
                'predicted_answer': predicted_answer,
                'reference_answers': reference_answers,
                'is_correct': is_correct,
            }
        except Exception as e:
            print(f"\n评估问题时出错: {e}")
            return {
                'question': question_data.get('question', ''),
                'predicted_answer': '',
                'reference_answers': question_data.get('answers', []),
                'is_correct': False,
                'error': str(e)
            }

    def evaluate_dataset(self, data: List[Dict], max_samples: int = None, seed: int = 42) -> Dict:
        """评估整个数据集"""
        import random

        if max_samples:
            random.seed(seed)
            data = random.sample(data, min(max_samples, len(data)))
            print(f"随机采样: {max_samples} 个样本（seed={seed}）")

        # 预热检测：先测试3个样本（不计入最终统计）
        num_warmup = min(3, len(data)) if not max_samples else min(3, max_samples, len(data))
        print(f"\n执行预热检测（测试 {num_warmup} 个样本）...")
        warmup_samples = data[:num_warmup]
        for i, sample in enumerate(warmup_samples):
            try:
                result = self.evaluate_single_question(sample)
                pred_text = result.get('predicted_answer', '')[:30]
                question_text = sample.get('question', '')[:50]
                print(f"✓ 样本{i+1}: 问题='{question_text}...' 预测='{pred_text}...'")
                if not pred_text or pred_text.strip() in ["<unk>", "<unk>.", ""]:
                    from openai import OpenAIError
                    raise OpenAIError(f"预热检测失败：模型返回异常响应 '{pred_text}'")
            except Exception as e:
                print(f"\n❌ 预热检测失败！")
                print(f"错误: {e}")
                print(f"请检查：")
                print(f"  1. API endpoint是否正确")
                print(f"  2. 模型是否正常运行")
                print(f"  3. max_tokens设置是否合理（当前: {self.max_tokens}）")
                raise
        print("✓ 预热检测通过，开始正式评估...\n")

        # 重置计数器（预热检测已完成，不计入正式统计）
        with self._lock:
            self._correct_count = 0
            self._total_count = 0

        # 正式评估：评估所有样本（包括预热的3个）
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.evaluate_single_question, item): item for item in data}

            with tqdm(total=len(data), desc="评估进度", unit="问题") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    pbar.set_postfix(accuracy=f"{self._correct_count / self._total_count * 100:.2f}%",
                                   correct=self._correct_count)

        return {
            'total': self._total_count,
            'correct': self._correct_count,
            'accuracy': self._correct_count / self._total_count * 100 if self._total_count > 0 else 0.0,
            'shot_num': self.shot_num,
            'seed': seed if max_samples else None,
            'max_samples': max_samples,
            'results': results
        }


def find_cmrc_data_path():
    """查找CMRC数据集路径（只从datasets目录查找）"""
    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'cmrc')
    test_file = os.path.join(datasets_dir, 'test_public.json')
    dev_file = os.path.join(datasets_dir, 'dev.json')
    
    if os.path.exists(test_file):
        return test_file, dev_file if os.path.exists(dev_file) else None
    
    return None, None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CMRC evaluation script")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("EVAL_MODEL_NAME", "qwen2-5-72b"),
        help="Model name (default: eval-qwen2-5-72b)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("EVAL_BASE_URL", "http://localhost:8000/v1"),
        help="API base URL",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Maximum number of concurrent workers (default: 32)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--shot-num",
        type=int,
        default=1,
        help="Number of few-shot examples (default: 1)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Random sample size, None means evaluate all data (default: None)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    
    args = parser.parse_args()

    # 自动查找数据路径
    TEST_FILE, DEV_FILE = find_cmrc_data_path()
    if TEST_FILE is None:
        print("⚠️  未找到CMRC数据集")
        print("   请确保数据集在以下位置：")
        print("   datasets/cmrc/test_public.json")
        return

    # 创建日志目录结构：logs/{model}/
    log_dir = os.path.join("logs", args.model)
    os.makedirs(log_dir, exist_ok=True)
    
    shot_desc = f"{args.shot_num}-shot" if args.shot_num > 0 else "0-shot"
    output_filename = f"cmrc_{args.shot_num}shot.json"
    if args.max_samples:
        output_filename = f"cmrc_{args.shot_num}shot_{args.max_samples}samples.json"
    OUTPUT_PATH = os.path.join(log_dir, output_filename)

    print("="*70)
    print(f"CMRC 评估 - {args.model} - {shot_desc} 生成方法")
    print("="*70)
    print(f"模型: {args.model}")
    print(f"API: completions")
    print(f"方法: {shot_desc} prompting + EM评估")
    print(f"Few-shot来源: datasets/cmrc/dev.json")
    print("="*70)

    evaluator = CMRCEvaluator(
        args.base_url,
        args.model,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        shot_num=args.shot_num
    )

    print(f"\n加载数据: {TEST_FILE}")
    if DEV_FILE:
        print(f"加载Few-shot示例: {DEV_FILE}")
    try:
        data = evaluator.load_cmrc_data(TEST_FILE, DEV_FILE)
        print(f"✓ 成功加载 {len(data)} 条测试数据")
        if evaluator._few_shot_example:
            print(f"✓ 成功加载 1 个few-shot示例")
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n开始评估...")

    try:
        results = evaluator.evaluate_dataset(data, max_samples=args.max_samples, seed=args.seed)

        print(f"\n保存结果: {OUTPUT_PATH}")
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("\n" + "="*70)
        print("评估结果")
        print("="*70)
        print(f"总问题数: {results['total']}")
        print(f"正确答案数: {results['correct']}")
        print(f"准确率: {results['accuracy']:.2f}%")
        print(f"Shot配置: {shot_desc}")
        if args.max_samples:
            print(f"随机采样: {args.max_samples} 个样本（seed={args.seed}）")

        print(f"\n详细结果已保存到: {OUTPUT_PATH}")
        print(f"日志目录: {log_dir}")
        print("="*70)

    except ModelResponseError as e:
        print(f"\n✗ 评估过程中检测到模型响应异常: {e}")
        print("评估已终止，未生成结果文件。")
    except KeyboardInterrupt:
        print("\n\n⚠️  评估被用户中断")
    except Exception as e:
        print(f"\n✗ 评估过程中出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()


