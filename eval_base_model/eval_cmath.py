#!/usr/bin/env python3
"""
CMATH 评估脚本
数据集: Chinese Elementary School Math Word Problems (CMATH)
来源: https://huggingface.co/datasets/weitianwen/cmath
评估方法: 3-shot EM (Exact Match)
"""

import argparse
import json
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from openai import OpenAI
from tqdm import tqdm
import eval_utils
class ModelResponseError(Exception):
    """模型响应错误"""
    pass


class CMATHEvaluator:
    def __init__(self, base_url: str, model: str, max_workers: int = 32, 
                 max_tokens: int = 512, shot_num: int = 3):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.shot_num = shot_num
        self._correct_count = 0
        self._total_count = 0
        self._lock = threading.Lock()
        self._few_shot_examples: List[Dict] = []

    def load_cmath_data(self, data_path: str) -> List[Dict]:
        """从本地JSON文件加载CMATH数据"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CMATH数据文件不存在: {data_path}")
        
        print(f"正在从本地加载 CMATH 数据集: {data_path}")
        
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                data.append({
                    'grade': item.get('grade', 0),
                    'question': item.get('question', ''),
                    'answer': str(item.get('golden', '')),  # 转为字符串
                    'reasoning_step': item.get('reasoning_step', 0),
                    'num_digits': item.get('num_digits', 0),
                })
        
        return data

    def load_few_shot_examples(self, validation_path: str, num_shots: int = 3):
        """从本地validation文件加载few-shot示例"""
        if num_shots == 0:
            self._few_shot_examples = []
            return
        
        print(f"加载 {num_shots}-shot 示例...")
        if not os.path.exists(validation_path):
            print(f"⚠️ validation文件不存在: {validation_path}")
            self._few_shot_examples = []
            return
        
        try:
            self._few_shot_examples = []
            with open(validation_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= num_shots:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    self._few_shot_examples.append({
                        'question': item.get('question', ''),
                        'answer': str(item.get('golden', '')),
                    })
            
            print(f"✓ 成功加载 {len(self._few_shot_examples)} 个few-shot示例")
        except Exception as e:
            print(f"⚠️ 加载few-shot示例失败: {e}")
            self._few_shot_examples = []

    def build_few_shot_prompt(self) -> str:
        """构建few-shot prompt"""
        if not self._few_shot_examples:
            return ""
        
        prompt_parts = []
        for example in self._few_shot_examples:
            prompt_parts.append(
                f"问题：{example['question']}\n"
                f"答案：{example['answer']}\n"
            )
        
        return "\n".join(prompt_parts) + "\n"

    def build_prompt(self, question: str) -> str:
        """构建评估prompt"""
        few_shot = self.build_few_shot_prompt()
        
        prompt = f"""{few_shot}问题：{question}
答案："""
        return prompt

    def extract_answer(self, text: str) -> Optional[str]:
        """从生成的文本中提取数字答案
        
        策略：优先提取第一行的数字（因为模型生成格式是：答案在第一行）
        """
        text = text.strip()
        
        # 策略1：提取第一行的数字（最可靠）
        first_line = text.split('\n')[0].strip()
        numbers_in_first_line = re.findall(r'-?\d+\.?\d*', first_line)
        if numbers_in_first_line:
            # 返回第一行的第一个数字
            answer = numbers_in_first_line[0]
            # 移除末尾的小数点（如果有）
            if answer.endswith('.'):
                answer = answer[:-1]
            return answer
        
        # 策略2：如果第一行没有数字，提取全文的第一个数字
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            answer = numbers[0]
            if answer.endswith('.'):
                answer = answer[:-1]
            return answer
        
        return None

    def is_answer_correct(self, pred: str, ref: str) -> bool:
        """判断答案是否正确（精确匹配）"""
        if pred is None or ref is None:
            return False
        
        # 规范化答案（移除空格，统一格式）
        pred = str(pred).strip().replace(' ', '')
        ref = str(ref).strip().replace(' ', '')
        
        # 尝试转换为数字比较
        try:
            pred_num = float(pred)
            ref_num = float(ref)
            # 使用小的容差进行比较
            return abs(pred_num - ref_num) < 1e-6
        except ValueError:
            # 如果无法转换为数字，则进行字符串比较
            return pred == ref

    def generate_answer(self, prompt: str, max_retries: int = 3) -> str:
        """生成答案"""
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                stop_sequences = ["\n\n问题：", "\n问题："]
                
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    stop=stop_sequences,
                )

                if not response.choices or not response.choices[0].text.strip():
                    raise ModelResponseError("模型返回空响应。")
                
                generated_text = response.choices[0].text.strip()
                
                if "<unk>" in generated_text.lower() or len(generated_text) < 1:
                    raise ModelResponseError(f"模型返回无效token: {generated_text}")
                
                return generated_text

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
            question = question_data['question']
            reference_answer = question_data['answer']

            prompt = self.build_prompt(question)
            generated_text = self.generate_answer(prompt)
            
            # 提取答案
            predicted_answer = self.extract_answer(generated_text)
            
            # 判断正确性
            is_correct = self.is_answer_correct(predicted_answer, reference_answer)

            with self._lock:
                if is_correct:
                    self._correct_count += 1
                self._total_count += 1

            return {
                'question': question,
                'generated_text': generated_text,
                'predicted_answer': predicted_answer,
                'reference_answer': reference_answer,
                'correct': is_correct,
                'grade': question_data.get('grade', 0),
            }
        except Exception as e:
            print(f"\n评估问题时出错: {e}")
            return {
                'question': question_data.get('question', ''),
                'predicted_answer': '',
                'reference_answer': question_data.get('answer', ''),
                'correct': False,
                'error': str(e),
            }

    def _warmup_check(self, examples: List[Dict], num_warmup: int = 3) -> None:
        """预热检测：在正式评估前检测模型响应是否正常"""
        print(f"\n执行预热检测（测试{num_warmup}个样本）...")
        
        warmup_samples = examples[:min(num_warmup, len(examples))]
        failed_count = 0
        
        for i, sample in enumerate(warmup_samples, 1):
            try:
                prompt = self.build_prompt(sample['question'])
                generated_text = self.generate_answer(prompt, max_retries=2)
                
                # 检查生成的文本是否有效
                if not generated_text or len(generated_text) < 1:
                    failed_count += 1
                    print(f"✗ 样本{i}: 生成为空")
                elif "<unk>" in generated_text.lower():
                    failed_count += 1
                    print(f"✗ 样本{i}: 包含<unk>")
                else:
                    print(f"✓ 样本{i}: 测试通过")
                    
            except Exception as e:
                failed_count += 1
                print(f"✗ 样本{i}: 异常 - {e}")
        
        if failed_count > 0:
            print(f"\n⚠️ 预热检测发现 {failed_count}/{num_warmup} 个样本失败")
            print("建议检查：")
            print("  1. 模型服务是否正常运行")
            print("  2. base_url 和 model 名称是否正确")
            print("  3. 网络连接是否稳定")
            
            response = input("\n是否继续评估？(y/n): ")
            if response.lower() != 'y':
                print("评估已取消")
                exit(0)
        else:
            print("✓ 预热检测通过，开始正式评估...\n")

        # 重置计数器（预热检测已完成，不计入正式统计）
        self._correct_count = 0
        self._total_count = 0

        # 正式评估：评估所有样本（预热检测不跳过任何数据）
        eval_examples = examples
                
    def evaluate_dataset(self, data: List[Dict], max_samples: Optional[int] = None, 
                        seed: int = 42) -> Dict:
        """评估整个数据集"""
        if max_samples is not None and max_samples < len(data):
            import random

            random.seed(seed)
            data = random.sample(data, max_samples)
            print(f"随机采样 {max_samples} 个样本进行评估（seed={seed}）")
        
        # 执行预热检测
        self._warmup_check(data)
        
        print(f"开始评估 {len(data)} 个问题...")
        print(f"并发设置: {self.max_workers} workers")
        print(f"评估方法: {self.shot_num}-shot EM\n")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.evaluate_single_question, q): q 
                for q in data
            }
            
            pbar = tqdm(total=len(data), desc="评估进度")
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 更新进度条
                    accuracy = (self._correct_count / self._total_count * 100) if self._total_count > 0 else 0.0
                    pbar.set_postfix({
                        'accuracy': f'{accuracy:.2f}%',
                        'correct': self._correct_count
                    })
                    pbar.update(1)
                except Exception as e:
                    print(f"\n处理future时出错: {e}")
                    pbar.update(1)
            
            pbar.close()
        
        accuracy = (self._correct_count / self._total_count * 100) if self._total_count > 0 else 0.0
        
        # 按年级统计
        grade_stats = {}
        for result in results:
            grade = result.get('grade', 0)
            if grade not in grade_stats:
                grade_stats[grade] = {'total': 0, 'correct': 0}
            grade_stats[grade]['total'] += 1
            if result.get('correct', False):
                grade_stats[grade]['correct'] += 1
        
        # 计算每个年级的准确率
        for grade in grade_stats:
            stats = grade_stats[grade]
            stats['accuracy'] = f"{(stats['correct'] / stats['total'] * 100):.2f}%" if stats['total'] > 0 else "0.00%"
        
        return {
            'accuracy': f"{accuracy:.2f}%",
            'correct': self._correct_count,
            'total': self._total_count,
            'shot_num': self.shot_num,
            'seed': seed,
            'max_samples': max_samples,
            'grade_stats': grade_stats,
            'results': results
        }


def find_cmath_data_path(split: str = 'test'):
    """查找CMATH数据集路径"""
    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'cmath')
    data_path = os.path.join(datasets_dir, f'{split}.jsonl')
    
    if os.path.exists(data_path):
        return data_path
    
    return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="CMATH evaluation script")
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
        "--max-workers",
        type=int,
        default=32,
        help="最大并发数（默认: 32）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="最大生成 token 数（默认: 512）",
    )
    parser.add_argument(
        "--shot-num",
        type=int,
        default=3,
        help="Few-shot示例数量（默认: 3-shot）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="随机采样数量，None表示评估全部数据（默认: None）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )
    
    args = parser.parse_args()
    
    # 查找数据路径
    test_path = find_cmath_data_path('test')
    validation_path = find_cmath_data_path('validation')
    
    if test_path is None:
        print("⚠️  未找到CMATH测试集文件")
        print("   请确保数据文件在以下位置：")
        print("   datasets/cmath/test.jsonl")
        print("\n   使用以下命令下载数据集：")
        print("   python3 download_datasets.py --dataset cmath")
        return
    
    # 创建日志目录结构：logs/{model}/
    log_dir = os.path.join("logs", args.model)
    os.makedirs(log_dir, exist_ok=True)
    
    shot_desc = f"{args.shot_num}shot" if args.shot_num > 0 else "0shot"
    output_filename = f"cmath_{shot_desc}.json"
    if args.max_samples:
        output_filename = f"cmath_{shot_desc}_{args.max_samples}samples.json"
    OUTPUT_PATH = os.path.join(log_dir, output_filename)
    
    print("="*70)
    print(f"CMATH 评估 - {args.model} - {args.shot_num}-shot EM")
    print("="*70)
    print(f"模型: {args.model}")
    print(f"API: completions")
    print(f"方法: {args.shot_num}-shot EM (Exact Match)")
    print(f"数据集: CMATH test split")
    print(f"数据路径: {test_path}")
    print("="*70)
    
    evaluator = CMATHEvaluator(
        args.base_url, 
        args.model, 
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        shot_num=args.shot_num
    )
    
    # 加载few-shot示例
    if args.shot_num > 0 and validation_path:
        evaluator.load_few_shot_examples(validation_path, args.shot_num)
    
    # 加载测试数据
    print(f"\n加载测试数据...")
    try:
        data = evaluator.load_cmath_data(test_path)
        print(f"✓ 成功加载 {len(data)} 条测试数据")
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        return
    
    print("\n开始评估...\n")
    results = evaluator.evaluate_dataset(data, args.max_samples, args.seed)
    
    # 保存结果
    print(f"\n保存结果: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印结果
    print("\n" + "="*70)
    print("评估结果")
    print("="*70)
    print(f"总问题数: {results['total']}")
    print(f"正确答案数: {results['correct']}")
    print(f"准确率: {results['accuracy']}")
    print(f"Shot配置: {args.shot_num}-shot")
    if args.max_samples:
        print(f"随机采样: {args.max_samples} 个样本（seed={args.seed}）")
    
    # 打印各年级的表现
    if 'grade_stats' in results and results['grade_stats']:
        print("\n按年级统计:")
        for grade in sorted(results['grade_stats'].keys()):
            stats = results['grade_stats'][grade]
            print(f"  年级{grade}: {stats['accuracy']} ({stats['correct']}/{stats['total']})")
    
    print(f"\n详细结果已保存到: {OUTPUT_PATH}")
    print("="*70)


if __name__ == "__main__":
    main()

