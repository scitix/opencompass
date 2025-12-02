"""C3数据集评估脚本 - 0-shot生成方法。"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
import threading
import eval_utils
class ModelResponseError(RuntimeError):
    """Raised when the model response is invalid."""


class C3Evaluator:
    """使用0-shot生成方法评估C3数据集"""
    
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", 
                 max_workers: int = 32, max_tokens: int = 512):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self._correct_count = 0
        self._total_count = 0
        self._lock = threading.Lock()
        

    # ========== PPL评估方法 ==========
    def extract_option_logprob(
        self, tokens: List[str], token_logprobs: List[Optional[float]], option_label: str
    ) -> Optional[float]:
        """从token列表中提取选项标签的logprob"""
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

    def get_option_logprob_with_length(self, prompt: str, option_label: str, max_retries: int = 3) -> Tuple[float, int]:
        """获取选项的logprob和token长度（用于长度归一化）"""
        last_exception: Optional[Exception] = None

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
                    raise ModelResponseError('模型返回的choices为空。')

                logprobs_obj = response.choices[0].logprobs
                if logprobs_obj is None:
                    raise ModelResponseError('模型返回的logprobs为空。')

                tokens = logprobs_obj.tokens
                token_logprobs = logprobs_obj.token_logprobs
                if not tokens or not token_logprobs:
                    raise ModelResponseError('模型返回的tokens/logprobs为空。')

                option_logprob = self.extract_option_logprob(tokens, token_logprobs, option_label)

                if option_logprob is not None:
                    token_length = len(tokens) - 1
                    return option_logprob, token_length

                raise ModelResponseError(f'无法提取选项 {option_label} 的logprob。')

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
            last_exception = ModelResponseError('logprob计算中出现未知错误。')

        raise last_exception


    def load_c3_data(self, json_path: str) -> List[Dict]:
        """从JSON文件加载C3数据"""
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        data = []
        for line in raw_data:
            content = ''.join([''.join(paragraph) for paragraph in line[0]])
            for question in line[1]:
                label = question['choice'].index(question['answer'])
                label = 'ABCD'[label]
                while len(question['choice']) < 4:
                    question['choice'].append('[NULL]')
                data.append({
                    'content': content,
                    'question': question['question'],
                    'choice0': question['choice'][0],
                    'choice1': question['choice'][1],
                    'choice2': question['choice'][2],
                    'choice3': question['choice'][3],
                    'label': label
                })
        return data
    
    def build_prompt(self, content: str, question: str, choice0: str, 
                     choice1: str, choice2: str, choice3: str, option_label: str = None) -> str:
        """构建0-shot prompt（PPL方法需要option_label）"""
        prompt = f"""{content}
问：{question}
A. {choice0}
B. {choice1}
C. {choice2}
D. {choice3}
请从"A"，"B"，"C"，"D"中进行选择。
答：{option_label if option_label else ''}"""
        return prompt
    
    def postprocess_answer(self, text: str) -> str:
        """后处理答案，提取A/B/C/D"""
        text = text.strip().upper()
        # 查找第一个A/B/C/D
        match = re.search(r'[ABCD]', text)
        if match:
            return match.group(0)
        # 如果没找到，尝试查找"答案是A"这样的模式
        match = re.search(r'答案[：:]\s*([ABCD])', text)
        if match:
            return match.group(1)
        # 如果还是没找到，返回空字符串
        return ""
    
    def generate_answer(self, prompt: str, max_retries: int = 3) -> str:
        """生成答案"""
        last_exception: Optional[Exception] = None
        
        for attempt in range(max_retries):
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=50,  # C3只需要一个字母
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
    
    def _get_logprob_for_label(self, content: str, question: str, choice0: str, 
                               choice1: str, choice2: str, choice3: str, label: str):
        """为单个选项获取logprob和长度"""
        from typing import Tuple
        prompt = self.build_prompt(content, question, choice0, choice1, choice2, choice3, label)
        logprob, length = self.get_option_logprob_with_length(prompt, label)
        return label, logprob, length

    def evaluate_single_question(self, question_data: Dict) -> Dict:
        """评估单个问题（使用PPL方法+长度归一化）"""
        import numpy as np
        try:
            content = question_data['content']
            question = question_data['question']
            choice0 = question_data['choice0']
            choice1 = question_data['choice1']
            choice2 = question_data['choice2']
            choice3 = question_data['choice3']
            reference_answer = question_data['label'].strip().upper()
            
            option_logprobs = {}
            option_normalized_logprobs = {}
            option_ppls = {}
            
            # 并发获取所有选项的logprob
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._get_logprob_for_label, content, question, 
                                  choice0, choice1, choice2, choice3, label): label
                    for label in ['A', 'B', 'C', 'D']
                }
                
                for future in as_completed(futures):
                    try:
                        label, logprob, length = future.result()
                        option_logprobs[label] = logprob
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
            
            predicted_answer = max(option_normalized_logprobs, key=option_normalized_logprobs.get)
            is_correct = (predicted_answer == reference_answer)
            
            with self._lock:
                if is_correct:
                    self._correct_count += 1
                self._total_count += 1
            
            return {
                'content': content,
                'question': question,
                'choice0': choice0,
                'choice1': choice1,
                'choice2': choice2,
                'choice3': choice3,
                'predicted_answer': predicted_answer,
                'reference_answer': reference_answer,
                'is_correct': is_correct,
                'logprobs': option_logprobs,
                'normalized_logprobs': option_normalized_logprobs,
                'ppls': option_ppls
            }
        except Exception as e:
            print(f"\n评估问题时出错: {e}")
            return {
                'question': question_data.get('question', ''),
                'predicted_answer': '',
                'reference_answer': question_data.get('label', ''),
                'is_correct': False,
                'error': str(e)
            }
    
    def evaluate_dataset(self, data: List[Dict], max_samples: int = None, seed: int = 42) -> Dict:
        """评估整个数据集"""
        import random

        if max_samples:
            random.seed(seed)
            data = random.sample(data, min(max_samples, len(data)))
        
        results = []
        
        self._correct_count = 0
        self._total_count = 0
        
        # 预热检测：先测试3个样本
        print("\n执行预热检测（测试3个样本）...")
        warmup_samples = data[:min(3, len(data))]
        warmup_failed = False
        for i, sample in enumerate(warmup_samples):
            try:
                result = self.evaluate_single_question(sample)
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
        self._correct_count = 0
        self._total_count = 0

        # 正式评估：评估所有样本（预热检测不跳过任何数据）
        eval_data = data
        total = len(eval_data)
        print(f"开始评估 {total} 个问题...")
        if max_samples:
            print(f"随机采样: {max_samples} 个样本（seed={seed}）")
        print(f"并发设置: {self.max_workers} 个问题并发")
        print(f"评估方法: PPL + 长度归一化")
        
        pbar = tqdm(total=total, desc="评估进度")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_data = {
                executor.submit(self.evaluate_single_question, q): q 
                for q in data
            }
            
            for future in as_completed(future_to_data):
                try:
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    
                    with self._lock:
                        current_total = self._total_count
                        current_correct = self._correct_count
                    
                    if current_total % 100 == 0 and current_total > 0:
                        current_acc = current_correct / current_total * 100
                        pbar.set_postfix({
                            'accuracy': f'{current_acc:.2f}%',
                            'correct': current_correct
                        })
                except Exception as e:
                    print(f"\n评估失败: {e}")
                    pbar.update(1)
        
        pbar.close()
        
        correct = self._correct_count
        accuracy = correct / total * 100 if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'shot_num': 0,
            'seed': seed if max_samples else None,
            'max_samples': max_samples,
            'results': results
        }


def find_c3_data_path(dataset_type: str = 'd', split: str = 'dev'):
    """查找C3数据集路径（只从datasets目录查找）
    
    Args:
        dataset_type: 'd' for dialogue, 'm' for mixed-genre
        split: 'train', 'dev', 'test'
    """
    # 优先查找官方格式：c3-{d,m}-{train,dev,test}.json
    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'c3')
    official_path = os.path.join(datasets_dir, f'c3-{dataset_type}-{split}.json')
    
    if os.path.exists(official_path):
        return official_path
    
    # 兼容旧格式：dev_0.json
    legacy_path = os.path.join(datasets_dir, 'dev_0.json')
    if os.path.exists(legacy_path):
        print(f"⚠️  使用旧格式数据集: {legacy_path}")
        print(f"   建议使用官方格式: c3-{dataset_type}-{split}.json")
        return legacy_path
    
    return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="C3 evaluation script")

    parser.add_argument(
        "--shot-num",
        type=int,
        default=0,
        help=f"Few-shot 示例数量（默认: 0）",
    )
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
        "--max-samples",
        type=int,
        default=None,
        help="Random sample size, None means evaluate all data (default: None)",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default='d',
        choices=['d', 'm'],
        help="Dataset type: d=dialogue, m=mixed-genre (default: d)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default='dev',
        choices=['train', 'dev', 'test'],
        help="Data split to evaluate (default: dev)",
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help="Dataset JSON file path (overrides --dataset-type and --split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    
    args = parser.parse_args()
    
    # 自动查找数据路径，如果找不到则使用默认路径
    JSON_PATH = args.dataset_file or find_c3_data_path(args.dataset_type, args.split)
    if JSON_PATH is None:
        # 尝试使用官方格式的默认路径
        default_path = f"datasets/c3/c3-{args.dataset_type}-{args.split}.json"
        print(f"⚠️  未找到C3数据集")
        print(f"   预期位置: {default_path}")
        print(f"   请使用--dataset-file参数指定路径或下载官方数据集")
        print(f"   下载命令: python download_datasets.py c3")
        return
    
    dataset_info = f"C3-{args.dataset_type.upper()} {args.split}"
    if args.dataset_file:
        dataset_info = os.path.basename(args.dataset_file)
    
    # 创建日志目录结构：logs/{model}/
    log_dir = os.path.join("logs", args.model)
    os.makedirs(log_dir, exist_ok=True)
    
    output_filename = f"c3_{args.dataset_type}_{args.split}_0shot.json"
    if args.max_samples:
        output_filename = f"c3_{args.dataset_type}_{args.split}_0shot_{args.max_samples}samples.json"
    OUTPUT_PATH = os.path.join(log_dir, output_filename)
    
    print("="*70)
    print(f"{args.shot_num}-shot 生成方法")
    print("="*70)
    print(f"模型: {args.model}")
    print(f"数据集: {dataset_info}")
    print(f"API: completions")
    print(f"{args.shot_num}-shot prompting")
    print("="*70)
    
    evaluator = C3Evaluator(
        args.base_url, 
        args.model, 
        max_workers=args.max_workers,
        max_tokens=args.max_tokens
    )
    
    print(f"\n加载数据: {JSON_PATH}")
    try:
        data = evaluator.load_c3_data(JSON_PATH)
        print(f"✓ 成功加载 {len(data)} 条数据")
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
        print(f"{args.shot_num}-shot")
        if args.max_samples:
            print(f"随机采样: {args.max_samples} 个样本（seed={args.seed}）")
        print(f"\n详细结果已保存到: {OUTPUT_PATH}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  评估被用户中断")
    except Exception as e:
        print(f"\n✗ 评估过程中出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

