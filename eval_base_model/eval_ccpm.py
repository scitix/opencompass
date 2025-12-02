"""CCPM (中国古典诗歌匹配) 数据集评估脚本 - PPL方法 + 长度归一化"""
import json
import numpy as np
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
import eval_utils
class ModelResponseError(RuntimeError):
    """Raised when the model response is invalid."""


class CCPMEvaluator:
    """使用PPL + 长度归一化方法评估CCPM数据集"""
    
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", 
                 max_workers: int = 32, shot_num: int = 5):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.shot_num = shot_num
        self._correct_count = 0
        self._total_count = 0
        self._failed_extractions = 0
        self._lock = threading.Lock()
        self._few_shot_examples: Optional[List[Dict]] = None
    
    def load_ccpm_data(self, data_file: str, train_file: str = None) -> List[Dict]:
        """加载CCPM数据"""
        # 加载few-shot示例（从训练集）
        if train_file and os.path.exists(train_file) and self.shot_num > 0:
            self._few_shot_examples = []
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(self._few_shot_examples) >= self.shot_num:
                        break
                    item = json.loads(line.strip())
                    if 'translation' in item and 'choices' in item and 'answer' in item:
                        self._few_shot_examples.append(item)
        
        # 加载测试数据
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                translation = item.get('translation', '')
                choices = item.get('choices', [])
                answer = item.get('answer')
                
                if not translation or not choices or len(choices) != 4:
                    continue
                
                # 转换answer为字母标签（测试集可能没有answer）
                answer_label = None
                if answer is not None:
                    if isinstance(answer, int) and 0 <= answer < 4:
                        answer_label = chr(65 + answer)  # 0->A, 1->B, etc.
                
                data.append({
                    'translation': translation,
                    'choices': choices,
                    'answer': answer_label
                })
        
        return data
    
    def build_few_shot_prompt(self) -> str:
        """构建few-shot prompt"""
        if not self._few_shot_examples or self.shot_num == 0:
            return ""
        
        prompt_parts = []
        for example in self._few_shot_examples[:self.shot_num]:
            translation = example['translation']
            choices = example['choices']
            answer_idx = example['answer']
            answer_label = chr(65 + answer_idx)  # 0->A, 1->B, etc.
            
            prompt_parts.append(f"现代文描述：{translation}")
            for i, choice in enumerate(choices):
                label = chr(65 + i)
                prompt_parts.append(f"{label}. {choice}")
            prompt_parts.append(f"答案：{answer_label}\n")
        
        return "\n".join(prompt_parts)
    
    def build_prompt(self, translation: str, choices: List[str], option_label: str) -> str:
        """构建评估prompt（PPL方法）"""
        few_shot = self.build_few_shot_prompt()
        
        options_str = ""
        for i, choice in enumerate(choices):
            label = chr(65 + i)
            options_str += f"{label}. {choice}\n"
        
        if few_shot:
            prompt = f"""{few_shot}
现代文描述：{translation}
{options_str}答案：{option_label}"""
        else:
            prompt = f"""现代文描述：{translation}
{options_str}答案：{option_label}"""
        
        return prompt
    
    def extract_option_logprob(self, tokens: List[str], token_logprobs: List[Optional[float]], 
                               option_label: str) -> Optional[float]:
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
    
    def get_option_logprob_with_length(self, prompt: str, option_label: str, 
                                       max_retries: int = 3) -> Tuple[float, int]:
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
    
    def _get_logprob_for_label(self, translation: str, choices: List[str], label: str) -> Tuple[str, float, int]:
        """为单个选项获取logprob和长度"""
        prompt = self.build_prompt(translation, choices, label)
        logprob, length = self.get_option_logprob_with_length(prompt, label)
        return label, logprob, length
    
    def evaluate_single_question(self, question_data: Dict) -> Dict:
        """评估单个问题（使用PPL方法+长度归一化）"""
        try:
            translation = question_data['translation']
            choices = question_data['choices']
            reference_answer = question_data.get('answer')  # 可能为None（测试集）
            
            option_logprobs = {}
            option_normalized_logprobs = {}
            option_ppls = {}
            
            # 并发获取所有选项的logprob
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._get_logprob_for_label, translation, choices, label): label
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
                        
                        with self._lock:
                            self._failed_extractions += 1
            
            # 确保所有选项都有结果
            for label in ['A', 'B', 'C', 'D']:
                if label not in option_normalized_logprobs:
                    option_logprobs[label] = -10.0
                    option_normalized_logprobs[label] = -10.0
                    option_ppls[label] = np.exp(10.0)
            
            # 使用归一化后的logprob选择答案
            predicted_answer = max(option_normalized_logprobs, key=option_normalized_logprobs.get)
            
            # 只有当有reference_answer时才计算准确率
            is_correct = None
            if reference_answer is not None:
                is_correct = (predicted_answer == reference_answer)
                
                with self._lock:
                    if is_correct:
                        self._correct_count += 1
                    self._total_count += 1
            
            return {
                'translation': translation[:100] + '...' if len(translation) > 100 else translation,
                'choices': choices,
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
                'translation': question_data.get('translation', '')[:100],
                'predicted_answer': '',
                'reference_answer': question_data.get('answer', ''),
                'is_correct': False,
                'error': str(e),
            }
    
    def evaluate_dataset(self, data: List[Dict], max_samples: Optional[int] = None, 
                        seed: int = 42) -> Dict:
        """评估整个数据集"""
        if max_samples is not None and max_samples < len(data):
            random.seed(seed)
            data = random.sample(data, max_samples)
        
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
        self._failed_extractions = 0

        # 正式评估：评估所有样本（预热检测不跳过任何数据）
        eval_data = data
        total = len(eval_data)
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.evaluate_single_question, item) for item in data]
            
            for future in tqdm(as_completed(futures), total=len(eval_data), desc="评估进度"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"\n处理问题时出错: {e}")
                    continue
        
        accuracy = (self._correct_count / self._total_count * 100) if self._total_count > 0 else 0.0
        
        return {
            'accuracy': f"{accuracy:.2f}%",
            'correct': self._correct_count,
            'total': self._total_count,
            'failed_extractions': self._failed_extractions,
            'results': results
        }


if __name__ == "__main__":
    import argparse
    import os

    # 从环境变量读取默认配置
    default_model = os.environ.get("EVAL_MODEL_NAME", "qwen2-5-72b")
    default_base_url = os.environ.get("EVAL_BASE_URL", "http://localhost:8000/v1")
    default_api_key = os.environ.get("EVAL_API_KEY", "EMPTY")

    parser = argparse.ArgumentParser(description="CCPM数据集评估 - PPL方法 + 长度归一化")
    parser.add_argument("--base-url", type=str,
                       default=default_base_url,
                       help=f"OpenAI API base URL（可通过 EVAL_BASE_URL 环境变量设置）")
    parser.add_argument("--model", type=str,
                       default=default_model,
                       help=f"模型名称（默认: {default_model}，可通过 EVAL_MODEL_NAME 环境变量设置）")
    parser.add_argument("--api-key", type=str, default=default_api_key, help="API密钥（可通过 EVAL_API_KEY 环境变量设置）")
    parser.add_argument("--max-workers", type=int, default=32, help="最大并发数")
    parser.add_argument("--shot-num", type=int, default=0, help="Few-shot示例数量（CCPM默认0-shot）")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Random sample size, None means evaluate all data (default: None)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--data-file", type=str,
                       default="datasets/ccpm/valid.jsonl",
                       help="测试数据文件（默认使用valid.jsonl，因为test_public.jsonl无标签）")
    parser.add_argument("--train-file", type=str,
                       default="datasets/ccpm/train.jsonl",
                       help="训练数据文件（用于few-shot）")
    
    args = parser.parse_args()

    # 创建输出目录（使用 logs/{model}/ 结构）
    log_dir = os.path.join(os.path.dirname(__file__), "logs", args.model)
    os.makedirs(log_dir, exist_ok=True)
    output_filename = f"ccpm_{args.shot_num}shot.json"
    if args.max_samples:
        output_filename = f"ccpm_{args.shot_num}shot_{args.max_samples}samples.json"
    output_file = os.path.join(log_dir, output_filename)
    
    print("="*70)
    print(f"CCPM 评估 - {args.model.split('/')[-1]} - {args.shot_num}-shot PPL方法")
    print("="*70)
    print(f"模型: {args.model.split('/')[-1]}")
    print(f"API: completions")
    print(f"方法: {args.shot_num}-shot PPL + 长度归一化")
    print(f"Few-shot来源: {args.train_file}")
    print("="*70)
    print()
    
    # 初始化评估器
    evaluator = CCPMEvaluator(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_workers=args.max_workers,
        shot_num=args.shot_num
    )
    
    # 加载数据
    print(f"加载数据: {args.data_file}")
    if args.shot_num > 0:
        print(f"加载Few-shot示例: {args.train_file}")
    
    try:
        data = evaluator.load_ccpm_data(args.data_file, args.train_file)
        print(f"✓ 成功加载 {len(data)} 条测试数据")
        if evaluator._few_shot_examples:
            print(f"✓ 成功加载 {len(evaluator._few_shot_examples)} 个few-shot示例")
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 评估
    print("\n开始评估...")
    if args.max_samples:
        print(f"随机采样: {args.max_samples} 个样本（seed={args.seed}）")
    
    try:
        results = evaluator.evaluate_dataset(data, max_samples=args.max_samples, seed=args.seed)
        
        print(f"\n保存结果: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*70)
        print("评估结果")
        print("="*70)
        print(f"总问题数: {results['total']}")
        print(f"正确答案数: {results['correct']}")
        print(f"准确率: {results['accuracy']}")
        print(f"Shot配置: {args.shot_num}-shot")
        if args.max_samples:
            print(f"随机采样: {args.max_samples} 个样本（seed={args.seed}）")
        
        if results['failed_extractions'] > 0:
            failure_rate = results['failed_extractions'] / (results['total'] * 4) * 100
            print(f"\nLogprob 提取失败次数: {results['failed_extractions']}")
            print(f"失败率: {failure_rate:.2f}%")
            if failure_rate > 10:
                print("⚠️  失败率较高，可能影响准确率")
        
        print(f"\n详细结果已保存到: {output_file}")
        print(f"日志目录: {args.output_dir}")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ 评估失败: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

