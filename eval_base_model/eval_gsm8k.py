"""GSM8K数据集评估脚本 - 8-shot生成方法。"""
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

# 4-shot few-shot examples
FEW_SHOT_EXAMPLES = """
Q: Max can mow the lawn in 40 minutes. If it takes him twice that long to fertilize the
lawn, how long will it take him to both mow and fertilize the lawn?
A: Let’s think step by step. It takes Max 2 * 40 minutes = 80 minutes to fertilize the
lawn. In total, Max takes 80 minutes + 40 minutes = 120 minutes to both mow and
fertilize the lawn. The answer is 120.

Q: The bagels cost $2.25 each, or a dozen for $24. How much is saved, per bagel, in
cents, by buying a dozen at a time?
A: Let’s think step by step. They cost 2.25*100=225 cents each. At the bulk rate, they
are 24/12=2 dollar each. They cost 2*100=200 cents each. 225-200=25 cents are saved
per bagel. The answer is 25.

Q: Tim is 5 years old. His cousin, Rommel, is thrice as old as he is. His other cousin,
Jenny, is 2 years older than Rommel. How many years younger is Tim than Jenny?
A: Let’s think step by step. Rommel is 5 x 3 = 15 years old. Jenny is 15 + 2 = 17 years
old. So, Tim is 17 - 5 = 12 years younger than Jenny. The answer is 12.

Q: The school has 14 boys and 10 girls. If 4 boys and 3 girls drop out, how many boys
and girls are left?
A: Let’s think step by step. There are 14 boys - 4 boys = 10 boys left. There are 10 girls
- 3 girls = 7 girls left. In total there are 10 boys + 7 girls = 17 boys and girls left. The
answer is 17.

Q: Building one birdhouse requires 7 planks and 20 nails. If 1 nail costs 0.05, and one
plank costs 3, what is the cost, in dollars, to build 4 birdhouses?
A: Let’s think step by step. The cost of the planks for one birdhouse is 7 * 3 = 21. And
the nails are a cost of 20 * 0.05 = 1 for each birdhouse. So to build one birdhouse one
will need 21 + 1 = 22. So the cost of building 4 birdhouses is at 4 * 22 = 88. The answer
is 88.

Q: Danny brings 3 watermelons to his family picnic. He cuts each watermelon into 10
slices. His sister brings 1 watermelon to the family picnic, and she cuts the watermelon
into 15 slices. How many watermelon slices are there in total at the picnic?
A: Let’s think step by step. From Danny, there are 3 * 10 = 30 watermelon slices. From
his sister, there are 1 * 15 = 15 watermelon slices. There are a total of 30 + 15 = 45
watermelon slices. The answer is 45.

Q: Angela is a bike messenger in New York. She needs to deliver 8 times as many
packages as meals. If she needs to deliver 27 meals and packages combined, how
many meals does she deliver?
A: Let’s think step by step. Let p be the number of packages Angela delivers and
m be the number of meals. We know that p + m = 27 and p = 8m. Substituting the
second equation into the first equation, we get 8m + m = 27. Combining like terms,
we get 9m = 27. Dividing both sides by 9, we get m = 3. The answer is 3.

Q: Cori is 3 years old today. In 5 years, she will be one-third the age of her aunt. How
old is her aunt today?
A: Let’s think step by step. In 5 years, Cori will be 3 + 5 = 8 years old. In 5 years,
Cori’s aunt will be 8 x 3 = 24 years old. Today, her aunt is 24 - 5 = 19 years old. The
answer is 19.
"""


class ModelResponseError(RuntimeError):
    """Raised when the model response is invalid."""


class GSM8KEvaluator:
    """使用8-shot生成方法评估GSM8K数据集"""
    
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 max_workers: int = 32, max_tokens: int = 1024, shot_num: int = 8):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.shot_num = shot_num
        self._correct_count = 0
        self._total_count = 0
        self._lock = threading.Lock()
        
    def load_gsm8k_data(self, jsonl_path: str) -> List[Dict]:
        """从JSONL文件加载GSM8K数据"""
        result = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                result.append({
                    'question': item['question'],
                    'answer': item['answer'],  # 完整答案，包含####标记
                })
        return result
    
    def extract_answer_from_reference(self, answer_text: str) -> str:
        """从参考答案中提取答案（####后面的数字）"""
        # GSM8K的答案格式：#### 72
        if '####' in answer_text:
            parts = answer_text.split('####')
            if len(parts) > 1:
                answer = parts[-1].strip().replace(',', '')
                return answer
        return answer_text.strip().replace(',', '')
    
    def build_prompt(self, question: str) -> str:
        """构建8-shot prompt"""
        return f"""{FEW_SHOT_EXAMPLES}Q: {question}
A:Let’s think step by step. """
    
    def extract_answer(self, text: str) -> Optional[str]:
        """从生成的文本中提取答案（参考OpenCompass的gsm8k_postprocess）
        
        提取策略（按优先级）：
        1. 查找####标记（GSM8K标准格式）
        2. 查找"The answer is"模式（few-shot示例中的格式）
        3. 提取最后一个数字（回退方案）
        """
        # 方法1：查找####标记（GSM8K标准格式）

        # 方法2：查找"The answer is"模式（few-shot示例中的格式）
        # 匹配 "The answer is 120" 或 "The answer is 120." 等格式
        answer_patterns = [
            r'The answer is\s+([\d,]+\.?\d*)',  # "The answer is 120" 或 "The answer is 8,000"
            r'the answer is\s+([\d,]+\.?\d*)',  # "the answer is 120" (小写)
            r'answer is\s+([\d,]+\.?\d*)',      # "answer is 120"
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # 移除可能的逗号（如 "8,000" -> "8000"）
                answer = answer.replace(',', '')
                # 移除末尾的点号（如 "120." -> "120"）
                answer = answer.rstrip('.')
                return answer
        
        # 方法2：提取最后一个数字（回退方案，参考OpenCompass的gsm8k_postprocess）
        # 先移除"Question:"之后的内容（避免提取到新问题）
        text_before_question = text.split('Question:')[0]
        # 提取所有数字（包括小数）
        numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text_before_question)
        if numbers:
            # 返回最后一个数字，并移除逗号
            return numbers[-1].replace(',', '')
        
        return None
    
    def is_equal(self, pred: str, refer: str) -> bool:
        """判断两个答案是否相等（参考OpenCompass的Gsm8kEvaluator）"""
        try:
            # 直接比较
            if pred == refer:
                return True
            # 数值比较（允许浮点数误差）
            pred_float = float(pred)
            refer_float = float(refer)
            if abs(pred_float - refer_float) < 1e-6:
                return True
            # 如果参考答案是整数，预测答案的浮点数应该接近整数
            if abs(pred_float - int(refer_float)) < 1e-6:
                return True
        except (ValueError, TypeError):
            pass
        return False
    
    def generate_solution(self, question: str, max_retries: int = 3) -> str:
        """生成解决方案"""
        prompt = self.build_prompt(question)
        last_exception: Optional[Exception] = None
        
        for attempt in range(max_retries):
            try:
                # 添加停止条件，避免模型生成多个问题
                stop_sequences = ["\n\nQuestion:", "Question:"]
                
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    stop=stop_sequences,
                )
                
                if not response.choices:
                    raise ModelResponseError('模型返回的choices为空。')
                
                generated_text = response.choices[0].text
                generated_text = generated_text.strip()
                
                # 如果生成了多个问题，只保留第一个问题的答案
                first_problem_idx = generated_text.find('\n\nQuestion:')
                if first_problem_idx > 0:
                    generated_text = generated_text[:first_problem_idx].strip()
                
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
    
    def evaluate_single_problem(self, problem_data: Dict) -> Dict:
        """评估单个问题"""
        try:
            question = problem_data['question']
            reference_answer_full = problem_data.get('answer', '')
            reference_answer = self.extract_answer_from_reference(reference_answer_full)
            
            generated_solution = self.generate_solution(question)
            
            # 提取答案
            predicted_answer = self.extract_answer(generated_solution)
            
            # 判断是否相等
            is_correct = False
            if predicted_answer is not None:
                is_correct = self.is_equal(predicted_answer, reference_answer)
            
            with self._lock:
                if is_correct:
                    self._correct_count += 1
                self._total_count += 1
            
            return {
                'question': question,
                'predicted_answer': predicted_answer,
                'reference_answer': reference_answer,
                'generated_solution': generated_solution,
                'is_correct': is_correct,
            }
        except Exception as e:
            print(f"\n评估问题时出错: {e}")
            return {
                'question': problem_data.get('question', ''),
                'predicted_answer': None,
                'reference_answer': self.extract_answer_from_reference(problem_data.get('answer', '')),
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
                result = self.evaluate_single_problem(sample)
                pred_answer = result.get('predicted_answer', '')
                question_text = sample.get('question', '')[:50]
                print(f"✓ 样本{i+1}: 问题='{question_text}...' 预测='{pred_answer}'")
                if pred_answer is None or str(pred_answer).strip() in ["<unk>", "<unk>.", "", "None"]:
                    from openai import OpenAIError
                    raise OpenAIError(f"预热检测失败：模型返回异常响应 '{pred_answer}'")
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
            futures = {executor.submit(self.evaluate_single_problem, item): item for item in data}

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


def find_gsm8k_data_path():
    """使用OpenCompass的get_data_path获取GSM8K数据集路径"""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'opencompass'))
        from opencompass.utils.datasets import get_data_path

        # OpenCompass配置：path='opencompass/gsm8k' -> local='./data/gsm8k/'
        data_dir = get_data_path('opencompass/gsm8k', local_mode=False)
        test_path = os.path.join(data_dir, 'test.jsonl')
        
        if os.path.exists(test_path):
            return test_path
        else:
            return None
    except Exception as e:
        print(f"⚠️  无法使用OpenCompass获取数据路径: {e}")
        return None


def main():
    """主函数"""
    MODEL = "qwen2-5-72b"
    BASE_URL = "http://172.18.178.129:8000/v1"
    
    # 自动查找数据路径
    JSONL_PATH = find_gsm8k_data_path()
    if JSONL_PATH is None:
        JSONL_PATH = "/volume/ai-infra/zkjia/projects/opencompass/data/gsm8k/test.jsonl"  # 默认路径
        print(f"⚠️  未找到GSM8K数据集，将使用默认路径: {JSONL_PATH}")
        print("   如果文件不存在，请修改脚本中的JSONL_PATH变量")
    
    MAX_WORKERS = 32
    MAX_TOKENS = 1024
    
    # 创建日志目录结构：logs/{MODEL}/
    log_dir = os.path.join("logs", MODEL)
    os.makedirs(log_dir, exist_ok=True)
    OUTPUT_PATH = os.path.join(log_dir, "gsm8k_8shot_deepseek.json")
    
    SHOT_NUM = 8

    print("="*70)
    print(f"GSM8K 评估 - {MODEL} - {SHOT_NUM}-shot 生成方法")
    print("="*70)
    print(f"模型: {MODEL}")
    print(f"API: completions")
    print(f"方法: {SHOT_NUM}-shot few-shot prompting + 答案提取")
    print("="*70)

    evaluator = GSM8KEvaluator(
        BASE_URL,
        MODEL,
        max_workers=MAX_WORKERS,
        max_tokens=MAX_TOKENS,
        shot_num=SHOT_NUM
    )
    
    print(f"\n加载数据: {JSONL_PATH}")
    try:
        data = evaluator.load_gsm8k_data(JSONL_PATH)
        print(f"✓ 成功加载 {len(data)} 条数据")
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n开始评估...")
    
    max_samples = 500  # 测试 50 个样本（随机采样）
    seed = 42
    # max_samples = None  # 评估全部数据
    
    try:
        results = evaluator.evaluate_dataset(data, max_samples=max_samples, seed=seed)
        
        print(f"\n保存结果: {OUTPUT_PATH}")
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*70)
        print("评估结果")
        print("="*70)
        print(f"总问题数: {results['total']}")
        print(f"正确答案数: {results['correct']}")
        print(f"准确率: {results['accuracy']:.2f}%")
        print(f"Shot配置: {results['shot_num']}-shot")
        if max_samples:
            print(f"随机采样: {max_samples} 个样本（seed={seed}）")

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

