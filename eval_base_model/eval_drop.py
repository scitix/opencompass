"""DROP (Discrete Reasoning Over Paragraphs) 数据集评估脚本 - 完全对齐OpenCompass Simple-Evals标准"""
import json
import os
import re
import string
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
import numpy as np
from scipy.optimize import linear_sum_assignment
import eval_utils
class ModelResponseError(RuntimeError):
    """Raised when the model response is invalid."""


# ============================================================================
# OpenCompass标准的Few-shot示例（完整passage，高质量）
# 来源: OpenAI Simple-Evals / OpenCompass
# ============================================================================
FIXED_FEW_SHOT_EXAMPLES = [
    {
        "passage": """Trunajaya rebellion or Trunajaya War was the ultimately unsuccessful rebellion waged by the Madurese prince Trunajaya and fighters from Makassar against the Mataram Sultanate and its Dutch East India Company supporters in Java during the 1670s. The rebellion was initially successful: the rebels defeated the royal army at Gegodog, captured most of the Javanese north coast, and took the Mataram capital Plered. King Amangkurat I died during the retreat of the royal court. His son and successor, Amangkurat II, requested help from the VOC in exchange for financial remuneration and geopolitical concessions. The VOC's subsequent involvement turned the tide of the war. VOC and Mataram forces recovered lost territories and overran Trunajaya's new capital at Kediri. However, the rebellion continued until the capture of Trunajaya at the end of 1679, and the defeat, death, or surrender of the other rebel leaders. Trunajaya was killed by Amangkurat II personally in 1680 while a prisoner of the VOC. After his father's death in 1677, Amangkurat II also faced rival claims to the throne. The most serious rival was his brother Pangeran Puger, who took the capital Plered in 1677 and did not surrender until 1681.""",
        "question": "How many years was it between Trunajaya's capture and his death while prisoner of the VOC?",
        "answer": "1"
    },
    {
        "passage": """Led by former Giant Kurt Warner, the defending NFC champions took the field at Giants Stadium against a Giants team still reeling from their bad loss in New Orleans. The Giants scored first, sending Jacobs in for a 4-yard touchdown run following a Terrell Thomas interception. Later, Arizona running back Beanie Wells scored his first career touchdown on a 13-yard rush. Manning responded by throwing a 62-yard touchdown to Nicks for his longest reception of the year. In the second half, the Cardinals' Tim Hightower and Jason Wright scored touchdowns. But it was turnovers that decided this game; Manning's 3 interceptions were as many as he had thrown all season. The Giants scored only 3 points in the second half, ending the game on an interception to Antrel Rolle. The Giants notable streak of 38 consecutive starts by the same offensive line unit was ended here, as offensive tackle Kareem McKenzie missed the game with a groin injury. McKenzie returned the following week.""",
        "question": "Which player made the first score of the game?",
        "answer": "Jacobs"
    },
    {
        "passage": """The median age in the city was 22.1 years. 10.1% of residents were under the age of 18; 56.2% were between the ages of 18 and 24; 16.1% were from 25 to 44; 10.5% were from 45 to 64; and 7% were 65 years of age or older. The gender makeup of the city was 64.3% male and 35.7% female.""",
        "question": "How many percent were not from 25 to 44?",
        "answer": "83.9"
    }
]


# ============================================================================
# OpenCompass标准F1计算（完全对齐drop_simple_evals.py）
# ============================================================================
def _is_number(text: str) -> bool:
    """检查文本是否为数字"""
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    """标准化数字"""
    if _is_number(text):
        return str(float(text))
    else:
        return text


def _remove_articles(text: str) -> str:
    """移除冠词"""
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def _white_space_fix(text: str) -> str:
    """修复空白字符"""
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    """移除标点符号（但保留数字中的标点）"""
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    """转小写"""
    return text.lower()


def _tokenize(text: str) -> list:
    """分词（使用空格和连字符）"""
    return re.split(" |-", text)


def _normalize_answer(text: str) -> str:
    """标准化答案文本（OpenCompass标准）"""
    parts = [
        _white_space_fix(
            _remove_articles(_normalize_number(_remove_punc(_lower(token))))
        )
        for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _answer_to_bags(answer) -> Tuple[List[str], List[set]]:
    """将答案转换为token bags"""
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    
    normalized_spans = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _compute_f1(predicted_bag: set, gold_bag: set) -> float:
    """计算F1分数（token级别）"""
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    ) * 100
    return f1


def _match_numbers_if_present(gold_bag: set, predicted_bag: set) -> bool:
    """检查数字是否匹配（如果存在数字）"""
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def _align_bags(predicted: List[set], gold: List[set]) -> List[float]:
    """对齐预测和参考答案的token bags（使用线性分配）"""
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)
    
    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def get_drop_metrics(predicted: str, gold) -> Tuple[float, float]:
    """
    计算DROP的EM和F1指标（OpenCompass标准）
    
    Args:
        predicted: 预测答案（字符串或列表）
        gold: 参考答案（字符串或列表）
    
    Returns:
        (em_score, f1_score): EM分数（0.0或1.0）和F1分数（0-100）
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    
    # 计算EM
    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0
    
    # 计算F1
    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    
    return exact_match, f1


def drop_metric(prediction: str, references: List[str]) -> Tuple[float, float]:
    """
    计算DROP指标（支持多个参考答案）
    
    Args:
        prediction: 预测答案
        references: 参考答案列表
    
    Returns:
        (max_em, max_f1): 最大EM和最大F1分数
    """
    em_scores = []
    f1_scores = []
    
    for answer in references:
        if answer.strip() != "":
            em, f1 = get_drop_metrics(prediction, answer)
            em_scores.append(em)
            f1_scores.append(f1)
    
    if not em_scores:
        return (1.0, 100.0) if not prediction.strip() else (0.0, 0.0)
    
    return (max(em_scores), max(f1_scores))


# ============================================================================
# DROPEvaluator - 完全对齐OpenCompass
# ============================================================================
class DROPEvaluator:
    """使用OpenCompass Simple-Evals标准评估DROP数据集"""
    
    # Prompt模板（适配Base模型的Completions API）
    # 注意：Base模型不适合"Think step by step"的复杂指令，使用简单直接的格式
    FEW_SHOT_TEMPLATE = """---
Passage: {passage}
Question: {question}
Answer: {answer}"""
    
    QUERY_TEMPLATE = """Answer the following questions based on the passage. Give a short and direct answer.

{examples}

---
Passage: {passage}
Question: {question}
Answer:"""
    
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", 
                 max_workers: int = 32, shot_num: int = 3):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.shot_num = shot_num
        self._total_f1 = 0.0
        self._total_em = 0.0
        self._total_count = 0
        self._lock = threading.Lock()
    
    def postprocess_answer(self, response_text: str) -> str:
        """后处理答案：提取第一行作为答案（Base模型直接生成）"""
        # Base模型通常直接给出答案，取第一行即可
        lines = response_text.strip().split('\n')
        return lines[0].strip() if lines else response_text.strip()
    
    def build_few_shot_examples(self) -> str:
        """构建few-shot示例部分（使用固定的高质量示例）"""
        examples = []
        for ex in FIXED_FEW_SHOT_EXAMPLES[:self.shot_num]:
            example_text = self.FEW_SHOT_TEMPLATE.format(
                passage=ex['passage'],
                question=ex['question'],
                answer=ex['answer']
            )
            examples.append(example_text)
        
        return "\n\n".join(examples)
    
    def build_prompt(self, passage: str, question: str) -> str:
        """构建评估prompt（适配Base模型）"""
        few_shot_examples = self.build_few_shot_examples()
        
        prompt = self.QUERY_TEMPLATE.format(
            examples=few_shot_examples,
            passage=passage,
            question=question
        )
        
        return prompt
    
    def load_drop_data(self, dataset_path: str, split: str = "dev") -> List[Dict]:
        """
        加载DROP Simple-Evals数据（OpenCompass标准格式）
        
        Args:
            dataset_path: 数据集目录路径
            split: 数据split，"train"或"dev"（Simple-Evals使用dev而不是validation）
        
        Returns:
            数据列表
        """
        # OpenCompass Simple-Evals格式：drop_v0_dev.jsonl
        if split == "validation":
            split = "dev"  # 转换为Simple-Evals命名
        
        jsonl_file = os.path.join(dataset_path, f"drop_v0_{split}.jsonl")
        
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(
                f"数据文件不存在: {jsonl_file}\n"
                f"请确保使用OpenCompass Simple-Evals格式的数据集"
            )
        
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                example = json.loads(line)
                # Simple-Evals格式: context包含"Passage: XXX\nQuestion: YYY\nAnswer:"
                context = example['context']
                ref_text = example['ref_text']  # 参考答案
                
                # 解析context提取passage和question
                # 格式: "Passage: XXX\nQuestion: YYY\nAnswer:"
                parts = context.split('\nQuestion: ')
                if len(parts) != 2:
                    continue
                
                passage = parts[0].replace('Passage: ', '').strip()
                question_part = parts[1].split('\nAnswer:')[0].strip()
                
                # ref_text可能包含多个答案（用|分隔）
                answers = [ans.strip() for ans in ref_text.split('|') if ans.strip()]
                
                if not passage or not question_part or not answers:
                    continue
                
                data.append({
                    'passage': passage,
                    'question': question_part,
                    'answers': answers,
                })
        
        return data
    
    def generate_answer(self, prompt: str, max_retries: int = 3) -> str:
        """生成答案（适配Base模型）"""
        for attempt in range(max_retries):
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=50,  # Base模型直接生成答案，50足够
                    temperature=0,
                    stop=["\n", "---"]  # 只在换行和分隔符处停止
                )
                
                if not response.choices:
                    raise ModelResponseError('模型返回的choices为空。')
                
                answer = response.choices[0].text.strip()
                
                # 检测异常响应
                if not answer or answer == "":
                    raise ModelResponseError("模型返回空响应。")
                if answer in ["<unk>", "<unk>.", "<UNK>", "<UNK>."]:
                    raise ModelResponseError(f"模型返回无效token: {answer}")
                
                return answer
                
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                else:
                    raise ModelResponseError(f'生成答案失败: {e}')
        
        return ""
    
    def evaluate_single_question(self, question_data: Dict) -> Dict:
        """评估单个问题"""
        try:
            passage = question_data['passage']
            question = question_data['question']
            reference_answers = question_data['answers']
            
            # 构建prompt并生成答案
            prompt = self.build_prompt(passage, question)
            raw_response = self.generate_answer(prompt)
            
            # 后处理：提取"Answer: XXX"
            predicted_answer = self.postprocess_answer(raw_response)
            
            # 计算F1和EM（使用OpenCompass标准）
            em_score, f1_score = drop_metric(predicted_answer, reference_answers)
            
            with self._lock:
                self._total_f1 += f1_score
                self._total_em += em_score
                self._total_count += 1
            
            return {
                'passage': passage[:100] + '...' if len(passage) > 100 else passage,
                'question': question,
                'raw_response': raw_response[:200] + '...' if len(raw_response) > 200 else raw_response,
                'predicted_answer': predicted_answer,
                'reference_answers': reference_answers,
                'f1_score': f1_score,
                'em_score': em_score,
                'section_id': question_data.get('section_id', ''),
                'query_id': question_data.get('query_id', ''),
            }
        except Exception as e:
            print(f"\n评估问题时出错: {e}")
            return {
                'question': question_data.get('question', ''),
                'raw_response': '',
                'predicted_answer': '',
                'reference_answers': question_data.get('answers', []),
                'f1_score': 0.0,
                'em_score': 0.0,
                'error': str(e),
            }
    
    def evaluate_dataset(self, data: List[Dict], max_samples: Optional[int] = None, 
                        seed: int = 42) -> Dict:
        """评估整个数据集"""
        if max_samples is not None and max_samples < len(data):
            import random
            random.seed(seed)
            data = random.sample(data, max_samples)
        
        # 预热检测：先测试3个样本（不计入最终统计）
        num_warmup = min(3, len(data)) if not max_samples else min(3, max_samples, len(data))
        print(f"\n执行预热检测（测试{num_warmup}个样本）...")
        warmup_samples = data[:num_warmup]
        for i, sample in enumerate(warmup_samples):
            try:
                result = self.evaluate_single_question(sample)
                predicted_answer = result.get('predicted_answer', '')[:30]
                print(f"✓ 样本{i+1}: 问题='{sample['question'][:50]}...' 预测='{predicted_answer}...'")
                if not predicted_answer or predicted_answer.strip() in ["<unk>", "<unk>."]:
                    raise ModelResponseError(f"预热检测失败：模型返回异常响应 '{predicted_answer}'")
            except ModelResponseError as e:
                print(f"\n❌ 预热检测失败！")
                print(f"错误: {e}")
                print(f"请检查：")
                print(f"  1. API endpoint是否正确")
                print(f"  2. 模型是否正常运行")
                print(f"  3. max_tokens设置是否合理")
                raise
        print("✓ 预热检测通过，开始正式评估...\n")
        
        # 重置计数器（预热检测已完成，不计入正式统计）
        with self._lock:
            self._total_f1 = 0.0
            self._total_em = 0.0
            self._total_count = 0

        # 正式评估：评估所有样本（包括预热的3个）
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
        
        avg_f1 = (self._total_f1 / self._total_count) if self._total_count > 0 else 0.0
        avg_em = (self._total_em / self._total_count * 100) if self._total_count > 0 else 0.0
        
        return {
            'f1_score': f'{avg_f1:.2f}',  # F1已经是0-100范围
            'em_score': f'{avg_em:.2f}%',
            'total': self._total_count,
            'shot_num': self.shot_num,
            'results': results
        }


if __name__ == "__main__":
    import argparse
    import os

    # 从环境变量读取默认配置
    default_model = os.environ.get("EVAL_MODEL_NAME", "qwen2-5-72b")
    default_base_url = os.environ.get("EVAL_BASE_URL", "http://localhost:8000/v1")
    default_api_key = os.environ.get("EVAL_API_KEY", "EMPTY")

    parser = argparse.ArgumentParser(description="DROP数据集评估 - OpenCompass Simple-Evals标准")
    parser.add_argument("--base-url", type=str,
                       default=default_base_url,
                       help=f"OpenAI API base URL（可通过 EVAL_BASE_URL 环境变量设置）")
    parser.add_argument("--model", type=str,
                       default=default_model,
                       help=f"模型名称（默认: {default_model}，可通过 EVAL_MODEL_NAME 环境变量设置）")
    parser.add_argument("--api-key", type=str, default=default_api_key, help="API密钥（可通过 EVAL_API_KEY 环境变量设置）")
    parser.add_argument("--max-workers", type=int, default=128, help="最大并发数")
    parser.add_argument("--shot-num", type=int, default=3, help="Few-shot示例数量")
    parser.add_argument("--max-samples", type=int, default=None, help="最大评估样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--dataset-path", type=str,
                       default="datasets/drop_simple-evals",
                       help="DROP Simple-Evals数据集路径")
    parser.add_argument("--split", type=str,
                       default="dev",
                       choices=["train", "dev"],
                       help="评估的数据split（Simple-Evals使用'dev'而非'validation'）")

    args = parser.parse_args()

    # 创建输出目录（使用 logs/{model}/ 结构）
    log_dir = os.path.join(os.path.dirname(__file__), "logs", args.model)
    os.makedirs(log_dir, exist_ok=True)
    output_file = os.path.join(log_dir, f"drop_{args.shot_num}shot.json")
    
    print("="*70)
    print(f"DROP 评估 - {args.model.split('/')[-1]} - {args.shot_num}-shot")
    print("="*70)
    print(f"模型: {args.model.split('/')[-1]}")
    print(f"API: completions (Base模型)")
    print(f"方法: {args.shot_num}-shot Generation + OpenCompass标准F1评估")
    print(f"数据split: {args.split}")
    print("="*70)
    print()
    
    # 初始化评估器
    evaluator = DROPEvaluator(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        max_workers=args.max_workers,
        shot_num=args.shot_num
    )
    
    # 显示few-shot示例
    print("使用的Few-shot示例:")
    print("-" * 70)
    for i, ex in enumerate(FIXED_FEW_SHOT_EXAMPLES[:args.shot_num], 1):
        print(f"示例{i}:")
        print(f"  问题: {ex['question']}")
        print(f"  答案: {ex['answer']}")
    print("-" * 70)
    print()
    
    # 加载数据
    print(f"加载数据: {args.dataset_path} ({args.split})")
    try:
        data = evaluator.load_drop_data(args.dataset_path, args.split)
        print(f"✓ 成功加载 {len(data)} 条数据")
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
        print(f"F1 分数: {results['f1_score']}")
        print(f"EM 分数: {results['em_score']}")
        print(f"Shot配置: {args.shot_num}-shot")
        if args.max_samples:
            print(f"随机采样: {args.max_samples} 个样本（seed={args.seed}）")
        
        print(f"\n详细结果已保存到: {output_file}")
        print(f"日志目录: {args.output_dir}")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ 评估失败: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
