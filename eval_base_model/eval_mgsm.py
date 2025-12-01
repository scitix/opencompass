"""MGSM数据集评估脚本 - 8-shot生成方法（使用训练集作为few-shot示例）。"""
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from openai import OpenAI
from tqdm import tqdm
import eval_utils


class ModelResponseError(RuntimeError):
    """Raised when the model response is invalid."""


class MGSMEvaluator:
    """使用8-shot生成方法评估MGSM数据集（从训练集构建few-shot示例）"""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 max_workers: int = 32, max_tokens: int = 1024, lang: str = 'en'):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.lang = lang
        self._correct_count = 0
        self._total_count = 0
        self._lock = threading.Lock()
        
        # 加载训练集作为few-shot示例
        self.few_shot_examples = self._load_few_shot_examples()

    def _load_few_shot_examples(self) -> str:
        """从TSV文件加载训练集作为8-shot示例"""
        try:
            # 从TSV文件加载训练集
            train_tsv_path = os.path.join(
                os.path.dirname(__file__), 
                'datasets', 
                'mgsm', 
                f'mgsm_{self.lang}_train.tsv'
            )
            
            if not os.path.exists(train_tsv_path):
                # 如果当前语言的训练集不存在，尝试使用英文版本
                if self.lang != 'en':
                    print(f"⚠️  语言 {self.lang} 的训练集不存在，使用英文版本")
                    train_tsv_path = os.path.join(
                        os.path.dirname(__file__), 
                        'datasets', 
                        'mgsm', 
                        'mgsm_en_train.tsv'
                    )
                else:
                    raise FileNotFoundError(f"未找到训练集文件: {train_tsv_path}")
            
            # 读取TSV文件
            train_examples = []
            with open(train_tsv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t', 1)
                    if len(parts) >= 2:
                        question = parts[0].strip()
                        answer = parts[1].strip()
                        train_examples.append({
                            'question': question,
                            'answer': answer,
                        })
            
            if not train_examples:
                raise ValueError(f"训练集文件为空: {train_tsv_path}")
            
            # 构建few-shot示例字符串
            # 注意：TSV文件只包含数字答案，没有推理步骤
            # 我们需要构建一个简单的格式，让模型知道如何回答
            few_shot_parts = []
            for example in train_examples:
                question = example['question']
                answer = example['answer']
                
                # 格式化示例（使用简单的格式，因为TSV只有数字答案）
                # 参考GSM8K的格式，但简化处理
                few_shot_parts.append(f"{question}\n\nAnswer: {answer}")
            
            return "\n\n".join(few_shot_parts)
            
        except Exception as e:
            print(f"⚠️  从TSV加载训练集失败: {e}")
            print("   尝试从exemplars.py加载（包含完整答案文本）...")
            import traceback
            traceback.print_exc()
            
            # 回退到exemplars.py
            try:
                exemplars_path = os.path.join(os.path.dirname(__file__), 'datasets', 'mgsm', 'exemplars.py')
                if os.path.exists(exemplars_path):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("exemplars", exemplars_path)
                    exemplars_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(exemplars_module)
                    
                    MGSM_EXEMPLARS = exemplars_module.MGSM_EXEMPLARS
                    
                    lang = self.lang if self.lang in MGSM_EXEMPLARS else 'en'
                    exemplars = MGSM_EXEMPLARS[lang]
                    
                    few_shot_parts = []
                    for key in sorted(exemplars.keys()):
                        exemplar = exemplars[key]
                        question = exemplar.get('q', '').strip()
                        answer = exemplar.get('a', '').strip()
                        few_shot_parts.append(f"{question}\n\n{answer}")
                    
                    print("✓ 已从exemplars.py加载训练集（包含完整答案文本）")
                    return "\n\n".join(few_shot_parts)
            except Exception as e2:
                print(f"⚠️  从exemplars.py加载也失败: {e2}")
                return ""

    def load_mgsm_data(self, tsv_path: str) -> List[Dict]:
        """从TSV文件加载MGSM测试集数据"""
        result = []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t', 1)
                if len(parts) >= 2:
                    question, answer = parts[0], parts[1]
                    result.append({
                        'question': question,
                        'answer': answer,
                    })
        return result

    def build_prompt(self, question: str) -> str:
        """构建8-shot prompt（使用训练集示例）"""
        # 语言特定的指令（来自OpenCompass的mgsm_gen_d967bc.py）
        LANG_TO_INSTRUCTIONS = {
            'en': """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".

{question}""",
            'bn': """এই গণিতের সমস্যাটি সমাধান করুন। চূড়ান্ত উত্তর দেওয়ার আগে যুক্তিসম্পন্ন পদক্ষেপ প্রদান করুন। চূড়ান্ত উত্তরটি একক সংখ্যা হিসাবে "উত্তর:" এর পরে শেষ লাইনে দিন। "উত্তর:" এর পরে অন্য কিছু যুক্ত করবেন না।

{question}""",
            'de': """Löse dieses Mathematikproblem. Gib die Schritte zur Begründung an, bevor du die endgültige Antwort in der letzten Zeile alleine im Format "Antwort:" gibst. Füge nichts anderes als die ganzzahlige Antwort nach "Antwort:" hinzu.

{question}""",
            'es': """Resuelve este problema matemático. Proporciona los pasos de razonamiento antes de dar la respuesta final en la última línea por sí misma en el formato de "Respuesta:". No añadas nada más que la respuesta entera después de "Respuesta:".

{question}""",
            'fr': """Résolvez ce problème de mathématiques. Donnez les étapes de raisonnement avant de fournir la réponse finale sur la dernière ligne elle-même dans le format de "Réponse:". N'ajoutez rien d'autre que la réponse entière après "Réponse:".

{question}""",
            'ja': """この数学の問題を解いてください。最終的な答えを「答え:」という形式で最後の行に単独で記述する前に、推論の手順を記述してください。「答え:」の後に整数以外の何も追加しないでください。

{question}""",
            'ru': """Решите эту математическую задачу. Предоставьте шаги рассуждения, прежде чем давать окончательный ответ в последней строке в формате "Ответ:". Не добавляйте ничего, кроме целочисленного ответа, после "Ответ:".

{question}""",
            'sw': """Tatua tatizo hili la hesabu. Toa hatua za hoja kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote isipokuwa jibu kamili baada ya "Jibu:".

{question}""",
            'te': """ఈ గణిత సమస్యను పరిష్కరించండి। చివరి సమాధానాన్ని ఇవ్వదానికి ముందు తర్కాత్మక అదుగులను ఇవ్వండి। చివరి పంక్తిలో మాత్రమే 'సమాధానం:' అనే ఆకారంలో చివరి సమాధానాద్ని ఇవ్వండి సమాధానం: తర్వాత పూర్ణాంక సమాధానానికి తప్పించి ఎదేనా చేర్చవద్దు.

{question}""",
            'th': """แก้ปัญหาคณิตศาสตร์นี้ ให้ให้ขั้นตอนการใช้เหตุผลก่อนที่จะให้คำตอบสุดท้ายในบรรทัดสุดท้ายโดยอยู่ในรูปแบบ "คำตอบ:" ไม่ควรเพิ่มอะไรนอกจากคำตอบที่เป็นจำนวนเต็มหลังจาก "คำตอบ:"

{question}""",
            'zh': """解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除整数答案之外的任何内容。

{question}""",
        }
        instruction = LANG_TO_INSTRUCTIONS.get(self.lang, LANG_TO_INSTRUCTIONS['en']).format(question=question)
        
        # 组合few-shot示例和当前问题
        if self.few_shot_examples:
            return f"{self.few_shot_examples}\n\n{instruction}"
        else:
            return instruction

    def extract_answer(self, text: str) -> Optional[str]:
        """从生成的文本中提取答案（参考OpenCompass的mgsm_postprocess）
        
        策略：提取第一个"Answer:"后的第一个数字
        """
        LANG_TO_ANSWER_PREFIX = {
            'en': 'Answer',
            'bn': 'উত্তর',
            'de': 'Antwort',
            'es': 'Respuesta',
            'fr': 'Réponse',
            'ja': '答え',
            'ru': 'Ответ',
            'sw': 'Jibu',
            'te': 'సమాధానం',
            'th': 'คำตอบ',
            'zh': '答案',
        }
        answer_prefix = LANG_TO_ANSWER_PREFIX.get(self.lang, 'Answer')
        
        if answer_prefix in text:
            # 取第一个"Answer:"后的内容（不是最后一个）
            answer_text = text.split(answer_prefix, 1)[1].strip()
            # 提取第一个数字（不是最后一个）
            numbers = re.findall(r'\-?\d+\.?\d*', answer_text.replace(',', ''))
            return numbers[0].rstrip('.') if numbers else ''
        
        # 如果没有找到前缀，尝试直接提取第一个数字
        numbers = re.findall(r'\-?\d+\.?\d*', text.replace(',', ''))
        return numbers[0].rstrip('.') if numbers else ''

    def is_equal(self, pred: str, refer: str) -> bool:
        """判断两个答案是否相等（参考OpenCompass的MGSM_Evaluator）"""
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
                # 包括各种可能的新问题开始模式
                stop_sequences = [
                    "\n\nQuestion:",
                    "Question:",
                    "\n\nSolve this",
                    "\nSolve this",
                    "\n\n\n",  # 三个换行符
                ]

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
                # 检查各种可能的新问题开始标记
                stop_markers = ['\n\nQuestion:', '\n\nSolve this', '\n\n\n']
                for marker in stop_markers:
                    idx = generated_text.find(marker)
                    if idx > 0:
                        generated_text = generated_text[:idx].strip()
                        break

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
            reference_answer = problem_data.get('answer', '')

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
                'reference_answer': problem_data.get('answer', ''),
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
            'shot_num': 8,
            'seed': seed if max_samples else None,
            'max_samples': max_samples,
            'results': results
        }


def find_mgsm_data_path(lang: str = 'en'):
    """查找MGSM数据集路径（只从datasets目录查找）"""
    # 只从datasets目录查找测试集
    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'mgsm')
    tsv_path = os.path.join(datasets_dir, f'mgsm_{lang}.tsv')
    
    if os.path.exists(tsv_path):
        return tsv_path
    
    return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MGSM evaluation script")

    parser.add_argument(
        "--shot-num",
        type=int,
        default=8,
        help=f"Few-shot 示例数量（默认: 8）",
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
        default=1024,
        help="Maximum tokens to generate (default: 1024)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Random sample size, None means evaluate all data (default: None)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    
    args = parser.parse_args()

    # 自动查找数据路径（只从datasets目录查找）
    TSV_PATH = find_mgsm_data_path(args.lang)
    if TSV_PATH is None:
        # 如果找不到，提示用户
        print(f"⚠️  未找到MGSM数据集文件 (mgsm_{args.lang}.tsv)")
        print("   请确保数据文件在以下位置：")
        print(f"   datasets/mgsm/mgsm_{args.lang}.tsv")
        return

    # 创建日志目录结构：logs/{model}/
    log_dir = os.path.join("logs", args.model)
    os.makedirs(log_dir, exist_ok=True)
    
    shot_desc = f"{args.shot_num}-shot" if args.shot_num > 0 else "0-shot"
    output_filename = f"mgsm_{args.lang}_{args.shot_num}shot.json"
    if args.max_samples:
        output_filename = f"mgsm_{args.lang}_{args.shot_num}shot_{args.max_samples}samples.json"
    OUTPUT_PATH = os.path.join(log_dir, output_filename)

    print("="*70)
    print(f"MGSM 评估 - {args.model} - {shot_desc} 生成方法 - {args.lang}")
    print("="*70)
    print(f"模型: {args.model}")
    print(f"API: completions")
    print(f"方法: {shot_desc} few-shot prompting（使用训练集示例）+ 答案提取")
    print(f"语言: {args.lang}")
    print(f"训练集示例来源: datasets/mgsm/mgsm_{args.lang}_train.tsv")
    print("="*70)

    evaluator = MGSMEvaluator(
        args.base_url,
        args.model,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        lang=args.lang
    )

    # 验证few-shot示例已加载
    if evaluator.few_shot_examples:
        example_count = evaluator.few_shot_examples.count('\n\n') + 1
        print(f"\n✓ 成功加载 {example_count} 个训练集示例作为few-shot")
    else:
        print(f"{args.shot_num}-shot")

    print(f"\n加载测试集数据: {TSV_PATH}")
    try:
        data = evaluator.load_mgsm_data(TSV_PATH)
        print(f"✓ 成功加载 {len(data)} 条测试数据")
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
        print(f"Shot配置: {shot_desc}（来自训练集）")
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
