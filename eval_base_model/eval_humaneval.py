"""HumanEval数据集评估脚本 - Base模型代码生成（0-shot Pass@1）。"""
import json
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from openai import OpenAI
from tqdm import tqdm
import eval_utils

HUMANEVAL_IMPORT_ERROR = '''\
Please install human_eval use following steps:
git clone git@github.com:open-compass/human-eval.git
cd human-eval && pip install -e .'''


class HumanEvalEvaluator:
    """使用代码生成方法评估HumanEval数据集（0-shot Pass@1）"""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 max_workers: int = 32, max_tokens: int = 512, shot_num: int = 0):
        # 延迟导入human_eval，只在需要时检查
        try:
            import human_eval
        except ImportError:
            raise ImportError(HUMANEVAL_IMPORT_ERROR)

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_tokens = max_tokens
        self.shot_num = shot_num

    def build_prompt(self, prompt: str) -> str:
        """构建0-shot prompt - 对齐 OpenCompass base 模型评估"""
        # Base 模型做代码补全，直接用 prompt（函数签名+docstring）
        return prompt

    def postprocess_code(self, text: str) -> str:
        """后处理生成的代码 - 完整对齐 OpenCompass humaneval_internal_v1_postprocess"""
        # 1. 处理引号包裹的字符串（如 'return ...'）
        try:
            eval_text = eval(text)
            if isinstance(eval_text, str):
                text = eval_text
        except Exception:
            pass
        
        # 2. 提取 markdown 代码块
        text = text.lstrip('\n')
        if '```' in text:
            blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
            if len(blocks) == 0:
                text = text.split('```')[1]
            else:
                text = blocks[0]
                if not text.startswith('\n'):
                    text = text[max(text.find('\n') + 1, 0):]
        
        # 3. 处理 import 语句
        if text.strip().startswith('from') or text.strip().startswith('import'):
            def_idx = text.find('def')
            if def_idx != -1:
                text = text[max(text.find('\n', def_idx) + 1, 0):]
        
        # 4. 移除空行
        text = '\n'.join([line for line in text.split('\n') if line != ''])
        text = text.lstrip('\n')
        
        # 5. 如果生成的代码以 def 开头，移除它（因为 prompt 已包含）
        if text.strip().startswith('def'):
            text = '\n'.join(text.split('\n')[1:])
        
        # 6. 处理缩进（关键！）
        if text.startswith(' '):
            text = '    ' + text.lstrip()
        else:
            text = '\n'.join(['    ' + line for line in text.split('\n')])
        
        # 7. 检测代码块结束（当缩进减少时停止）
        text_lines = text.split('\n')
        min_leading_space = None
        end_index = None
        for index, line in enumerate(text_lines):
            if line.strip() == '' or (line.strip() and line.strip()[0] in ["'", '"', '#']):
                continue
            current_leading_space = len(line.rstrip()) - len(line.strip())
            if min_leading_space is None:
                min_leading_space = current_leading_space
            elif current_leading_space < min_leading_space:
                end_index = index
                break
        
        if end_index is not None:
            text = '\n'.join(text_lines[:end_index])
        else:
            text = '\n'.join(text_lines)
        
        return text

    def generate_code(self, prompt: str, max_retries: int = 3) -> str:
        """生成代码"""
        last_exception: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=0.0,  # 使用greedy decoding
                    stop=None,
                )
                
                if not response.choices:
                    raise RuntimeError('模型返回的choices为空。')
                
                generated_text = response.choices[0].text
                return self.postprocess_code(generated_text)
                    
            except Exception as e:
                error_msg = str(e)
                
                # 速率限制处理
                if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                    wait_time = min(10 * (attempt + 1), 60)
                    time.sleep(wait_time)
                    continue
                
                # 其他错误
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    last_exception = e
                    break

        if last_exception is None:
            last_exception = RuntimeError('未知原因导致代码生成失败。')

        raise last_exception

    def evaluate_single_task(self, task_data: Dict) -> Dict:
        """评估单个任务 - 对齐 OpenCompass base 模型评估"""
        try:
            prompt = self.build_prompt(task_data['prompt'])
            completion = self.generate_code(prompt)
            
            # 对于 Base 模型：
            # - prompt 是函数签名+docstring
            # - completion 是函数体
            # - human_eval 库会自动组合 prompt + completion
            
            return {
                'task_id': task_data['task_id'],
                'prompt': task_data['prompt'],
                'generated_code': completion,  # 这是 completion，传给 human_eval
                'canonical_solution': task_data.get('canonical_solution', ''),
            }
        except Exception as e:
            print(f"\n任务 {task_data.get('task_id', 'unknown')} 生成代码时出错: {e}")
            return {
                'task_id': task_data.get('task_id', 'unknown'),
                'prompt': task_data.get('prompt', ''),
                'generated_code': '',
                'error': str(e),
            }

    def load_humaneval_data(self, data_path: Optional[str] = None) -> List[Dict]:
        """加载HumanEval数据集"""
        # 只从datasets目录查找
        if data_path is None:
            datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets', 'humaneval')
            # 尝试多个可能的文件名
            possible_files = [
                'HumanEval.jsonl',
                'human-eval-v2-20210705.jsonl',
                'data.jsonl',
            ]
            for filename in possible_files:
                data_path = os.path.join(datasets_dir, filename)
                if os.path.exists(data_path):
                    break
            else:
                raise FileNotFoundError(
                    f"未找到HumanEval数据集，请确保数据集已复制到 datasets/humaneval/ 目录"
                )
        
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        
        return data

    def evaluate_dataset(self, data: List[Dict], max_samples: Optional[int] = None) -> Dict:
        """评估整个数据集"""
        if max_samples:
            data = data[:max_samples]
        
        total = len(data)
        print(f"开始评估 {total} 个任务...")
        print(f"并发设置: {self.max_workers} 个任务并发")
        print(f"方法: {self.shot_num}-shot 代码生成")
        
        results = []
        pbar = tqdm(total=total, desc="评估进度")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self.evaluate_single_task, task): task 
                for task in data
            }
            
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                except Exception as e:
                    print(f"\n评估失败: {e}")
                    pbar.update(1)
        
        pbar.close()
        
        # 使用human_eval库进行评估
        print("\n执行代码评估...")
        # 延迟导入，确保在需要时才检查
        from human_eval.data import HUMAN_EVAL, write_jsonl
        from human_eval.evaluation import evaluate_functional_correctness

        with tempfile.TemporaryDirectory() as tmp_dir:
            predictions_file = os.path.join(tmp_dir, 'predictions.jsonl')
            
            # 准备预测数据
            predictions = []
            for result in results:
                if 'error' not in result:
                    predictions.append({
                        'task_id': result['task_id'],
                        'completion': result['generated_code']
                    })
            
            # write_jsonl 需要文件路径，不是文件对象
            write_jsonl(predictions_file, predictions)
            
            # 评估
            score = evaluate_functional_correctness(
                predictions_file,
                k=[1],
                n_workers=4,
                timeout=3.0,
                problem_file=HUMAN_EVAL
            )
            
            # 读取详细结果
            detail_path = predictions_file + '_results.jsonl'
            details = {}
            if os.path.exists(detail_path):
                with open(detail_path, 'r') as f:
                    for index, line in enumerate(f):
                        line_data = json.loads(line)
                        details[str(index)] = {
                            **line_data,
                            'is_correct': line_data.get('passed', False),
                        }
        
        # score 格式: {'pass@1': 0.4634...}（已经是小数）
        # 需要乘以 100 转换为百分比
        pass_at_1 = score.get('pass@1', score.get(1, 0.0)) * 100
        
        return {
            'pass@1': pass_at_1,
            'total': total,
            'score_details': score,
            'details': details,
            'results': results
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HumanEval evaluation script (0-shot Pass@1)")

    parser.add_argument(
        "--shot-num",
        type=int,
        default=0,
        help=f"Few-shot 示例数量（默认: 0）",
    )
    parser.add_argument("--model", type=str, default=os.environ.get("EVAL_MODEL_NAME", "qwen2-5-72b"), help="模型名称（默认: eval-qwen2-5-72b）")
    parser.add_argument("--base-url", type=str, default=os.environ.get("EVAL_BASE_URL", "http://localhost:8000/v1"), help="API base URL")
    parser.add_argument("--max-workers", type=int, default=32, help="最大并发数（默认: 32）")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大生成 token 数（默认: 512）")
    parser.add_argument("--max-samples", type=int, default=None, help="随机采样数量，None表示评估全部数据（默认: None）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认: 42）")
    
    args = parser.parse_args()
    
    # 创建日志目录结构：logs/{model}/
    log_dir = os.path.join("logs", args.model)
    os.makedirs(log_dir, exist_ok=True)
    
    output_filename = "humaneval_0shot.json"
    if args.max_samples:
        output_filename = f"humaneval_0shot_{args.max_samples}samples.json"
    OUTPUT_PATH = os.path.join(log_dir, output_filename)
    
    print("="*70)
    print(f"{args.shot_num}-shot Pass@1")
    print("="*70)
    print(f"模型: {args.model}")
    print(f"API: completions")
    print(f"{args.shot_num}-shot 代码生成")
    print("="*70)
    
    evaluator = HumanEvalEvaluator(
        args.base_url,
        args.model,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        shot_num=args.shot_num
    )
    
    print(f"\n加载数据集...")
    try:
        data = evaluator.load_humaneval_data()
        print(f"✓ 成功加载 {len(data)} 条数据")
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        return
    
    print("\n开始评估...")
    
    try:
        results = evaluator.evaluate_dataset(data, max_samples=args.max_samples)
        
        print(f"\n保存结果: {OUTPUT_PATH}")
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*70)
        print("评估结果")
        print("="*70)
        print(f"总任务数: {results['total']}")
        print(f"Pass@1: {results['pass@1']:.2f}%")
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

