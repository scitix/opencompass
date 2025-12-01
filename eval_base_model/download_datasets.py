#!/usr/bin/env python3
"""数据集下载脚本 - 统一管理所有评估数据集的下载"""
import argparse
import json
import os
import shutil
import subprocess
from typing import Tuple

# 数据集配置
DATASET_CONFIGS = {
    'mmlu': {
        'type': 'huggingface',
        'source': 'cais/mmlu',
        'config': 'all',
        'target_dir': 'datasets/cais_mmlu',
        'description': 'MMLU 数据集（标准版）',
    },
    'mmlu_redux': {
        'type': 'huggingface',
        'source': 'edinburgh-dawg/mmlu-redux',
        'config': None,
        'target_dir': 'datasets/mmlu_redux',
        'description': 'MMLU-Redux 数据集',
    },
    'mmlu_pro': {
        'type': 'huggingface',
        'source': 'TIGER-Lab/MMLU-Pro',
        'config': None,
        'target_dir': 'datasets/mmlu_pro',
        'description': 'MMLU-Pro 数据集',
    },
    'arc_easy': {
        'type': 'huggingface',
        'source': 'opencompass/ai2_arc',
        'config': 'ARC-Easy',
        'target_dir': 'datasets/arc_easy',
        'description': 'ARC-Easy 数据集',
    },
    'arc_challenge': {
        'type': 'huggingface',
        'source': 'opencompass/ai2_arc',
        'config': 'ARC-Challenge',
        'target_dir': 'datasets/arc_challenge',
        'description': 'ARC-Challenge 数据集',
    },
    'agieval': {
        'type': 'github',
        'source': 'https://github.com/ruixiangcui/AGIEval.git',
        'config': 'data/v1',
        'target_dir': 'datasets/agieval',
        'description': 'AGIEval 数据集',
    },
    'hellaswag': {
        'type': 'huggingface',
        'source': 'rowan/hellaswag',
        'config': None,
        'target_dir': 'datasets/hellaswag',
        'description': 'HellaSwag 数据集',
    },
    'piqa': {
        'type': 'huggingface',
        'source': 'ybisk/piqa',
        'config': None,
        'target_dir': 'datasets/piqa',
        'description': 'PIQA 数据集',
    },
    'winogrande': {
        'type': 'huggingface',
        'source': 'allenai/winogrande',
        'config': 'winogrande_xl',
        'target_dir': 'datasets/winogrande',
        'description': 'WinoGrande 数据集',
    },
    'race': {
        'type': 'huggingface',
        'source': 'ehovy/race',
        'config': None,
        'target_dir': 'datasets/race',
        'description': 'RACE 数据集',
    },
    'triviaqa': {
        'type': 'huggingface',
        'source': 'mandarjoshi/trivia_qa',
        'config': 'rc',
        'target_dir': 'datasets/triviaqa',
        'description': 'TriviaQA 数据集 (rc subset, validation only, few-shot 需单独创建)',
    },
    'nq': {
        'type': 'huggingface',
        'source': 'google-research-datasets/natural_questions',
        'config': 'default',
        'target_dir': 'datasets/nq',
        'description': 'NaturalQuestions 数据集 (default subset, 18.5k)',
    },
    'drop': {
        'type': 'huggingface',
        'source': 'ucinlp/drop',
        'config': None,
        'target_dir': 'datasets/drop',
        'description': 'DROP 数据集',
    },
    # 中文数据集
    'ceval': {
        'type': 'huggingface_ceval',
        'source': 'ceval/ceval-exam',
        'config': None,
        'target_dir': 'datasets/ceval/formal_ceval',
        'description': 'C-Eval 数据集（52个学科）',
    },
    'cmmlu': {
        'type': 'huggingface',
        'source': 'haonan-li/cmmlu',
        'config': None,
        'target_dir': 'datasets/cmmlu',
        'description': 'CMMLU 数据集',
    },
    'c3': {
        'type': 'github',
        'source': 'https://github.com/nlpdata/c3.git',
        'config': 'data',
        'target_dir': 'datasets/c3',
        'description': 'C3 数据集（dialogue + mixed-genre）',
    },
    'cluewsc': {
        'type': 'huggingface',
        'source': 'ChineseGLUE/cluewsc_public',
        'config': None,
        'target_dir': 'datasets/cluewsc',
        'description': 'CLUEWSC 数据集',
    },
    'cmrc': {
        'type': 'huggingface',
        'source': 'cmrc2018/cmrc2018',
        'config': None,
        'target_dir': 'datasets/cmrc',
        'description': 'CMRC 2018 数据集',
    },
    # 数学数据集
    'gsm8k': {
        'type': 'huggingface',
        'source': 'openai/gsm8k',
        'config': 'main',
        'target_dir': 'datasets/gsm8k',
        'description': 'GSM8K 数据集',
    },
    'math': {
        'type': 'huggingface',
        'source': 'lighteval/MATH',
        'config': None,
        'target_dir': 'datasets/math',
        'description': 'MATH 数据集',
    },
    'mgsm': {
        'type': 'huggingface',
        'source': 'juletxara/mgsm',
        'config': None,
        'target_dir': 'datasets/mgsm',
        'description': 'MGSM 数据集（多语言）',
    },
    'cmath': {
        'type': 'huggingface',
        'source': 'weitianwen/cmath',
        'config': None,
        'target_dir': 'datasets/cmath',
        'description': 'CMATH 数据集（中国小学数学）',
    },
    # 代码数据集
    'humaneval': {
        'type': 'huggingface',
        'source': 'openai/openai_humaneval',
        'config': None,
        'target_dir': 'datasets/humaneval',
        'description': 'HumanEval 数据集（164个Python编程问题）',
    },
    'mbpp': {
        'type': 'huggingface',
        'source': 'google-research-datasets/mbpp',
        'config': 'full',
        'target_dir': 'datasets/mbpp',
        'description': 'MBPP 数据集（Mostly Basic Python Problems）',
    },
    'livecodebench': {
        'type': 'huggingface_livecodebench',
        'source': 'livecodebench/code_generation_lite',
        'config': None,
        'target_dir': 'datasets/livecodebench',
        'description': 'LiveCodeBench-Base 数据集（代码生成，支持多个版本）',
    },
}


def download_huggingface_dataset(source: str, config: str, target_dir: str, splits: list = None):
    """从 HuggingFace 下载数据集"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets 库: pip install datasets")
    
    print(f"正在从 HuggingFace 下载: {source}")
    if config:
        print(f"配置: {config}")
    print(f"目标目录: {target_dir}")
    
    os.makedirs(target_dir, exist_ok=True)
    
    if splits is None:
        splits = ['test', 'validation', 'train']
    
    downloaded_files = []
    
    for split in splits:
        try:
            if config:
                dataset = load_dataset(source, config, split=split)
            else:
                dataset = load_dataset(source, split=split)
            
            # 保存为 JSONL 格式（统一格式）
            jsonl_path = os.path.join(target_dir, f"{split}.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for item in dataset:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
            downloaded_files.append(jsonl_path)
            print(f"  ✓ {split}: {jsonl_path} ({len(dataset)} 个样本)")
        except Exception as e:
            print(f"  ⚠ {split}: 下载失败或不存在 - {e}")
    
    return downloaded_files


def download_arc_dataset(source: str, config: str, target_dir: str):
    """下载 ARC 数据集（特殊处理）"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets 库: pip install datasets")
    
    print(f"正在从 HuggingFace 下载 ARC 数据集: {source} ({config})")
    print(f"目标目录: {target_dir}")
    
    os.makedirs(target_dir, exist_ok=True)
    
    # ARC 数据集需要下载 test 和 validation
    test_dataset = load_dataset(source, name=config, split="test")
    dev_dataset = load_dataset(source, name=config, split="validation")
    
    # 保存为 JSONL 格式
    test_path = os.path.join(target_dir, f"{config}-Test.jsonl")
    dev_path = os.path.join(target_dir, f"{config}-Dev.jsonl")
    
    with open(test_path, "w", encoding="utf-8") as f:
        for item in test_dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    with open(dev_path, "w", encoding="utf-8") as f:
        for item in dev_dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"✓ 数据集已下载并保存到:")
    print(f"  - Test: {test_path} ({len(test_dataset)} 个样本)")
    print(f"  - Dev: {dev_path} ({len(dev_dataset)} 个样本)")
    
    return [test_path, dev_path]


def download_race_dataset(source: str, target_dir: str):
    """下载 RACE 数据集（特殊处理，需要 middle 和 high）"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets 库: pip install datasets")
    
    print(f"正在从 HuggingFace 下载 RACE 数据集: {source}")
    print(f"目标目录: {target_dir}")
    
    os.makedirs(target_dir, exist_ok=True)
    
    downloaded_files = []
    
    for level in ['middle', 'high']:
        for split in ['test', 'validation']:
            try:
                dataset = load_dataset(source, level, split=split)
                
                split_dir = os.path.join(target_dir, split)
                os.makedirs(split_dir, exist_ok=True)
                
                jsonl_path = os.path.join(split_dir, f"{level}.jsonl")
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for item in dataset:
                        record = {
                            "article": item.get("article", ""),
                            "question": item.get("question", ""),
                            "options": item.get("options", ["", "", "", ""]),
                            "answer": item.get("answer", ""),
                        }
                        json.dump(record, f, ensure_ascii=False)
                        f.write("\n")
                
                downloaded_files.append(jsonl_path)
                print(f"  ✓ {level} {split}: {jsonl_path} ({len(dataset)} 个样本)")
            except Exception as e:
                print(f"  ⚠ {level} {split}: 下载失败 - {e}")
    
    return downloaded_files


def download_triviaqa_dataset(source: str, config: str, target_dir: str):
    """下载 TriviaQA 数据集（特殊处理）
    
    注意: train 数据用于 few-shot，如果需要，请手动创建 few-shot.jsonl
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets 库: pip install datasets")
    
    print(f"正在从 HuggingFace 下载 TriviaQA 数据集: {source} ({config})")
    print(f"目标目录: {target_dir}")
    
    os.makedirs(target_dir, exist_ok=True)
    
    downloaded_files = []
    
    # 只下载 validation（用作测试集）
    # few-shot 数据请手动创建 few-shot.jsonl
    for split in ['validation']:
        try:
            jsonl_path = os.path.join(target_dir, f"{split}.jsonl")
            
            # 检查文件是否已存在
            if os.path.exists(jsonl_path):
                print(f"  ✓ {split}: {jsonl_path} 已存在，跳过下载")
                downloaded_files.append(jsonl_path)
                continue
            
            print(f"  正在下载 {split}...")
            dataset = load_dataset(source, config, split=split)
            
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for sample in dataset:
                    answers = []
                    answer_obj = sample.get("answer", {})
                    value = answer_obj.get("value", "")
                    aliases = answer_obj.get("aliases", []) or []
                    if value:
                        answers.append(value)
                    for alias in aliases:
                        if alias and alias not in answers:
                            answers.append(alias)
                    record = {
                        "question": sample.get("question", ""),
                        "answers": answers,
                    }
                    json.dump(record, f, ensure_ascii=False)
                    f.write("\n")
            
            downloaded_files.append(jsonl_path)
            print(f"  ✓ {split}: {jsonl_path} ({len(dataset)} 个样本)")
        except Exception as e:
            print(f"  ⚠ {split}: 下载失败 - {e}")
    
    # 检查 few-shot.jsonl 是否存在
    few_shot_path = os.path.join(target_dir, "few-shot.jsonl")
    if not os.path.exists(few_shot_path):
        print(f"\n  ⚠️  未找到 {few_shot_path}")
        print(f"  请运行: python create_triviaqa_fewshot.py")
    else:
        print(f"  ✓ few-shot: {few_shot_path} 已存在")
        downloaded_files.append(few_shot_path)
    
    return downloaded_files


def download_nq_dataset(source: str, target_dir: str):
    """下载 NaturalQuestions 数据集（特殊处理）
    
    下载 'default' subset（简化版本，约 18.5k 样本）
    - train: 10.6k samples
    - validation: 7.83k samples
    
    NQ 数据集结构：
    - question: {"text": "...", "tokens": [...]}
    - annotations: {"short_answers": [{...}], "yes_no_answer": [...]}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets 库: pip install datasets")
    
    print(f"正在从 HuggingFace 下载 NaturalQuestions 数据集: {source}")
    print(f"目标目录: {target_dir}")
    print("注意: 下载 'default' subset（约 18.5k 样本）")
    
    os.makedirs(target_dir, exist_ok=True)
    
    downloaded_files = []
    
    for split in ['validation', 'train']:
        try:
            print(f"\n正在加载 {split} split...")
            # 使用 'default' subset
            dataset = load_dataset(source, 'default', split=split)
            
            jsonl_path = os.path.join(target_dir, f"{split}.jsonl")
            valid_count = 0
            
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for sample in dataset:
                    # 提取问题文本
                    question_obj = sample.get("question", {})
                    if isinstance(question_obj, dict):
                        question_text = question_obj.get("text", "")
                    else:
                        question_text = str(question_obj)
                    
                    if not question_text:
                        continue
                    
                    # 提取答案（从 annotations 中）
                    # NQ 的 annotations 结构：{'short_answers': [{...}, {...}], 'yes_no_answer': [...]}
                    answers = []
                    annotations = sample.get("annotations", {})
                    
                    if annotations and isinstance(annotations, dict):
                        # annotations 是字典，每个字段都是列表（多个标注者）
                        short_answers_list = annotations.get("short_answers", [])
                        yes_no_list = annotations.get("yes_no_answer", [])
                        
                        # 遍历每个标注者的答案
                        if isinstance(short_answers_list, list):
                            for short_answer_dict in short_answers_list:
                                if isinstance(short_answer_dict, dict):
                                    # short_answer_dict 格式：{'text': ['answer1', 'answer2'], ...}
                                    texts = short_answer_dict.get("text", [])
                                    if isinstance(texts, list):
                                        for text in texts:
                                            if text and text not in answers:
                                                answers.append(text)
                        
                        # 如果没有 short_answer，尝试 yes_no_answer
                        if not answers and isinstance(yes_no_list, list):
                            for yes_no in yes_no_list:
                                if yes_no == 0:
                                    answers.append("no")
                                    break
                                elif yes_no == 1:
                                    answers.append("yes")
                                    break
                    
                    # 保存所有样本（包括没有答案的，因为 test set 可能没答案）
                    record = {
                        "question": question_text,
                        "answers": answers,
                    }
                    json.dump(record, f, ensure_ascii=False)
                    f.write("\n")
                    valid_count += 1
            
            downloaded_files.append(jsonl_path)
            print(f"  ✓ {split}: {jsonl_path} (原始: {len(dataset)} 个样本, 保存: {valid_count} 个样本)")
        except Exception as e:
            print(f"  ⚠ {split}: 下载失败 - {e}")
            import traceback
            traceback.print_exc()
    
    return downloaded_files


def download_mmlu_dataset(source: str, config: str, target_dir: str):
    """下载 MMLU 数据集（特殊处理，需要按 subject 分组）"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets 库: pip install datasets")
    
    print(f"正在从 HuggingFace 下载 MMLU 数据集: {source}")
    print(f"目标目录: {target_dir}")
    print("注意: 这可能需要一些时间，请耐心等待...")
    
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        dataset = load_dataset(source, config)
    except Exception as e:
        raise RuntimeError(f"无法从 HuggingFace 加载数据集 {source}: {e}")
    
    # 创建 dev 和 test 目录
    dev_dir = os.path.join(target_dir, "dev")
    test_dir = os.path.join(target_dir, "test")
    os.makedirs(dev_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    downloaded_files = []
    
    # 处理 dev 集
    if "dev" in dataset:
        dev_data = dataset["dev"]
        subjects = set()
        for item in dev_data:
            subject = item.get("subject", "miscellaneous")
            subjects.add(subject)
        
        for subject in subjects:
            subject_data = [item for item in dev_data if item.get("subject") == subject]
            if subject_data:
                jsonl_path = os.path.join(dev_dir, f"{subject}.jsonl")
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for item in subject_data:
                        json.dump(item, f, ensure_ascii=False)
                        f.write("\n")
                downloaded_files.append(jsonl_path)
                print(f"  ✓ dev/{subject}.jsonl: {len(subject_data)} 个样本")
    
    # 处理 test 集
    if "test" in dataset:
        test_data = dataset["test"]
        subjects = set()
        for item in test_data:
            subject = item.get("subject", "miscellaneous")
            subjects.add(subject)
        
        for subject in subjects:
            subject_data = [item for item in test_data if item.get("subject") == subject]
            if subject_data:
                jsonl_path = os.path.join(test_dir, f"{subject}.jsonl")
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for item in subject_data:
                        json.dump(item, f, ensure_ascii=False)
                        f.write("\n")
                downloaded_files.append(jsonl_path)
                print(f"  ✓ test/{subject}.jsonl: {len(subject_data)} 个样本")
    
    return downloaded_files


def download_ceval_dataset(source: str, target_dir: str):
    """下载 C-Eval 数据集（特殊处理，52个学科，每个有dev/val/test）"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets 库: pip install datasets")
    
    print(f"正在从 HuggingFace 下载 C-Eval 数据集: {source}")
    print(f"目标目录: {target_dir}")
    print("注意: C-Eval包含52个学科，可能需要较长时间...")
    
    os.makedirs(target_dir, exist_ok=True)
    
    # C-Eval的所有学科（52个）
    subjects = [
        'accountant', 'advanced_mathematics', 'art_studies', 'basic_medicine', 
        'business_administration', 'chinese_language_and_literature', 'civil_servant', 
        'clinical_medicine', 'college_chemistry', 'college_economics', 'college_physics', 
        'college_programming', 'computer_architecture', 'computer_network', 
        'discrete_mathematics', 'education_science', 'electrical_engineer', 
        'environmental_impact_assessment_engineer', 'fire_engineer', 'high_school_biology', 
        'high_school_chemistry', 'high_school_chinese', 'high_school_geography', 
        'high_school_history', 'high_school_mathematics', 'high_school_physics', 
        'high_school_politics', 'ideological_and_moral_cultivation', 'law', 
        'legal_professional', 'logic', 'mao_zedong_thought', 'marxism', 'metrology_engineer', 
        'middle_school_biology', 'middle_school_chemistry', 'middle_school_geography', 
        'middle_school_history', 'middle_school_mathematics', 'middle_school_physics', 
        'middle_school_politics', 'modern_chinese_history', 'operating_system', 'physician', 
        'plant_protection', 'probability_and_statistics', 'professional_tour_guide', 
        'sports_science', 'tax_accountant', 'teacher_qualification', 'urban_and_rural_planner', 
        'veterinary_medicine'
    ]
    
    # 创建 dev, val, test 目录
    dev_dir = os.path.join(target_dir, "dev")
    val_dir = os.path.join(target_dir, "val")
    test_dir = os.path.join(target_dir, "test")
    os.makedirs(dev_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    downloaded_files = []
    
    # 为每个学科下载数据
    for subject in subjects:
        try:
            # 加载该学科的数据
            dataset = load_dataset(source, subject)
            
            # 处理每个split (dev有5个样本, val和test数量不等)
            for split_name in ['dev', 'val', 'test']:
                if split_name not in dataset:
                    continue
                
                split_data = dataset[split_name]
                split_dir = os.path.join(target_dir, split_name)
                
                # 保存为CSV
                csv_path = os.path.join(split_dir, f"{subject}.csv")
                
                # 转换为列表
                items = [dict(item) for item in split_data]
                
                # 写入CSV格式
                import csv
                with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                    if items:
                        writer = csv.DictWriter(f, fieldnames=items[0].keys())
                        writer.writeheader()
                        writer.writerows(items)
                
                downloaded_files.append(csv_path)
                print(f"  ✓ {split_name}/{subject}.csv: {len(items)} 个样本")
        
        except Exception as e:
            print(f"  ⚠️ 下载 {subject} 失败: {e}")
            continue
    
    print(f"\n✓ C-Eval数据集下载完成")
    print(f"  共 {len(downloaded_files)} 个文件")
    return downloaded_files


def download_c3_dataset(repo_url: str, data_path: str, target_dir: str):
    """从 GitHub 下载 C3 数据集（dialogue + mixed-genre）"""
    print(f"正在从 GitHub 下载 C3 数据集: {repo_url}")
    print(f"目标目录: {target_dir}")
    
    os.makedirs(target_dir, exist_ok=True)
    
    temp_dir = os.path.join(os.path.dirname(target_dir), "temp_c3_download")
    
    try:
        # 检查是否已存在临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        # 克隆仓库
        print("正在克隆 GitHub 仓库...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, temp_dir],
            check=True,
            capture_output=True
        )
        
        # 复制数据文件
        source_data_dir = os.path.join(temp_dir, data_path)
        if not os.path.exists(source_data_dir):
            raise FileNotFoundError(f"未找到数据目录: {source_data_dir}")
        
        # 复制所有 JSON 文件
        json_files = [f for f in os.listdir(source_data_dir) if f.endswith('.json')]
        if not json_files:
            raise FileNotFoundError(f"数据目录中没有找到 JSON 文件: {source_data_dir}")
        
        downloaded_files = []
        for json_file in json_files:
            src_path = os.path.join(source_data_dir, json_file)
            dst_path = os.path.join(target_dir, json_file)
            shutil.copy2(src_path, dst_path)
            downloaded_files.append(dst_path)
            
            # 统计数据量
            with open(dst_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                doc_count = len(data)
                q_count = sum(len(d[1]) for d in data)
                print(f"  ✓ {json_file}: {doc_count} 文档, {q_count} 问题")
        
        print(f"✓ C3数据集已下载并保存到: {target_dir}")
        print(f"  - 共找到 {len(json_files)} 个文件")
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
        
        return downloaded_files
        
    except subprocess.CalledProcessError as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise RuntimeError(f"Git 克隆失败: {e.stderr.decode() if e.stderr else str(e)}")
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise


def download_livecodebench_dataset(source: str, target_dir: str, version_tags: list = None):
    """下载 LiveCodeBench 数据集（支持多个版本）
    
    支持的版本：
    - release_v1: 2023年5月至2024年3月，400个问题
    - release_v2: 2023年5月至2024年5月，511个问题
    - release_v3: 2023年5月至2024年7月，612个问题
    - release_v4: 2023年5月至2024年9月，713个问题（包含2024-08-01数据）
    - release_v5: 2023年5月至2025年1月，880个问题（包含完整目标日期范围）
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets 库: pip install datasets")
    
    print(f"正在从 HuggingFace 下载 LiveCodeBench 数据集: {source}")
    print(f"目标目录: {target_dir}")
    
    if version_tags is None:
        # 默认下载 release_v4 和 release_v5（包含目标日期范围）
        version_tags = ['release_v4', 'release_v5']
    
    os.makedirs(target_dir, exist_ok=True)
    
    downloaded_files = []
    
    for version_tag in version_tags:
        try:
            print(f"\n正在下载版本: {version_tag}...")
            dataset = load_dataset(source, split='test', version_tag=version_tag)
            
            # 保存为 JSONL 格式
            jsonl_path = os.path.join(target_dir, f"test_{version_tag}.jsonl")
            
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for item in dataset:
                    # 确保日期字段正确序列化
                    record = dict(item)
                    if 'contest_date' in record and isinstance(record['contest_date'], type):
                        # 如果是datetime对象，转换为字符串
                        from datetime import datetime
                        if isinstance(record['contest_date'], datetime):
                            record['contest_date'] = record['contest_date'].isoformat()
                    
                    json.dump(record, f, ensure_ascii=False, default=str)
                    f.write("\n")
            
            downloaded_files.append(jsonl_path)
            
            # 统计日期范围
            from datetime import datetime
            dates = []
            for item in dataset:
                if 'contest_date' in item:
                    date_val = item['contest_date']
                    if isinstance(date_val, str):
                        try:
                            dates.append(datetime.fromisoformat(date_val.split('T')[0]))
                        except:
                            pass
                    elif isinstance(date_val, datetime):
                        dates.append(date_val)
            
            if dates:
                min_date = min(dates).date()
                max_date = max(dates).date()
                print(f"  ✓ {version_tag}: {jsonl_path} ({len(dataset)} 个样本)")
                print(f"    日期范围: {min_date} 至 {max_date}")
            else:
                print(f"  ✓ {version_tag}: {jsonl_path} ({len(dataset)} 个样本)")
        
        except Exception as e:
            print(f"  ⚠️ {version_tag}: 下载失败 - {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ LiveCodeBench数据集下载完成")
    print(f"  共 {len(downloaded_files)} 个文件")
    return downloaded_files


def download_github_dataset(repo_url: str, data_path: str, target_dir: str):
    """从 GitHub 下载数据集"""
    print(f"正在从 GitHub 下载数据集: {repo_url}")
    print(f"数据路径: {data_path}")
    print(f"目标目录: {target_dir}")
    
    os.makedirs(target_dir, exist_ok=True)
    
    # GitHub 仓库 URL
    temp_dir = os.path.join(os.path.dirname(target_dir), "temp_download")
    
    try:
        # 检查是否已存在临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        # 克隆仓库
        print("正在克隆 GitHub 仓库...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, temp_dir],
            check=True,
            capture_output=True
        )
        
        # 复制数据文件
        source_data_dir = os.path.join(temp_dir, data_path)
        if not os.path.exists(source_data_dir):
            raise FileNotFoundError(f"未找到数据目录: {source_data_dir}")
        
        # 复制所有 JSONL 文件
        jsonl_files = [f for f in os.listdir(source_data_dir) if f.endswith('.jsonl')]
        if not jsonl_files:
            raise FileNotFoundError(f"数据目录中没有找到 JSONL 文件: {source_data_dir}")
        
        for jsonl_file in jsonl_files:
            src_path = os.path.join(source_data_dir, jsonl_file)
            dst_path = os.path.join(target_dir, jsonl_file)
            shutil.copy2(src_path, dst_path)
        
        print(f"✓ 数据集已下载并保存到: {target_dir}")
        print(f"  - 共找到 {len(jsonl_files)} 个子集文件")
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
        
        return [os.path.join(target_dir, f) for f in jsonl_files]
        
    except subprocess.CalledProcessError as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise RuntimeError(f"Git 克隆失败: {e.stderr.decode() if e.stderr else str(e)}")
    except Exception as e:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise


def download_dataset(dataset_name: str, base_dir: str = None):
    """下载指定的数据集"""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"未知的数据集: {dataset_name}。可用数据集: {', '.join(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    base_dir = base_dir or os.path.dirname(__file__)
    target_dir = os.path.join(base_dir, config['target_dir'])
    
    print("=" * 70)
    print(f"下载数据集: {config['description']}")
    print("=" * 70)
    
    try:
        if dataset_name == 'mmlu':
            return download_mmlu_dataset(config['source'], config['config'], target_dir)
        elif dataset_name in ['arc_easy', 'arc_challenge']:
            return download_arc_dataset(config['source'], config['config'], target_dir)
        elif dataset_name == 'race':
            return download_race_dataset(config['source'], target_dir)
        elif dataset_name == 'triviaqa':
            return download_triviaqa_dataset(config['source'], config['config'], target_dir)
        elif dataset_name == 'nq':
            return download_nq_dataset(config['source'], target_dir)
        elif dataset_name == 'ceval':
            return download_ceval_dataset(config['source'], target_dir)
        elif dataset_name == 'c3':
            return download_c3_dataset(config['source'], config['config'], target_dir)
        elif dataset_name == 'drop':
            # DROP 数据集特殊处理
            return download_huggingface_dataset(config['source'], None, target_dir, ['validation', 'train'])
        elif dataset_name == 'gsm8k':
            # GSM8K 需要指定config
            return download_huggingface_dataset(config['source'], config['config'], target_dir, ['test', 'train'])
        elif dataset_name == 'math':
            # MATH 数据集
            return download_huggingface_dataset(config['source'], None, target_dir, ['test', 'train'])
        elif dataset_name == 'mgsm':
            # MGSM 多语言数据集
            return download_huggingface_dataset(config['source'], None, target_dir, ['test', 'train'])
        elif dataset_name == 'cmath':
            # CMATH 中国小学数学数据集
            return download_huggingface_dataset(config['source'], None, target_dir, ['test', 'validation'])
        elif dataset_name in ['cmmlu', 'cluewsc', 'cmrc']:
            # 中文数据集通用处理
            splits = ['test', 'dev', 'train'] if dataset_name != 'cluewsc' else ['test', 'train']
            return download_huggingface_dataset(config['source'], config['config'], target_dir, splits)
        elif dataset_name == 'humaneval':
            # HumanEval 只有 test split
            return download_huggingface_dataset(config['source'], None, target_dir, ['test'])
        elif dataset_name == 'mbpp':
            # MBPP 数据集
            return download_huggingface_dataset(config['source'], config['config'], target_dir, ['train', 'test', 'validation'])
        elif dataset_name == 'livecodebench':
            # LiveCodeBench 数据集（支持多个版本）
            return download_livecodebench_dataset(config['source'], target_dir)
        elif config['type'] == 'github':
            return download_github_dataset(config['source'], config['config'], target_dir)
        elif config['type'] == 'huggingface':
            if dataset_name == 'mmlu_redux':
                # MMLU-Redux 需要保存为 parquet
                try:
                    from datasets import load_dataset
                except ImportError:
                    raise ImportError("请安装 datasets 库: pip install datasets")
                
                os.makedirs(target_dir, exist_ok=True)
                dataset = load_dataset(config['source'], split="test")
                parquet_path = os.path.join(target_dir, "test-00000-of-00001.parquet")
                dataset.to_parquet(parquet_path)
                print(f"  ✓ test: {parquet_path} ({len(dataset)} 个样本)")
                return [parquet_path]
            elif dataset_name == 'mmlu_pro':
                # MMLU-Pro 需要保存为 parquet
                try:
                    from datasets import load_dataset
                except ImportError:
                    raise ImportError("请安装 datasets 库: pip install datasets")
                
                os.makedirs(target_dir, exist_ok=True)
                test_dataset = load_dataset(config['source'], split="test")
                validation_dataset = load_dataset(config['source'], split="validation")
                
                test_path = os.path.join(target_dir, "test-00000-of-00001.parquet")
                validation_path = os.path.join(target_dir, "validation-00000-of-00001.parquet")
                
                test_dataset.to_parquet(test_path)
                validation_dataset.to_parquet(validation_path)
                
                print(f"  ✓ test: {test_path} ({len(test_dataset)} 个样本)")
                print(f"  ✓ validation: {validation_path} ({len(validation_dataset)} 个样本)")
                return [test_path, validation_path]
            elif dataset_name == 'hellaswag':
                splits = ['validation', 'train']
            elif dataset_name == 'piqa':
                splits = ['validation']
            elif dataset_name == 'winogrande':
                splits = ['validation', 'train']
            else:
                splits = ['test', 'validation', 'train']
            
            if config['config']:
                return download_huggingface_dataset(config['source'], config['config'], target_dir, splits)
            else:
                return download_huggingface_dataset(config['source'], None, target_dir, splits)
        else:
            raise ValueError(f"不支持的数据集类型: {config['type']}")
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description="数据集下载脚本")
    parser.add_argument(
        'datasets',
        nargs='+',
        choices=list(DATASET_CONFIGS.keys()) + ['all'],
        help='要下载的数据集名称（使用 "all" 下载所有数据集）',
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='基础目录（默认: 脚本所在目录）',
    )
    
    args = parser.parse_args()
    
    if 'all' in args.datasets:
        datasets_to_download = list(DATASET_CONFIGS.keys())
    else:
        datasets_to_download = args.datasets
    
    print(f"将下载以下数据集: {', '.join(datasets_to_download)}")
    print()
    
    for dataset_name in datasets_to_download:
        try:
            download_dataset(dataset_name, args.base_dir)
            print()
        except Exception as e:
            print(f"\n✗ {dataset_name} 下载失败: {e}\n")
            continue
    
    print("=" * 70)
    print("下载完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

