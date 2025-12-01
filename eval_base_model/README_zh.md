# LLM 基座模型评估框架

大语言模型基座模型综合评估框架，支持 30+ 评估任务，覆盖数学、代码、知识问答、推理、中文等多个领域。

## 特性

- 🚀 **30+ 评估任务**: GSM8K、MATH、MMLU、HumanEval、BBH、AGIEval、C-Eval、CMMLU 等
- 🔧 **统一接口**: 兼容 OpenAI API，支持各种开源模型
- 📊 **多种评估方法**: 生成方法 (Generation) 和困惑度方法 (PPL)
- 🎯 **高效并发**: 使用 ThreadPoolExecutor，支持自定义并发数
- 🔐 **安全配置**: 环境变量管理 API 凭证
- 📦 **模块化设计**: 公共工具模块 `eval_utils.py`，易于扩展

## 快速开始

### 1. 环境配置

本基座模型评估框架是 OpenCompass 项目的一部分。请先按照 OpenCompass 的要求搭建基础环境，然后再安装基座模型评估所需的额外依赖。

**第一步：搭建 OpenCompass 环境**

请参考 [OpenCompass 安装指南](../README_zh-CN.md) 搭建基础环境：

```bash
# 进入 OpenCompass 根目录
# 创建 conda 虚拟环境（推荐）
conda create --name opencompass python=3.10 -y
conda activate opencompass

# 安装 OpenCompass
pip install -U opencompass
# 或从源码安装以使用最新功能：
# git clone https://github.com/open-compass/opencompass opencompass
# cd opencompass
# pip install -e .
```

**第二步：安装基座模型评估的额外依赖**

```bash
# 进入基座模型评估目录
cd eval_base_model

# 安装额外依赖
pip install tqdm openai datasets pyarrow numpy

# 安装 HumanEval 代码评估工具（可选）
cd human-eval && pip install -e . && cd ..
```

### 2. API 配置

**方式一：环境变量（推荐）**

```bash
# 复制示例文件
cp .env.example .env

# 编辑 .env 文件，填入你的 API 信息
# EVAL_BASE_URL=https://your-api-endpoint/v1
# EVAL_API_KEY=your-api-key
# EVAL_MODEL_NAME=your-model-name
```

**方式二：配置文件**

```bash
# 复制示例文件
cp eval_config.example.json eval_config.json

# 自定义各数据集参数
```

### 3. 下载数据集

```bash
# 交互式下载（推荐）
python download_datasets.py

# 下载指定数据集
python download_datasets.py --dataset gsm8k
python download_datasets.py --dataset mmlu

# BBH 数据集说明：CoT prompts 存储在 datasets/bbh/lib_prompt/
# 这些 prompts 是 BBH 评估必需的
```

**BBH 特别说明：**
- BBH 需要任务特定的 CoT（思维链）prompts，存储在 `datasets/bbh/lib_prompt/`
- 27 个任务各有对应的 `.txt` 文件，包含 few-shot 示例
- 下载 BBH 数据集会自动包含这些 prompts
- **请勿删除** `lib_prompt/` 目录

### 4. 运行评估

```bash
# 使用环境变量中的配置
python eval_gsm8k.py

# 或手动指定参数
python eval_gsm8k.py \
  --base-url https://your-api/v1 \
  --model your-model \
  --max-workers 32 \
  --shot-num 8

# 快速测试（采样 100 个样本）
python eval_mmlu.py --max-samples 100 --seed 42

# 并行运行多个评估
python eval_gsm8k.py &
python eval_math.py &
python eval_mmlu.py &
wait
```

### 5. 使用 run_eval.py 批量评估

`run_eval.py` 脚本提供了统一的接口，可以通过单个命令运行多个评估子集。所有参数都有合理的默认值，从 `.env` 文件自动加载。

**快速开始：**

```bash
# 运行所有评估（全量测试） - 使用 .env 中的所有默认值
python run_eval.py

# 运行指定子集
python run_eval.py --subset english
python run_eval.py --subset chinese
python run_eval.py --subset code
python run_eval.py --subset math

# 显式运行所有子集
python run_eval.py --subset all
```

**高级用法：**

```bash
# 快速测试（采样 100 个样本，3-shot）
python run_eval.py --subset english --max-samples 100 --shot-num 3

# 并行执行（同时运行 4 个评估）
python run_eval.py --subset all --parallel --max-parallel 4

# 自定义配置
python run_eval.py \
  --subset english \
  --shot-num 5 \
  --max-workers 64 \
  --seed 42

# 覆盖模型设置（不使用 .env）
python run_eval.py \
  --subset code \
  --model custom-model \
  --base-url https://custom-api/v1
```

**可用子集：**

- `english`: MMLU、MMLU-Pro、MMLU-Redux、BBH、DROP、ARC、HellaSwag、PIQA、WinoGrande、RACE、AGIEval、NQ、TriviaQA
- `chinese`: C3、CCPM、CLUEWSC、C-Eval、CMMLU、CMRC
- `code`: HumanEval、MBPP、LiveCodeBench、CRUXEval-I、CRUXEval-O
- `math`: GSM8K、MATH、MGSM、CMATH
- `all`: 以上所有子集

**输出结果：**

结果保存为每个评估的单独日志文件，以及一个汇总文件：

```
logs/{model_name}/
├── gsm8k_5shot.json
├── math_5shot.json
├── mmlu_5shot.json
├── ...
└── summary_english_5shot.json  # 汇总结果
```

汇总文件包含：
- 总体统计信息（总数/成功/失败的评估数）
- 所有任务的平均准确率
- 每个评估的结果（准确率和运行时间）
- 使用的配置（shot_num、max_samples、seed）

**常用模式：**

```bash
# 完整基准测试（所有数据集，默认 5-shot）
python run_eval.py

# 快速健全性检查（所有子集，每个采样 100 个样本）
python run_eval.py --max-samples 100 --shot-num 3

# 生产基准测试（仅英文数据集，并行执行）
python run_eval.py --subset english --parallel --max-parallel 8

# 比较不同的 shot 数量
python run_eval.py --subset math --shot-num 0  # 0-shot
python run_eval.py --subset math --shot-num 3  # 3-shot
python run_eval.py --subset math --shot-num 8  # 8-shot
```

## 目录结构

```
.
├── eval_utils.py              # 公共工具模块（重试、参数解析、答案提取等）
├── eval_*.py                  # 各数据集评估脚本（31 个文件）
├── download_datasets.py       # 数据集下载工具
├── create_triviaqa_fewshot.py # TriviaQA few-shot 样本生成器
├── datasets/                  # 数据集存储目录（不上传到 git）
│   ├── bbh/
│   │   ├── *.json            # 27 个任务数据文件
│   │   └── lib_prompt/       # 各任务的 CoT prompts（必需）
│   ├── gsm8k/
│   ├── mmlu/
│   └── ...
├── logs/                      # 评估结果日志（不上传到 git）
│   └── {model_name}/         # 按模型名称组织的结果
├── archive/                   # 旧版本脚本存档
├── .env.example              # 环境变量模板
├── eval_config.example.json  # 配置文件模板
├── README_zh.md              # 中文文档
└── README_en.md              # 英文文档
```

## 支持的评估任务

### 数学推理
- **GSM8K** (8-shot): 小学数学应用题
- **MATH** (4-shot): 竞赛级数学问题
- **MGSM** (8-shot): 多语言小学数学
- **CMATH** (8-shot): 中文数学问题

### 代码生成
- **HumanEval** (0-shot): Python 函数合成
- **MBPP** (3-shot): Python 编程问题
- **LiveCodeBench** (0-shot): 真实编程任务
- **CRUXEval-I** (1-shot): 代码执行输入预测
- **CRUXEval-O** (1-shot): 代码执行输出预测

### 知识与推理
- **MMLU** (5-shot, PPL): 57 个学科，知识问答
- **MMLU-Pro** (5-shot, PPL): 更难的 MMLU 版本
- **MMLU-Redux** (5-shot, PPL): 精炼的 MMLU 子集
- **BBH** (3-shot, CoT): 27 个挑战性推理任务
- **AGIEval** (0-shot): 人类考试题目
- **ARC-Easy** (25-shot, PPL): 科学问题
- **ARC-Challenge** (25-shot, PPL): 更难的科学问题
- **HellaSwag** (10-shot, PPL): 常识推理
- **PIQA** (0-shot, PPL): 物理推理
- **WinoGrande** (5-shot, PPL): 代词消歧

### 中文评估
- **C-Eval** (5-shot, PPL): 52 个学科，中文知识
- **CMMLU** (5-shot, PPL): 67 个学科，中文知识
- **C3** (0-shot): 中文多选问答
- **CMRC** (1-shot): 中文阅读理解
- **CCPM** (0-shot, PPL): 中文阅读理解
- **CLUEWSC** (0-shot, PPL): 中文 Winograd 消歧

### 问答与阅读理解
- **TriviaQA** (5-shot): 知识问答
- **NaturalQuestions** (5-shot): Google 搜索问题
- **DROP** (3-shot): 段落离散推理
- **RACE** (0-shot): 考试阅读理解

## 评估方法

### 生成方法 (Generation)
用于：GSM8K、MATH、HumanEval、MBPP、TriviaQA、NQ、DROP 等

- 模型生成完整解答/答案
- 使用正则表达式提取最终答案
- 与参考答案比对（精确匹配或代码执行）

### 困惑度方法 (PPL - Perplexity)
用于：MMLU、C-Eval、ARC、HellaSwag、PIQA、WinoGrande 等

- 对每个选项计算对数概率
- 选择困惑度最低（似然度最高）的选项
- 更适合基座模型的多选题任务

### 思维链提示 (CoT - Chain-of-Thought)
用于：BBH

- 提供任务特定的 CoT 示例（来自 `lib_prompt/`）
- 模型生成分步推理过程
- 从推理链中提取最终答案

## 配置说明

### 环境变量 (.env)

```bash
EVAL_BASE_URL=http://localhost:8000/v1
EVAL_API_KEY=EMPTY
EVAL_MODEL_NAME=qwen2-5-72b
```

### 数据集默认配置 (eval_utils.py)

每个数据集的默认配置（shot_num、max_workers、max_tokens）：

```python
"gsm8k": {"shot_num": 8, "max_workers": 32, "max_tokens": 1024}
"mmlu": {"shot_num": 5, "max_workers": 128, "max_tokens": 1, "method": "ppl"}
"humaneval": {"shot_num": 0, "max_workers": 16, "max_tokens": 512}
# ... 完整列表见 eval_utils.py
```

### 命令行覆盖

所有参数都可以通过命令行覆盖：

```bash
python eval_gsm8k.py \
  --base-url https://api.example.com/v1 \
  --model my-model \
  --api-key sk-xxx \
  --shot-num 5 \
  --max-workers 64 \
  --max-tokens 2048 \
  --max-samples 100
```

## 输出格式

结果保存在 `logs/{model_name}/{dataset}_{shot}shot.json`：

```json
{
  "accuracy": 85.5,
  "correct": 855,
  "total": 1000,
  "shot_num": 8,
  "results": [
    {
      "question": "...",
      "prediction": "...",
      "reference": "...",
      "is_correct": true
    }
  ]
}
```

多任务评估（BBH、MMLU、AGIEval）：

```json
{
  "overall_accuracy": 67.5,
  "total_correct": 1350,
  "total_count": 2000,
  "num_tasks": 27,
  "tasks": {
    "task1": {"accuracy": 75.0, "correct": 60, "total": 80, "results": [...]},
    "task2": {"accuracy": 82.5, "correct": 66, "total": 80, "results": [...]}
  }
}
```

## 重要说明

### BBH Prompts
- BBH 需要预存储的 CoT prompts，位于 `datasets/bbh/lib_prompt/`
- 包含 27 个任务特定的 `.txt` 文件，带有 few-shot 示例
- 下载 BBH 数据集时会自动包含这些文件
- 这些 prompts 是 BBH 评估正常工作的必要条件

### 基座模型 vs 指令模型
- 本框架专为**基座模型**设计
- 使用 completion API (`/v1/completions`)，而非 chat API
- 指令模型使用 chat 风格的提示词可能表现更好

### Few-Shot 示例
- Few-shot 示例从数据集文件中加载
- 部分数据集（MMLU、C-Eval）使用学科特定的 few-shot
- TriviaQA/NQ 需要先运行 `create_triviaqa_fewshot.py` 生成示例

### 代码执行安全
- HumanEval/MBPP/LiveCodeBench 会执行生成的代码
- 代码执行在独立进程中隔离
- 设有超时防止无限循环
- 生产环境启用代码执行前请检查生成的代码

## 故障排除

**数据集未找到：**
```bash
# 重新下载数据集
python download_datasets.py --dataset {dataset_name}
```

**BBH 评估失败：**
```bash
# 确认 lib_prompt 存在
ls datasets/bbh/lib_prompt/
# 应该显示 27 个 .txt 文件

# 如果缺失，重新下载 BBH
python download_datasets.py --dataset bbh
```

**API 连接错误：**
```bash
# 检查 .env 文件
cat .env

# 手动测试 API
curl -X POST https://your-api/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "your-model", "prompt": "Hello", "max_tokens": 5}'
```

**内存问题：**
```bash
# 减少并发数
python eval_mmlu.py --max-workers 16

# 对大数据集使用采样
python eval_mmlu.py --max-samples 500
```

## 引用

如果使用本框架，请引用相应的评估任务论文：

- GSM8K: Cobbe et al., 2021
- MATH: Hendrycks et al., 2021
- MMLU: Hendrycks et al., 2020
- HumanEval: Chen et al., 2021
- BBH: Suzgun et al., 2022
- 其他任务请参考各自的论文

## 许可证

本框架采用 MIT 许可证。各数据集可能有各自的许可证。
