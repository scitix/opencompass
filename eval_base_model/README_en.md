# LLM Base Model Evaluation Framework

Comprehensive evaluation framework for LLM base models, supporting 30+ evaluation tasks across math, code, knowledge QA, reasoning, and Chinese benchmarks.

## Features

- 🚀 **30+ Evaluation Tasks**: GSM8K, MATH, MMLU, HumanEval, BBH, AGIEval, C-Eval, CMMLU, etc.
- 🔧 **Unified Interface**: OpenAI-compatible API, supports various open-source models
- 📊 **Multiple Methods**: Generation method & Perplexity (PPL) method
- 🎯 **Efficient Concurrency**: ThreadPoolExecutor with customizable max_workers
- 🔐 **Secure Configuration**: Environment variables for API credentials
- 📦 **Modular Design**: Common utilities in `eval_utils.py`, easy to extend

## Quick Start

### 1. Environment Setup

This base model evaluation framework is part of the OpenCompass project. Please set up the OpenCompass environment first, then install additional dependencies for base model evaluation.

**Step 1: Setup OpenCompass Environment**

Follow the [OpenCompass Installation Guide](../README.md) to set up the base environment:

```bash
# Navigate to OpenCompass root directory
# Create conda environment (recommended)
conda create --name opencompass python=3.10 -y
conda activate opencompass

# Install OpenCompass
pip install -U opencompass
# Or install from source for latest features:
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
```

**Step 2: Install Additional Dependencies for Base Model Evaluation**

```bash
# Navigate to base model evaluation directory
cd eval_base_model

# Install additional dependencies
pip install tqdm openai datasets pyarrow numpy

# For HumanEval code evaluation (optional)
cd human-eval && pip install -e . && cd ..
```

### 2. API Configuration

**Method 1: Environment Variables (Recommended)**

```bash
# Copy example file
cp .env.example .env

# Edit .env file with your API info
# EVAL_BASE_URL=https://your-api-endpoint/v1
# EVAL_API_KEY=your-api-key
# EVAL_MODEL_NAME=your-model-name
```

**Method 2: Configuration File**

```bash
# Copy example file
cp eval_config.example.json eval_config.json

# Customize parameters for each dataset
```

### 3. Download Datasets

```bash
# Interactive download (recommended)
python download_datasets.py

# Download specific dataset
python download_datasets.py --dataset gsm8k
python download_datasets.py --dataset mmlu

# For BBH: CoT prompts are included in datasets/bbh/lib_prompt/
# These prompts are required for BBH evaluation
```

**Special Note for BBH:**
- BBH requires task-specific CoT (Chain-of-Thought) prompts stored in `datasets/bbh/lib_prompt/`
- Each of the 27 tasks has a corresponding `.txt` file with few-shot examples
- Download BBH dataset will automatically include these prompts
- Do NOT delete the `lib_prompt/` directory

### 4. Run Evaluations

```bash
# Use configuration from environment variables
python eval_gsm8k.py

# Or specify parameters manually
python eval_gsm8k.py \
  --base-url https://your-api/v1 \
  --model your-model \
  --max-workers 32 \
  --shot-num 8

# Quick test (sample 100 items)
python eval_mmlu.py --max-samples 100 --seed 42

# Run multiple evaluations in parallel
python eval_gsm8k.py &
python eval_math.py &
python eval_mmlu.py &
wait
```

### 5. Batch Evaluation with run_eval.py

The `run_eval.py` script provides a unified interface to run multiple evaluation subsets with a single command. All parameters have sensible defaults loaded from `.env`.

**Quick Start:**

```bash
# Run all evaluations (full test) - uses all defaults from .env
python run_eval.py

# Run specific subset
python run_eval.py --subset english
python run_eval.py --subset chinese
python run_eval.py --subset code
python run_eval.py --subset math

# Run all subsets explicitly
python run_eval.py --subset all
```

**Advanced Usage:**

```bash
# Quick test with sampling (100 samples, 3-shot)
python run_eval.py --subset english --max-samples 100 --shot-num 3

# Parallel execution (run 4 evaluations concurrently)
python run_eval.py --subset all --parallel --max-parallel 4

# Custom configuration
python run_eval.py \
  --subset english \
  --shot-num 5 \
  --max-workers 64 \
  --seed 42

# Override model settings (without .env)
python run_eval.py \
  --subset code \
  --model custom-model \
  --base-url https://custom-api/v1
```

**Available Subsets:**

- `english`: MMLU, MMLU-Pro, MMLU-Redux, BBH, DROP, ARC, HellaSwag, PIQA, WinoGrande, RACE, AGIEval, NQ, TriviaQA
- `chinese`: C3, CCPM, CLUEWSC, C-Eval, CMMLU, CMRC
- `code`: HumanEval, MBPP, LiveCodeBench, CRUXEval-I, CRUXEval-O
- `math`: GSM8K, MATH, MGSM, CMATH
- `all`: All of the above

**Output:**

Results are saved to individual log files per evaluation, plus a summary file:

```
logs/{model_name}/
├── gsm8k_5shot.json
├── math_5shot.json
├── mmlu_5shot.json
├── ...
└── summary_english_5shot.json  # Aggregated summary
```

The summary file contains:
- Overall statistics (total/successful/failed evaluations)
- Average accuracy across all tasks
- Per-evaluation results with accuracy and duration
- Configuration used (shot_num, max_samples, seed)

**Common Patterns:**

```bash
# Full benchmark suite (all datasets, default 5-shot)
python run_eval.py

# Quick sanity check (all subsets, 100 samples each)
python run_eval.py --max-samples 100 --shot-num 3

# Production benchmark (English only, parallel execution)
python run_eval.py --subset english --parallel --max-parallel 8

# Compare different shot numbers
python run_eval.py --subset math --shot-num 0  # 0-shot
python run_eval.py --subset math --shot-num 3  # 3-shot
python run_eval.py --subset math --shot-num 8  # 8-shot
```

## Directory Structure

```
.
├── eval_utils.py              # Common utilities (retry, arg parsing, answer extraction)
├── eval_*.py                  # Evaluation scripts for each dataset (31 files)
├── download_datasets.py       # Dataset download utility
├── create_triviaqa_fewshot.py # TriviaQA few-shot sample generator
├── datasets/                  # Dataset storage (not in git)
│   ├── bbh/
│   │   ├── *.json            # 27 task data files
│   │   └── lib_prompt/       # CoT prompts for each task (REQUIRED)
│   ├── gsm8k/
│   ├── mmlu/
│   └── ...
├── logs/                      # Evaluation results (not in git)
│   └── {model_name}/         # Results organized by model
├── archive/                   # Archived old scripts
├── .env.example              # Environment variable template
├── eval_config.example.json  # Configuration file template
└── README.md
```

## Supported Benchmarks

### Math Reasoning
- **GSM8K** (8-shot): Grade school math problems
- **MATH** (4-shot): Competition-level math problems
- **MGSM** (8-shot): Multilingual grade school math
- **CMATH** (8-shot): Chinese math problems

### Code Generation
- **HumanEval** (0-shot): Python function synthesis
- **MBPP** (3-shot): Python programming problems
- **LiveCodeBench** (0-shot): Real-world coding tasks
- **CRUXEval-I** (1-shot): Code execution input prediction
- **CRUXEval-O** (1-shot): Code execution output prediction

### Knowledge & Reasoning
- **MMLU** (5-shot, PPL): 57 subjects, knowledge Q&A
- **MMLU-Pro** (5-shot, PPL): Harder version of MMLU
- **MMLU-Redux** (5-shot, PPL): Refined MMLU subset
- **BBH** (3-shot, CoT): 27 challenging reasoning tasks
- **AGIEval** (0-shot): Human exam questions
- **ARC-Easy** (25-shot, PPL): Science questions
- **ARC-Challenge** (25-shot, PPL): Harder science questions
- **HellaSwag** (10-shot, PPL): Commonsense reasoning
- **PIQA** (0-shot, PPL): Physical reasoning
- **WinoGrande** (5-shot, PPL): Pronoun resolution

### Chinese Benchmarks
- **C-Eval** (5-shot, PPL): 52 subjects, Chinese knowledge
- **CMMLU** (5-shot, PPL): 67 subjects, Chinese knowledge
- **C3** (0-shot): Chinese multiple-choice QA
- **CMRC** (1-shot): Chinese reading comprehension
- **CCPM** (0-shot, PPL): Chinese reading comprehension
- **CLUEWSC** (0-shot, PPL): Chinese Winograd Schema

### QA & Reading Comprehension
- **TriviaQA** (5-shot): Trivia question answering
- **NaturalQuestions** (5-shot): Google search questions
- **DROP** (3-shot): Discrete reasoning over paragraphs
- **RACE** (0-shot): Reading comprehension from exams

## Evaluation Methods

### Generation Method
Used for: GSM8K, MATH, HumanEval, MBPP, TriviaQA, NQ, DROP, etc.

- Model generates full solution/answer
- Extract final answer using regex patterns
- Compare with reference answer (EM or execution-based)

### PPL (Perplexity) Method
Used for: MMLU, C-Eval, ARC, HellaSwag, PIQA, WinoGrande, etc.

- For each option, compute log probability
- Select option with lowest perplexity (highest likelihood)
- More suitable for base models on multiple-choice tasks

### CoT (Chain-of-Thought) Prompting
Used for: BBH

- Provide task-specific CoT examples from `lib_prompt/`
- Model generates step-by-step reasoning
- Extract final answer from reasoning chain

## Configuration

### Environment Variables (.env)

```bash
EVAL_BASE_URL=http://localhost:8000/v1
EVAL_API_KEY=EMPTY
EVAL_MODEL_NAME=qwen2-5-72b
```

### Per-Dataset Defaults (eval_utils.py)

Default configurations for each dataset (shot_num, max_workers, max_tokens):

```python
"gsm8k": {"shot_num": 8, "max_workers": 32, "max_tokens": 1024}
"mmlu": {"shot_num": 5, "max_workers": 128, "max_tokens": 1, "method": "ppl"}
"humaneval": {"shot_num": 0, "max_workers": 16, "max_tokens": 512}
# ... see eval_utils.py for full list
```

### Command-Line Override

All parameters can be overridden via command-line:

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

## Output Format

Results are saved to `logs/{model_name}/{dataset}_{shot}shot.json`:

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

For multi-task benchmarks (BBH, MMLU, AGIEval):

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

## Important Notes

### BBH Prompts
- BBH requires pre-stored CoT prompts in `datasets/bbh/lib_prompt/`
- 27 task-specific `.txt` files with few-shot examples
- These are downloaded automatically with the BBH dataset
- Prompts are essential for BBH evaluation to work

### Base Model vs Instruct Model
- This framework is designed for **base models**
- Uses completion API (`/v1/completions`), not chat API
- Instruct models may perform better with chat-style prompting

### Few-Shot Examples
- Few-shot examples are loaded from dataset files
- Some datasets (MMLU, C-Eval) use subject-specific few-shot
- TriviaQA/NQ require running `create_triviaqa_fewshot.py` first

### Code Execution Safety
- HumanEval/MBPP/LiveCodeBench execute generated code
- Code execution is isolated in separate processes
- Timeouts prevent infinite loops
- Review generated code before enabling execution in production

## Troubleshooting

**Dataset not found:**
```bash
# Re-download the dataset
python download_datasets.py --dataset {dataset_name}
```

**BBH evaluation fails:**
```bash
# Ensure lib_prompt exists
ls datasets/bbh/lib_prompt/
# Should show 27 .txt files

# If missing, re-download BBH
python download_datasets.py --dataset bbh
```

**API connection errors:**
```bash
# Check .env file
cat .env

# Test API manually
curl -X POST https://your-api/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "your-model", "prompt": "Hello", "max_tokens": 5}'
```

**Memory issues:**
```bash
# Reduce max_workers
python eval_mmlu.py --max-workers 16

# Use sampling for large datasets
python eval_mmlu.py --max-samples 500
```

## Citation

If you use this framework, please cite the respective benchmark papers:

- GSM8K: Cobbe et al., 2021
- MATH: Hendrycks et al., 2021
- MMLU: Hendrycks et al., 2020
- HumanEval: Chen et al., 2021
- BBH: Suzgun et al., 2022
- And others... (see individual benchmark papers)

## License

This framework is MIT licensed. Individual datasets may have their own licenses.
