from mmengine.config import read_base

from opencompass.models import OpenAISDKStreaming
from opencompass.partitioners import NumWorkerPartitioner
from opencompass.partitioners.sub_naive import SubjectiveNaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask
from opencompass.tasks.subjective_eval import SubjectiveEvalTask

with read_base():
    from opencompass.configs.datasets.scitix.compass_bench_v1_3_instruct_gen_llmjudge import (
        compass_bench_v1_3_instruct_datasets,
    )

# models
qwen2_5_72b_instruct_api = dict(
    abbr="Qwen2.5-72B-Instruct",
    type=OpenAISDKStreaming,
    openai_api_base=[
        "http://localhost:8000/v1",
    ],
    key="EMPTY",
    path="/models/preset/Qwen/Qwen2.5-72B-Instruct/v1.0/",
    tokenizer_path="/models/preset/Qwen/Qwen2.5-72B-Instruct/v1.0/",
    max_seq_len=32768,
)
qwen2_5_72b_instruct = dict(
    **qwen2_5_72b_instruct_api,
    query_per_second=32,
    batch_size=128,
    temperature=0.7,
    max_out_len=8192,
)

qwen3_235b_a22b_instruct_2507_api = dict(
    abbr="Qwen3-235B-A22B-Instruct-2507",
    type=OpenAISDKStreaming,
    openai_api_base=[
        "https://console.siflow.cn/model-api",
    ],
    key="",
    path="simaas-qwen3-235b-a22b-instruct-2507-v1",
    tokenizer_path="/models/preset/Qwen/Qwen3-235B-A22B-Instruct-2507/v1.0/",
    max_seq_len=32768,
)
qwen3_235b_a22b_instruct_2507 = dict(
    **qwen3_235b_a22b_instruct_2507_api,
    query_per_second=32,
    batch_size=128,
    temperature=0.7,
    max_out_len=8192,
)

deepseek_v3_0324_api = dict(
    abbr="DeepSeek-V3-0324",
    type=OpenAISDKStreaming,
    openai_api_base=[
        "https://console.siflow.cn/model-api",
    ],
    key="",
    path="simaas-deepseek-v3-v1",
    tokenizer_path="/models/preset/deepseek-ai/DeepSeek-V3-0324/v1.0/",
    max_seq_len=32768,
)
deepseek_v3_0324 = dict(
    **deepseek_v3_0324_api,
    query_per_second=32,
    batch_size=128,
    temperature=0.7,
    max_out_len=8192,
)

deepseek_v3_1_api = dict(
    abbr="DeepSeek-V3.1",
    type=OpenAISDKStreaming,
    openai_api_base=[
        "http://localhost:8000/v1",
    ],
    key="EMPTY",
    path="/models/preset/deepseek-ai/DeepSeek-V3.1/v1.0/",
    tokenizer_path="/models/preset/deepseek-ai/DeepSeek-V3.1/v1.0/",
    max_seq_len=32768,
)
deepseek_v3_1 = dict(
    **deepseek_v3_1_api,
    query_per_second=32,
    batch_size=128,
    temperature=0.7,
    max_out_len=8192,
)

gemini2_5_flash_api = dict(
    abbr="Gemini-2.5-Flash",
    type=OpenAISDKStreaming,
    openai_api_base=[
        "https://console.siflow.cn/model-api",
    ],
    key="",
    path="gemini-2.5-flash",
    tokenizer_path="/models/preset/Qwen/Qwen2.5-72B-Instruct/v1.0/",
    max_seq_len=32768,
)
gemini2_5_flash = dict(
    **gemini2_5_flash_api,
    query_per_second=32,
    batch_size=128,
    temperature=0.7,
    max_out_len=8192,
)

gemini2_5_pro_api = dict(
    abbr="Gemini-2.5-Pro",
    type=OpenAISDKStreaming,
    openai_api_base=[
        "https://console.siflow.cn/model-api",
    ],
    key="",
    path="gemini-2.5-pro",
    tokenizer_path="/models/preset/Qwen/Qwen2.5-72B-Instruct/v1.0/",
    max_seq_len=32768,
)
gemini2_5_pro = dict(
    **gemini2_5_pro_api,
    query_per_second=32,
    batch_size=128,
    temperature=0.7,
    max_out_len=8192,
)

claude_sonnet_4_api = dict(
    abbr="Claude-Sonnet-4",
    type=OpenAISDKStreaming,
    openai_api_base=[
        "https://console.siflow.cn/model-api",
    ],
    key="",
    path="claude-sonnet-4-20250514",
    tokenizer_path="/models/preset/Qwen/Qwen2.5-72B-Instruct/v1.0/",
    max_seq_len=32768,
)
claude_sonnet_4 = dict(
    **claude_sonnet_4_api,
    query_per_second=16,
    batch_size=128,
    temperature=0.7,
    max_out_len=8192,
)

gpt_4_1_api = dict(
    abbr="GPT-4.1",
    type=OpenAISDKStreaming,
    openai_api_base=[
        "https://console.siflow.cn/model-api",
    ],
    key="",
    path="gpt-4.1-2025-04-14",
    tokenizer_path="/models/preset/Qwen/Qwen2.5-72B-Instruct/v1.0/",
    max_seq_len=32768,
)
gpt_4_1 = dict(
    **gpt_4_1_api,
    query_per_second=32,
    batch_size=128,
    temperature=0.7,
    max_out_len=8192,
)

# models for inference
base_models = [deepseek_v3_0324, deepseek_v3_1, gemini2_5_flash]
compare_models = [qwen2_5_72b_instruct]
models = base_models + compare_models
# models for judging
judge_models = [qwen3_235b_a22b_instruct_2507, gemini2_5_pro, claude_sonnet_4, gpt_4_1]

# datasets
for compass_bench_v1_3_instruct_dataset in compass_bench_v1_3_instruct_datasets:
    n = 1
    n_repeats = 1
    num_examples = None

    compass_bench_v1_3_instruct_dataset["n"] = n
    compass_bench_v1_3_instruct_dataset["n_repeats"] = n_repeats
    compass_bench_v1_3_instruct_dataset["num_examples"] = num_examples

    if n > 1:
        compass_bench_v1_3_instruct_dataset["abbr"] += f"-n{n}"
    if n_repeats > 1:
        compass_bench_v1_3_instruct_dataset["abbr"] += f"-r{n_repeats}"
    if num_examples is not None:
        compass_bench_v1_3_instruct_dataset["abbr"] += f"-test{num_examples}"

    # for llm judge partitioners
    compass_bench_v1_3_instruct_dataset["mode"] = "allperm"

datasets = [*compass_bench_v1_3_instruct_datasets]

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=4,
    ),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(
        # MODE m2n    : product of `base_models` and `compare_models` (remove duplicate pairs and self-pair)
        # MODE allpair: combination of `models`
        # MODE allperm: permutations of `models`
        type=SubjectiveNaivePartitioner,
        base_models=base_models,
        compare_models=compare_models,
        models=models,
        judge_models=judge_models,
    ),
    runner=dict(
        type=LocalRunner,
        task=dict(type=SubjectiveEvalTask),
    ),
)

work_dir = "./outputs/compass-bench-v1.3-instruct"
