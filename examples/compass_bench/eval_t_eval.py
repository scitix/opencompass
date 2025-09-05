from mmengine.config import read_base

from opencompass.models import OpenAISDKStreaming
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.scitix.compass_bench_v1_3_t_eval_gen import (
        compass_bench_v1_3_t_eval_datasets,
    )

# datasets
for compass_bench_v1_3_t_eval_dataset in compass_bench_v1_3_t_eval_datasets:
    lang = "cn"
    legacy_model = True
    n = 1
    n_repeats = 1
    num_examples = None

    compass_bench_v1_3_t_eval_dataset["lang"] = lang
    compass_bench_v1_3_t_eval_dataset["legacy_model"] = legacy_model
    compass_bench_v1_3_t_eval_dataset["n"] = n
    compass_bench_v1_3_t_eval_dataset["n_repeats"] = n_repeats
    compass_bench_v1_3_t_eval_dataset["num_examples"] = num_examples

    if lang != "cn":
        compass_bench_v1_3_t_eval_dataset["abbr"] += f"-{lang}"
    if legacy_model:
        compass_bench_v1_3_t_eval_dataset["abbr"] += "-legacy"
    if n > 1:
        compass_bench_v1_3_t_eval_dataset["abbr"] += f"-n{n}"
    if n_repeats > 1:
        compass_bench_v1_3_t_eval_dataset["abbr"] += f"-r{n_repeats}"
    if num_examples is not None:
        compass_bench_v1_3_t_eval_dataset["abbr"] += f"-test{num_examples}"

datasets = [*compass_bench_v1_3_t_eval_datasets]

# models
llama2_13b_api = dict(
    abbr="Llama2-13B",
    type=OpenAISDKStreaming,
    openai_api_base=[
        "http://localhost:8000/v1",
    ],
    key="EMPTY",
    path="/models/preset/meta-llama/Llama-2-13b-chat-hf/v1.0/",
    tokenizer_path="/models/preset/meta-llama/Llama-2-13b-chat-hf/v1.0/",
    max_seq_len=8192,
)
llama2_13b = dict(
    **llama2_13b_api,
    query_per_second=32,
    batch_size=128,
    temperature=0.7,
    max_out_len=2048,
)

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

deepseek_v3_0324_api = dict(
    abbr="DeepSeek-V3-0324",
    type=OpenAISDKStreaming,
    openai_api_base=[
        "https://console.siflow.cn/model-api",
    ],
    key="sk-eI0rVe74K9Xq17FJHiFcrOH9fm5I0zdYPsXMRyDRXoSV4vYTSntsHOpyHPGWvWSU",
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

models = [qwen2_5_72b_instruct, deepseek_v3_0324, deepseek_v3_1, gemini2_5_flash]

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
        type=NaivePartitioner,
        n=8,
    ),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLEvalTask),
    ),
)

work_dir = "./outputs/compass-bench-v1.3-t-eval"
