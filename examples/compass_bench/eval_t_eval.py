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
api_meta_template = dict(
    round=[
        dict(role="SYSTEM", api_role="SYSTEM"),
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ]
)

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

qwen2_5_72b_instruct = dict(
    abbr="Qwen2.5-72B-Instruct",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        "https://console.siflow.cn/model-api",
    ],
    key="",
    path="simaas-qwen2-5-72b-instruct-v1",
    tokenizer_path="/models/preset/deepseek-ai/DeepSeek-V3.1/v1.0/",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=8192,
    temperature=0.6,
    mode="mid",  # truncation
    retry=10,
)
deepseek_v3_0324 = dict(
    abbr="DeepSeek-V3-0324",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        "https://console.siflow.cn/model-api",
    ],
    key="",
    path="simaas-deepseek-v3-v1",
    tokenizer_path="/models/preset/deepseek-ai/DeepSeek-V3.1/v1.0/",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=8192,
    temperature=0.6,
    mode="mid",  # truncation
    retry=10,
)
deepseek_v3_1 = dict(
    abbr="Deepseek-V3.1",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        "http://eval-deepseek-v3-1.t-ai-infra-ylsun.svc/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-1",
    tokenizer_path="/models/preset/deepseek-ai/DeepSeek-V3.1/v1.0/",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=8192,
    temperature=0.6,
    mode="mid",  # truncation
    retry=10,
)
exp262 = dict(
    abbr="Exp262",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        "http://eval-deepseek-v3-exp262.t-ai-infra-ylsun.svc/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-exp262",
    tokenizer_path="/everything/models/deepseek-ai/DeepSeek-V3.1-Base",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=8192,
    temperature=0.6,
    mode="mid",  # truncation
    retry=10,
)

models = [qwen2_5_72b_instruct, deepseek_v3_0324, deepseek_v3_1, exp262]

infer = dict(
    partitioner=dict(
        type=NumWorkerPartitioner,
        num_worker=16,
    ),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(
        type=NaivePartitioner,
        n=16,
    ),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLEvalTask),
    ),
)

work_dir = "./outputs/compass-bench-v1.3-t-eval"
