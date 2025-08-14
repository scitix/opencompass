from mmengine.config import read_base

from opencompass.models import OpenAISDKStreaming
from opencompass.partitioners import NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask

with read_base():
    from opencompass.configs.datasets.scitix.mmlu_simple_evals_gen_761cfe import (
        mmlu_datasets,
    )

# datasets
for mmlu_dataset in mmlu_datasets:
    mmlu_dataset["n"] = 1

datasets = [*mmlu_datasets]

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
    query_per_second=64,
    batch_size=128,
    temperature=0.7,
    max_out_len=8192,
)

models = [qwen2_5_72b_instruct]

# infer = dict(
#     partitioner=dict(
#         type=NumWorkerPartitioner,
#         num_worker=8,
#     ),
#     runner=dict(
#         type=LocalRunner,
#         task=dict(type=OpenICLInferTask),
#     ),
# )

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
