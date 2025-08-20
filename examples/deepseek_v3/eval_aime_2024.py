from mmengine.config import read_base

from opencompass.models import OpenAISDKStreaming
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    # --- openai-style prompt ---
    from opencompass.configs.datasets.scitix.aime_2024_gen_9882d4 import (
        aime_2024_datasets,
    )

    # --- qwen-style prompt ---
    # from opencompass.configs.datasets.scitix.aime_2024_gen_349c4e import (
    #     aime_2024_datasets,
    # )

    # --- deepseek-style prompt ---
    # from opencompass.configs.datasets.scitix.aime_2024_gen_11c4b5 import (
    #     aime_2024_datasets,
    # )

# datasets
for aime_2024_dataset in aime_2024_datasets:
    aime_2024_dataset["n"] = 16
    aime_2024_dataset["n_repeats"] = 16
    # aime_2024_dataset["num_examples"] = 10

datasets = [*aime_2024_datasets]

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

models = [qwen2_5_72b_instruct]

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

work_dir = "./outputs/aime-2024"
