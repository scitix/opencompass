from mmengine.config import read_base

from opencompass.models import OpenAISDKStreaming
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    # from opencompass.configs.datasets.scitix.longbench_v2_0shot_cot_gen import (
    #     longbench_v2_datasets,
    # )
    # from opencompass.configs.datasets.scitix.longbench_v2_0shot_no_context_gen import (
    #     longbench_v2_datasets,
    # )
    from opencompass.configs.datasets.scitix.longbench_v2_0shot_gen import (
        longbench_v2_datasets,
    )

# datasets
for longbench_v2_dataset in longbench_v2_datasets:
    n = 1
    n_repeats = 1
    num_examples = None

    longbench_v2_dataset["n"] = n
    longbench_v2_dataset["n_repeats"] = n_repeats
    longbench_v2_dataset["num_examples"] = num_examples

    if n > 1:
        longbench_v2_dataset["abbr"] += f"-n{n}"
    if n_repeats > 1:
        longbench_v2_dataset["abbr"] += f"-r{n_repeats}"
    if num_examples is not None:
        longbench_v2_dataset["abbr"] += f"-test{num_examples}"

datasets = [*longbench_v2_datasets]

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
    mode="mid",  # truncation
)
qwen2_5_72b_instruct = dict(
    **qwen2_5_72b_instruct_api,
    query_per_second=32,
    batch_size=128,
    temperature=0.0,
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

work_dir = "./outputs/longbench-v2"
