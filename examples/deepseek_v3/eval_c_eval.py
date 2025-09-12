from mmengine.config import read_base

from opencompass.models import OpenAISDKStreaming
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.scitix.c_eval_gen import c_eval_datasets

# datasets
for c_eval_dataset in c_eval_datasets:
    few_shot = True
    few_shot_k = 5
    cot = True

    n = 1
    n_repeats = 1
    num_examples = None

    c_eval_dataset["few_shot"] = few_shot
    c_eval_dataset["few_shot_k"] = few_shot_k
    c_eval_dataset["cot"] = cot
    c_eval_dataset["n"] = n
    c_eval_dataset["n_repeats"] = n_repeats
    c_eval_dataset["num_examples"] = num_examples

    if few_shot:
        if few_shot_k == -1:
            c_eval_dataset["abbr"] += "-5shot"
        else:
            c_eval_dataset["abbr"] += f"-{few_shot_k}shot"
    if cot:
        c_eval_dataset["abbr"] += "-cot"
    if n > 1:
        c_eval_dataset["abbr"] += f"-n{n}"
    if n_repeats > 1:
        c_eval_dataset["abbr"] += f"-r{n_repeats}"
    if num_examples is not None:
        c_eval_dataset["abbr"] += f"-test{num_examples}"

datasets = [*c_eval_datasets]

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

work_dir = "./outputs/c-eval"
