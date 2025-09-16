from mmengine.config import read_base

from opencompass.models import OpenAISDKStreaming
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from opencompass.configs.datasets.scitix.frames_gen_llmjudge import frames_datasets

# datasets
for frames_dataset in frames_datasets:
    n = 1
    n_repeats = 1
    num_examples = None

    frames_dataset["n"] = n
    frames_dataset["n_repeats"] = n_repeats
    frames_dataset["num_examples"] = num_examples
    # for evaluator
    frames_dataset["eval_cfg"]["evaluator"]["dataset_cfg"]["n"] = 1
    frames_dataset["eval_cfg"]["evaluator"]["dataset_cfg"]["n_repeats"] = n_repeats
    frames_dataset["eval_cfg"]["evaluator"]["dataset_cfg"]["num_examples"] = (
        num_examples
    )

    if n > 1:
        frames_dataset["abbr"] += f"-n{n}"
        frames_dataset["eval_cfg"]["evaluator"]["dataset_cfg"]["abbr"] += f"-n{n}"
    if n_repeats > 1:
        frames_dataset["abbr"] += f"-r{n_repeats}"
        frames_dataset["eval_cfg"]["evaluator"]["dataset_cfg"]["abbr"] += (
            f"-r{n_repeats}"
        )
    if num_examples is not None:
        frames_dataset["abbr"] += f"-test{num_examples}"
        frames_dataset["eval_cfg"]["evaluator"]["dataset_cfg"]["abbr"] += (
            f"-test{num_examples}"
        )

datasets = [*frames_datasets]

# models
api_meta_template = dict(
    round=[
        dict(role="SYSTEM", api_role="SYSTEM"),
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ]
)

qwen2_5_72b_instruct_api = dict(
    abbr="Qwen2.5-72B-Instruct",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
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

work_dir = "./outputs/frames"
