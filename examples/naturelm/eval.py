from mmengine.config import read_base

from opencompass.models import OpenAISDKStreaming
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################
with read_base():
    from opencompass.configs.datasets.scitix.drop_simple_evals_gen import drop_datasets
    from opencompass.configs.datasets.scitix.human_eval_gen import human_eval_datasets
    from opencompass.configs.datasets.scitix.ifeval_gen import ifeval_datasets
    from opencompass.configs.datasets.scitix.math_500_gen_oai_style import (
        math_500_datasets,
    )
    from opencompass.configs.datasets.scitix.mmlu_simple_evals_gen import mmlu_datasets

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################

datasets = [
    *drop_datasets,
    *human_eval_datasets,
    *ifeval_datasets,
    *math_500_datasets,
    *mmlu_datasets,
]

# LLM judge config: using LLM to evaluate predictions
api_meta_template = dict(
    round=[
        dict(role="SYSTEM", api_role="SYSTEM"),
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ]
)

for item in datasets:
    n = item["n"] if "n" in item else 1
    n_repeats = item["n_repeats"] if "n_repeats" in item else 1
    num_examples = None

    item["num_examples"] = num_examples

    if n > 1:
        item["abbr"] += f"-n{n}"
    if n_repeats > 1:
        item["abbr"] += f"-r{n_repeats}"
    if num_examples is not None:
        item["abbr"] += f"-test{num_examples}"

#######################################################################
#                        PART 2  Models List                          #
#######################################################################

mixtral_8x7b = dict(
    abbr="mixtral-8x7b",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        "http://localhost:8000/v1",
    ],
    key="EMPTY",
    path="mixtral-8x7b",
    tokenizer_path="/models/preset/mistralai/Mixtral-8x7B-Instruct-v0.1/",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=8192,
    temperature=0.0,
)
naturelm_8x7b = dict(
    abbr="naturelm-8x7b",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        "http://172.18.145.156:19198/v1",
    ],
    key="EMPTY",
    path="naturelm-8x7b",
    tokenizer_path="/volume/ai-infra/ylsun/NatureLM-8x7B-Inst/",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=8192,
    temperature=0.6,
)

models = [mixtral_8x7b, naturelm_8x7b]

#######################################################################
#                 PART 3  Inference/Evaluation Configuaration         #
#######################################################################

# infer with local runner
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=16),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLInferTask),
    ),
)

# eval with local runner
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=16),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLEvalTask),
    ),
)

#######################################################################
#                      PART 4  Utils Configuaration                   #
#######################################################################

work_dir = "./outputs/naturelm"
