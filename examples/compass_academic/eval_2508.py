from mmengine.config import read_base

from opencompass.models import OpenAISDKStreaming
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################
with read_base():
    from opencompass.configs.datasets.scitix.compass_academic_2508_aime_2025_gen import (
        aime2025_datasets,
    )
    from opencompass.configs.datasets.scitix.compass_academic_2508_gpqa_diamond_gen_llmjudge import (
        gpqa_diamond_datasets,
    )
    from opencompass.configs.datasets.scitix.compass_academic_2508_hle_gen_llmjudge import (
        hle_datasets,
    )
    from opencompass.configs.datasets.scitix.compass_academic_2508_lcb_v6_gen import (
        lcb_datasets,
    )
    from opencompass.configs.datasets.scitix.compass_academic_2508_mmlu_pro_gen import (
        mmlu_pro_datasets,
    )
    from opencompass.configs.datasets.scitix.ifeval_gen import ifeval_datasets

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################

datasets = [
    *aime2025_datasets,
    *gpqa_diamond_datasets,
    *hle_datasets,
    *lcb_datasets,
    *mmlu_pro_datasets,
    *ifeval_datasets,
]

# LLM judge config: using LLM to evaluate predictions
judge_cfg = dict(
    abbr="CompassVerifier-32B",
    type=OpenAISDKStreaming,
    openai_api_base=[
        "http://localhost:8000/v1",
    ],
    key="EMPTY",
    path="/volume/ai-infra/CompassVerifier-32B/",
    tokenizer_path="/volume/ai-infra/CompassVerifier-32B/",
    meta_template=dict(
        round=[
            dict(role="HUMAN", api_role="HUMAN"),
            dict(role="BOT", api_role="BOT", generate=True),
        ]
    ),
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=32000,
    temperature=0.0,
    mode="mid",
)

for item in datasets:
    if "judge_cfg" in item["eval_cfg"]["evaluator"]:
        item["eval_cfg"]["evaluator"]["judge_cfg"] = judge_cfg
    if (
        "llm_evaluator" in item["eval_cfg"]["evaluator"].keys()
        and "judge_cfg" in item["eval_cfg"]["evaluator"]["llm_evaluator"]
    ):
        item["eval_cfg"]["evaluator"]["llm_evaluator"]["judge_cfg"] = judge_cfg

    n = item["n"] if "n" in item else 1
    n_repeats = 1
    num_examples = None

    item["n_repeats"] = n_repeats
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

qwen2_5_72b_instruct = dict(
    abbr="Qwen2.5-72B-Instruct",
    type=OpenAISDKStreaming,
    openai_api_base=[
        "http://localhost:8000/v1",
    ],
    key="EMPTY",
    path="/models/preset/Qwen/Qwen2.5-72B-Instruct/v1.0/",
    tokenizer_path="/models/preset/Qwen/Qwen2.5-72B-Instruct/v1.0/",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=32000,
    temperature=0.6,
)
deepseek_v3_0324 = dict(
    abbr="DeepSeek-V3-0324",
    type=OpenAISDKStreaming,
    openai_api_base=[
        "https://console.siflow.cn/model-api",
    ],
    key="",
    path="simaas-deepseek-v3-v1",
    tokenizer_path="/models/preset/deepseek-ai/DeepSeek-V3-0324/v1.0/",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=32000,
    temperature=0.6,
)
deepseek_v3_1 = dict(
    abbr="DeepSeek-V3.1",
    type=OpenAISDKStreaming,
    openai_api_base=[
        "http://localhost:8000/v1",
    ],
    key="EMPTY",
    path="/models/preset/deepseek-ai/DeepSeek-V3.1/v1.0/",
    tokenizer_path="/models/preset/deepseek-ai/DeepSeek-V3.1/v1.0/",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=32000,
    temperature=0.6,
)

models = [qwen2_5_72b_instruct, deepseek_v3_0324, deepseek_v3_1]

#######################################################################
#                 PART 3  Inference/Evaluation Configuaration         #
#######################################################################

# infer with local runner
infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLInferTask),
    ),
)

# eval with local runner
eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLEvalTask),
    ),
)

#######################################################################
#                      PART 4  Utils Configuaration                   #
#######################################################################

work_dir = "./outputs/compass-academic-202508"
