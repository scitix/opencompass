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
    from opencompass.configs.datasets.scitix.compass_academic_2508_gpqa_diamond_gen import (
        gpqa_diamond_datasets,
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
    *lcb_datasets,
    *mmlu_pro_datasets,
    *ifeval_datasets,
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

deepseek_v3_1 = dict(
    abbr="deepseek-v3.1",
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
    max_out_len=32000,
    temperature=0.6,
)
exp91 = dict(
    abbr="deepseek-v3-exp91",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        "http://eval-deepseek-v3-exp91.t-ai-infra-ylsun.svc/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-exp91",
    tokenizer_path="/everything/models/deepseek-ai/DeepSeek-V3.1-Base",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=32000,
    temperature=0.6,
)
exp93 = dict(
    abbr="deepseek-v3-exp93",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        "http://eval-deepseek-v3-exp93.t-ai-infra-ylsun.svc/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-exp93",
    tokenizer_path="/everything/models/deepseek-ai/DeepSeek-V3.1-Base",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=32000,
    temperature=0.6,
)
exp103 = dict(
    abbr="deepseek-v3-exp103",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        "http://eval-deepseek-v3-exp103.t-ai-infra-ylsun.svc/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-exp103",
    tokenizer_path="/everything/models/deepseek-ai/DeepSeek-V3.1-Base",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=32000,
    temperature=0.6,
)
exp108 = dict(
    abbr="deepseek-v3-exp108",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        "http://eval-deepseek-v3-exp108.t-ai-infra-ylsun.svc/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-exp108",
    tokenizer_path="/everything/models/deepseek-ai/DeepSeek-V3.1-Base",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=32000,
    temperature=0.6,
)
exp110 = dict(
    abbr="deepseek-v3-exp110",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        "http://eval-deepseek-v3-exp110.t-ai-infra-ylsun.svc/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-exp110",
    tokenizer_path="/everything/models/deepseek-ai/DeepSeek-V3.1-Base",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=32000,
    temperature=0.6,
)
exp111 = dict(
    abbr="deepseek-v3-exp111",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        "http://eval-deepseek-v3-exp111.t-ai-infra-ylsun.svc/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-exp111",
    tokenizer_path="/everything/models/deepseek-ai/DeepSeek-V3.1-Base",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=32000,
    temperature=0.6,
)

models = [deepseek_v3_1, exp91, exp93, exp103, exp108, exp110, exp111]

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

work_dir = "./outputs/compass-academic-202510-simple"
