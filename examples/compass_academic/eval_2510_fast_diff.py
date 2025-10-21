from mmengine.config import read_base

from opencompass.models import OpenAISDKStreaming
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

#######################################################################
#                          PART 0  Essential Configs                  #
#######################################################################
with read_base():
    from opencompass.configs.datasets.scitix.aime_2024_gen_oai_style import (
        aime_2024_datasets,
    )
    from opencompass.configs.datasets.scitix.aime_2025_gen_oai_style import (
        aime_2025_datasets,
    )
    from opencompass.configs.datasets.scitix.lcb_v6_gen import (
        lcb_datasets,
    )

#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################

for aime_2024_dataset in aime_2024_datasets:
    aime_2024_dataset["n"] = 64
    aime_2024_dataset["n_repeats"] = 1
    aime_2024_dataset["num_examples"] = None

for aime_2025_dataset in aime_2025_datasets:
    aime_2025_dataset["n"] = 64
    aime_2025_dataset["n_repeats"] = 1
    aime_2025_dataset["num_examples"] = None

for lcb_dataset in lcb_datasets:
    lcb_dataset["abbr"] = "lcb-code-generation-lite_2408_2505"
    lcb_dataset["start_date"] = "2024-08-01"
    lcb_dataset["end_date"] = "2025-05-31"
    lcb_dataset["cot"] = False

    lcb_dataset["n"] = 10
    lcb_dataset["n_repeats"] = 1
    lcb_dataset["num_examples"] = None

datasets = [
    *aime_2024_datasets,
    *aime_2025_datasets,
    *lcb_datasets,
]

for item in datasets:
    n = item.get("n", 1)
    n_repeats = item.get("n_repeats", 1)
    num_examples = item.get("num_examples", None)

    if n > 1:
        item["abbr"] += f"-n{n}"
    if n_repeats > 1:
        item["abbr"] += f"-r{n_repeats}"
    if num_examples is not None:
        item["abbr"] += f"-test{num_examples}"

#######################################################################
#                        PART 2  Models List                          #
#######################################################################

api_meta_template = dict(
    round=[
        dict(role="SYSTEM", api_role="SYSTEM"),
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ]
)

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
    max_out_len=8192,
    temperature=0.6,
    mode="mid",  # truncation
    retry=10,
)
deepseek_v3_1_thinking = dict(
    abbr="deepseek-v3.1-thinking",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        "http://eval-deepseek-v3-1.t-ai-infra-ylsun.svc/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-1",
    tokenizer_path="/models/preset/deepseek-ai/DeepSeek-V3.1/v1.0/",
    extra_body=dict(
        chat_template_kwargs=dict(thinking=True),
    ),
    query_per_second=32,
    batch_size=128,
    max_seq_len=131072,
    max_out_len=32768,
    temperature=0.6,
    mode="mid",  # truncation
    retry=10,
)
deepseek_v3_1_terminus = dict(
    abbr="deepseek-v3.1-terminus",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        "http://eval-deepseek-v3-1-terminus.t-ai-infra-ylsun.svc/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-1-terminus",
    tokenizer_path="/models/preset/deepseek-ai/DeepSeek-V3.1-Terminus/v1.0/",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=8192,
    temperature=0.6,
    mode="mid",  # truncation
    retry=10,
)
deepseek_v3_1_terminus_thinking = dict(
    abbr="deepseek-v3.1-terminus-thinking",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        "http://eval-deepseek-v3-1-terminus.t-ai-infra-ylsun.svc/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-1-terminus",
    tokenizer_path="/models/preset/deepseek-ai/DeepSeek-V3.1-Terminus/v1.0/",
    extra_body=dict(
        chat_template_kwargs=dict(thinking=True),
    ),
    query_per_second=32,
    batch_size=128,
    max_seq_len=131072,
    max_out_len=32768,
    temperature=0.6,
    mode="mid",  # truncation
    retry=10,
)
deepseek_v3_2_exp_thinking = dict(
    abbr="deepseek-v3.2-exp-thinking",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        "http://eval-deepseek-v3-2-exp-sgl.t-ai-infra-ylsun.svc/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-2-exp",
    tokenizer_path="/models/preset/deepseek-ai/DeepSeek-V3.2-Exp/v1.0/",
    extra_body=dict(
        chat_template_kwargs=dict(thinking=True),
    ),
    query_per_second=32,
    batch_size=128,
    max_seq_len=131072,
    max_out_len=32768,
    temperature=0.6,
    mode="mid",  # truncation
    retry=10,
)
deepseek_v3_0324 = dict(
    abbr="deepseek-v3-0324",
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
exp91 = dict(
    abbr="deepseek-v3-exp91",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        # "http://eval-deepseek-v3-exp91.t-ai-infra-ylsun.svc/v1",
        "http://172.16.161.31:30000/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-exp91",
    tokenizer_path="/everything/models/deepseek-ai/DeepSeek-V3.1-Base",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=8192,
    temperature=0.6,
    mode="mid",  # truncation
    retry=10,
)
exp93 = dict(
    abbr="deepseek-v3-exp93",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        # "http://eval-deepseek-v3-exp93.t-ai-infra-ylsun.svc/v1",
        "http://172.16.82.68:30000/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-exp93",
    tokenizer_path="/everything/models/deepseek-ai/DeepSeek-V3.1-Base",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=8192,
    temperature=0.6,
    mode="mid",  # truncation
    retry=10,
)
exp103 = dict(
    abbr="deepseek-v3-exp103",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        # "http://eval-deepseek-v3-exp103.t-ai-infra-ylsun.svc/v1",
        "http://172.16.148.82:30000/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-exp103",
    tokenizer_path="/everything/models/deepseek-ai/DeepSeek-V3.1-Base",
    query_per_second=32,
    batch_size=128,
    # max_seq_len=32768,
    max_out_len=8192,
    temperature=0.6,
    mode="mid",  # truncation
    retry=10,
)
exp108 = dict(
    abbr="deepseek-v3-exp108",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        # "http://eval-deepseek-v3-exp108.t-ai-infra-ylsun.svc/v1",
        "http://172.16.45.236:30000/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-exp108",
    tokenizer_path="/everything/models/deepseek-ai/DeepSeek-V3.1-Base",
    query_per_second=32,
    batch_size=128,
    # max_seq_len=32768,
    max_out_len=8192,
    temperature=0.6,
    mode="mid",  # truncation
    retry=10,
)
exp110 = dict(
    abbr="deepseek-v3-exp110",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        # "http://eval-deepseek-v3-exp110.t-ai-infra-ylsun.svc/v1",
        "http://172.16.177.180:30000/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-exp110",
    tokenizer_path="/everything/models/deepseek-ai/DeepSeek-V3.1-Base",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=8192,
    temperature=0.6,
    mode="mid",  # truncation
    retry=10,
)
exp111 = dict(
    abbr="deepseek-v3-exp111",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        # "http://eval-deepseek-v3-exp111.t-ai-infra-ylsun.svc/v1",
        "http://172.16.65.148:30000/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-exp111",
    tokenizer_path="/everything/models/deepseek-ai/DeepSeek-V3.1-Base",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=8192,
    temperature=0.6,
    mode="mid",  # truncation
    retry=10,
)
exp260 = dict(
    abbr="deepseek-v3-exp260",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        # "http://eval-deepseek-v3-exp260.t-ai-infra-ylsun.svc/v1",
        "http://172.16.189.143:30000/v1",
    ],
    key="EMPTY",
    path="deepseek-v3-exp260",
    tokenizer_path="/everything/models/deepseek-ai/DeepSeek-V3.1-Base",
    query_per_second=32,
    batch_size=128,
    max_seq_len=32768,
    max_out_len=8192,
    temperature=0.6,
    mode="mid",  # truncation
    retry=10,
)
exp262 = dict(
    abbr="deepseek-v3-exp262",
    type=OpenAISDKStreaming,
    meta_template=api_meta_template,
    openai_api_base=[
        # "http://eval-deepseek-v3-exp262.t-ai-infra-ylsun.svc/v1",
        "http://172.16.63.126:30000/v1",
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

models = [
    deepseek_v3_1,
    deepseek_v3_1_thinking,
    deepseek_v3_1_terminus,
    deepseek_v3_1_terminus_thinking,
    deepseek_v3_2_exp_thinking,
    deepseek_v3_0324,
    exp91,
    exp93,
    exp103,
    exp108,
    exp110,
    exp111,
    exp260,
    exp262,
]

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

work_dir = "./outputs/compass-academic-202510-fast-diff"
