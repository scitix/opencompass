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
    from opencompass.configs.datasets.scitix.c_eval_gen import c_eval_datasets
    from opencompass.configs.datasets.scitix.c_simpleqa_gen_llmjudge import (
        c_simpleqa_datasets,
    )
    from opencompass.configs.datasets.scitix.cluewsc_gen import cluewsc_datasets
    from opencompass.configs.datasets.scitix.cnmo_2024_gen_oai_style import (
        cnmo_2024_datasets,
    )
    from opencompass.configs.datasets.scitix.compass_academic_2508_mmlu_pro_gen import (
        mmlu_pro_datasets,
    )
    from opencompass.configs.datasets.scitix.drop_simple_evals_gen import drop_datasets
    from opencompass.configs.datasets.scitix.frames_gen_llmjudge import frames_datasets
    from opencompass.configs.datasets.scitix.gpqa_diamond_simple_evals_gen import (
        gpqa_diamond_datasets,
    )
    from opencompass.configs.datasets.scitix.human_eval_gen import human_eval_datasets
    from opencompass.configs.datasets.scitix.ifeval_gen import ifeval_datasets
    from opencompass.configs.datasets.scitix.lcb_v6_gen import lcb_datasets
    from opencompass.configs.datasets.scitix.longbench_v2_0shot_gen import (
        longbench_v2_datasets,
    )
    from opencompass.configs.datasets.scitix.math_500_gen_oai_style import (
        math_500_datasets,
    )
    from opencompass.configs.datasets.scitix.mmlu_redux_zero_eval_gen import (
        mmlu_redux_datasets,
    )
    from opencompass.configs.datasets.scitix.mmlu_simple_evals_gen import mmlu_datasets
    from opencompass.configs.datasets.scitix.simpleqa_simple_evals_gen_llmjudge import (
        simpleqa_datasets,
    )


#######################################################################
#                          PART 1  Datasets List                      #
#######################################################################

for aime_2024_dataset in aime_2024_datasets:
    aime_2024_dataset["n"] = 16
    aime_2024_dataset["n_repeats"] = 1
    aime_2024_dataset["num_examples"] = None

for aime_2025_dataset in aime_2025_datasets:
    aime_2025_dataset["n"] = 16
    aime_2025_dataset["n_repeats"] = 1
    aime_2025_dataset["num_examples"] = None

for c_eval_dataset in c_eval_datasets:
    c_eval_dataset["few_shot"] = True
    c_eval_dataset["few_shot_k"] = 5
    c_eval_dataset["cot"] = True
    c_eval_dataset["n"] = 1
    c_eval_dataset["n_repeats"] = 1
    c_eval_dataset["num_examples"] = None

for c_simpleqa_dataset in c_simpleqa_datasets:
    c_simpleqa_dataset["n"] = 1
    c_simpleqa_dataset["n_repeats"] = 1
    c_simpleqa_dataset["num_examples"] = None
    # for evaluator
    c_simpleqa_dataset["eval_cfg"]["evaluator"]["dataset_cfg"]["n"] = 1
    c_simpleqa_dataset["eval_cfg"]["evaluator"]["dataset_cfg"]["n_repeats"] = 1
    c_simpleqa_dataset["eval_cfg"]["evaluator"]["dataset_cfg"]["num_examples"] = None

for cluewsc_dataset in cluewsc_datasets:
    cluewsc_dataset["n"] = 1
    cluewsc_dataset["n_repeats"] = 1
    cluewsc_dataset["num_examples"] = None

for cnmo_2024_dataset in cnmo_2024_datasets:
    cnmo_2024_dataset["lang"] = "cn"
    cnmo_2024_dataset["n"] = 16
    cnmo_2024_dataset["n_repeats"] = 1
    cnmo_2024_dataset["num_examples"] = None

for mmlu_pro_dataset in mmlu_pro_datasets:
    mmlu_pro_dataset["n"] = 1
    mmlu_pro_dataset["n_repeats"] = 1
    mmlu_pro_dataset["num_examples"] = None

for drop_dataset in drop_datasets:
    drop_dataset["n"] = 1
    drop_dataset["n_repeats"] = 1
    drop_dataset["num_examples"] = None

for frames_dataset in frames_datasets:
    frames_dataset["n"] = 1
    frames_dataset["n_repeats"] = 1
    frames_dataset["num_examples"] = None
    # for evaluator
    frames_dataset["eval_cfg"]["evaluator"]["dataset_cfg"]["n"] = 1
    frames_dataset["eval_cfg"]["evaluator"]["dataset_cfg"]["n_repeats"] = 1
    frames_dataset["eval_cfg"]["evaluator"]["dataset_cfg"]["num_examples"] = None

for gpqa_diamond_dataset in gpqa_diamond_datasets:
    gpqa_diamond_dataset["n"] = 1
    gpqa_diamond_dataset["n_repeats"] = 4
    gpqa_diamond_dataset["num_examples"] = None

for human_eval_dataset in human_eval_datasets:
    human_eval_dataset["n"] = 1
    human_eval_dataset["n_repeats"] = 1
    human_eval_dataset["num_examples"] = None
    # for evalutor
    # repeat k times to get pass@k
    human_eval_dataset["eval_cfg"]["evaluator"]["k"] = human_eval_dataset["n_repeats"]

for ifeval_dataset in ifeval_datasets:
    ifeval_dataset["n"] = 1
    ifeval_dataset["n_repeats"] = 1
    ifeval_dataset["num_examples"] = None

for lcb_dataset in lcb_datasets:
    lcb_dataset["cot"] = False
    lcb_dataset["n"] = 1
    lcb_dataset["n_repeats"] = 1
    lcb_dataset["num_examples"] = None

for longbench_v2_dataset in longbench_v2_datasets:
    longbench_v2_dataset["n"] = 1
    longbench_v2_dataset["n_repeats"] = 1
    longbench_v2_dataset["num_examples"] = None

for math_500_dataset in math_500_datasets:
    math_500_dataset["n"] = 1
    math_500_dataset["n_repeats"] = 1
    math_500_dataset["num_examples"] = None

for mmlu_redux_dataset in mmlu_redux_datasets:
    mmlu_redux_dataset["n"] = 1
    mmlu_redux_dataset["n_repeats"] = 1
    mmlu_redux_dataset["num_examples"] = None

for mmlu_dataset in mmlu_datasets:
    mmlu_dataset["n"] = 1
    mmlu_dataset["n_repeats"] = 1
    mmlu_dataset["num_examples"] = None

for simpleqa_dataset in simpleqa_datasets:
    simpleqa_dataset["n"] = 1
    simpleqa_dataset["n_repeats"] = 1
    simpleqa_dataset["num_examples"] = None
    # for evaluator
    simpleqa_dataset["eval_cfg"]["evaluator"]["dataset_cfg"]["n"] = 1
    simpleqa_dataset["eval_cfg"]["evaluator"]["dataset_cfg"]["n_repeats"] = 1
    simpleqa_dataset["eval_cfg"]["evaluator"]["dataset_cfg"]["num_examples"] = None

datasets = [
    *aime_2024_datasets,
    # *aime_2025_datasets,
    *c_eval_datasets,
    *c_simpleqa_datasets,
    *cluewsc_datasets,
    *cnmo_2024_datasets,
    # *mmlu_pro_datasets,
    *drop_datasets,
    *frames_datasets,
    # *gpqa_diamond_datasets,
    *human_eval_datasets,
    # *ifeval_datasets,
    # *lcb_datasets,
    *longbench_v2_datasets,
    *math_500_datasets,
    *mmlu_redux_datasets,
    *mmlu_datasets,
    *simpleqa_datasets,
]

for item in datasets:
    n = item.get("n", 1)
    n_repeats = item.get("n_repeats", 1)
    num_examples = item.get("num_examples", None)

    # debug
    # item["n"] = 1
    # item["n_repeats"] = 1
    # item["num_examples"] = 1
    # if "dataset_cfg" in item["eval_cfg"]["evaluator"]:
    #     item["eval_cfg"]["evaluator"]["dataset_cfg"]["n"] = 1
    #     item["eval_cfg"]["evaluator"]["dataset_cfg"]["n_repeats"] = 1
    #     item["eval_cfg"]["evaluator"]["dataset_cfg"]["num_examples"] = 1

    if n > 1:
        item["abbr"] += f"-n{n}"
        if "dataset_cfg" in item["eval_cfg"]["evaluator"]:
            item["eval_cfg"]["evaluator"]["dataset_cfg"]["abbr"] += f"-n{n}"
    if n_repeats > 1:
        item["abbr"] += f"-r{n_repeats}"
        if "dataset_cfg" in item["eval_cfg"]["evaluator"]:
            item["eval_cfg"]["evaluator"]["dataset_cfg"]["abbr"] += f"-r{n_repeats}"
    if num_examples is not None:
        item["abbr"] += f"-test{num_examples}"
        if "dataset_cfg" in item["eval_cfg"]["evaluator"]:
            item["eval_cfg"]["evaluator"]["dataset_cfg"]["abbr"] += (
                f"-test{num_examples}"
            )

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
        # "http://172.16.239.229:30000/v1",
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

work_dir = "./outputs/deepseek-v3.1"
