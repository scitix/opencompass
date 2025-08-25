from opencompass.datasets.scitix import AIME2024Dataset, AIME2024Evaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# adapted from https://github.com/QwenLM/Qwen2.5-Math/blob/a45202bd16f1ec06f433442dc1152d0074773465/evaluation/utils.py
aime_2024_reader_cfg = dict(
    input_columns=["problem"],
    output_column="answer",
)
aime_2024_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role="SYSTEM",
                    fallback_role="HUMAN",
                    prompt="Please reason step by step, and put your final answer within \\boxed{}.",
                ),
            ],
            round=[
                dict(role="HUMAN", prompt="{problem}"),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
aime_2024_eval_cfg = dict(
    evaluator=dict(type=AIME2024Evaluator),
    pred_role="BOT",
)

aime_2024_datasets = [
    dict(
        abbr="aime-2024",
        type=AIME2024Dataset,
        path="scitix/aime-2024",
        reader_cfg=aime_2024_reader_cfg,
        infer_cfg=aime_2024_infer_cfg,
        eval_cfg=aime_2024_eval_cfg,
    )
]
