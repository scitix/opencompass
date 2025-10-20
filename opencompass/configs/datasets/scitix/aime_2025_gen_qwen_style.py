from opencompass.datasets.scitix import AIME2025Dataset, AIME2025Evaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# adapted from https://github.com/QwenLM/Qwen2.5-Math/blob/a45202bd16f1ec06f433442dc1152d0074773465/evaluation/utils.py
aime_2025_reader_cfg = dict(
    input_columns=["question"],
    output_column="answer",
)
aime_2025_infer_cfg = dict(
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
                dict(role="HUMAN", prompt="{question}"),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
aime_2025_eval_cfg = dict(
    evaluator=dict(type=AIME2025Evaluator),
    pred_role="BOT",
)

aime_2025_datasets = [
    dict(
        abbr="aime-2025",
        type=AIME2025Dataset,
        path="scitix/aime-2025",
        reader_cfg=aime_2025_reader_cfg,
        infer_cfg=aime_2025_infer_cfg,
        eval_cfg=aime_2025_eval_cfg,
    )
]
