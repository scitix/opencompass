from opencompass.datasets.scitix import CNMO2024Dataset, CNMO2024Evaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# adapted from https://github.com/QwenLM/Qwen2.5-Math/blob/a45202bd16f1ec06f433442dc1152d0074773465/evaluation/utils.py
cnmo_2024_reader_cfg = dict(
    input_columns=["question"],
    output_column="answer",
)
cnmo_2024_infer_cfg = dict(
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
cnmo_2024_eval_cfg = dict(
    evaluator=dict(type=CNMO2024Evaluator),
    pred_role="BOT",
)

cnmo_2024_datasets = [
    dict(
        abbr="cnmo-2024",
        type=CNMO2024Dataset,
        path="scitix/cnmo-2024",
        reader_cfg=cnmo_2024_reader_cfg,
        infer_cfg=cnmo_2024_infer_cfg,
        eval_cfg=cnmo_2024_eval_cfg,
    )
]
