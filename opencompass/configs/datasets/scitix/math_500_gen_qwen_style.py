from opencompass.datasets.scitix import MATH500Dataset, MATH500Evaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# adapted from https://github.com/QwenLM/Qwen2.5-Math/blob/a45202bd16f1ec06f433442dc1152d0074773465/evaluation/utils.py
math_500_reader_cfg = dict(
    input_columns=["problem"],
    output_column="answer",
)
math_500_infer_cfg = dict(
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
math_500_eval_cfg = dict(
    evaluator=dict(type=MATH500Evaluator),
    pred_role="BOT",
)

math_500_datasets = [
    dict(
        abbr="math-500",
        type=MATH500Dataset,
        path="scitix/math-500",
        reader_cfg=math_500_reader_cfg,
        infer_cfg=math_500_infer_cfg,
        eval_cfg=math_500_eval_cfg,
    )
]
