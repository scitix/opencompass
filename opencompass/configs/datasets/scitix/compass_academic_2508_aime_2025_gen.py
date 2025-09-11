from opencompass.datasets.scitix import AIME2025Dataset, AIME2025Evaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.utils.text_postprocessors import match_answer_pattern

aime2025_reader_cfg = dict(
    input_columns=["question"],
    output_column="answer",
)
aime2025_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt="{question}\nRemember to put your final answer within \\boxed{}.",
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
aime2025_eval_cfg = dict(
    evaluator=dict(
        type=AIME2025Evaluator,
        pred_postprocessor=dict(
            type=match_answer_pattern,
            answer_pattern=r"\\boxed\{(.*?)\}",
        ),
    ),
    pred_role="BOT",
)

aime2025_datasets = [
    dict(
        abbr="aime-2025",
        type=AIME2025Dataset,
        path="scitix/aime-2025",
        reader_cfg=aime2025_reader_cfg,
        infer_cfg=aime2025_infer_cfg,
        eval_cfg=aime2025_eval_cfg,
        n=32,
    )
]
