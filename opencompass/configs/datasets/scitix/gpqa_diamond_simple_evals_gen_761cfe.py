from mmengine.config import read_base

from opencompass.datasets.scitix import (
    GPQADiamondSimpleEvalsDataset,
    GPQADiamondSimpleEvalsEvaluator,
)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

with read_base():
    from opencompass.datasets.scitix.simple_evals import QUERY_TEMPLATE_MULTICHOICE

gpqa_diamond_reader_cfg = dict(
    input_columns=["Question", "A", "B", "C", "D"],
    output_column="Answer",
)
gpqa_diamond_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt=QUERY_TEMPLATE_MULTICHOICE),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
gpqa_diamond_eval_cfg = dict(
    evaluator=dict(type=GPQADiamondSimpleEvalsEvaluator),
    pred_role="BOT",
)

gpqa_diamond_datasets = [
    dict(
        abbr="gpqa-diamond_simple-evals",
        type=GPQADiamondSimpleEvalsDataset,
        path="scitix/gpqa-diamond_simple-evals",
        reader_cfg=gpqa_diamond_reader_cfg,
        infer_cfg=gpqa_diamond_infer_cfg,
        eval_cfg=gpqa_diamond_eval_cfg,
    )
]
