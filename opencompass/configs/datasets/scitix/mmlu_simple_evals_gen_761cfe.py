from mmengine.config import read_base

from opencompass.datasets.scitix import MMLUSimpleEvalsDataset, MMLUSimpleEvalsEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

with read_base():
    from opencompass.datasets.scitix.simple_evals import QUERY_TEMPLATE_MULTICHOICE

mmlu_reader_cfg = dict(
    input_columns=["Question", "A", "B", "C", "D"],
    output_column="Answer",
)
mmlu_infer_cfg = dict(
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
mmlu_eval_cfg = dict(
    evaluator=dict(type=MMLUSimpleEvalsEvaluator),
    pred_role="BOT",
)

mmlu_datasets = [
    dict(
        abbr="mmlu_simple-evals",
        type=MMLUSimpleEvalsDataset,
        path="scitix/mmlu_simple-evals",
        reader_cfg=mmlu_reader_cfg,
        infer_cfg=mmlu_infer_cfg,
        eval_cfg=mmlu_eval_cfg,
    )
]
