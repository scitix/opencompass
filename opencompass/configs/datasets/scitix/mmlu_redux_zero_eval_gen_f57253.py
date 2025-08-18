from mmengine.config import read_base

from opencompass.datasets.scitix import (
    MMLUReduxZeroEvalDataset,
    MMLUReduxZeroEvalEvaluator,
    MMLUReduxZeroEvalPromptTemplate,
)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_retriever import ZeroRetriever

with read_base():
    from opencompass.datasets.scitix.zero_eval import MCQA

mmlu_redux_reader_cfg = dict(
    input_columns=["question", "choices"],
    output_column="correct_answer",
)
mmlu_redux_infer_cfg = dict(
    prompt_template=dict(
        type=MMLUReduxZeroEvalPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt=MCQA),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
mmlu_redux_eval_cfg = dict(
    evaluator=dict(type=MMLUReduxZeroEvalEvaluator),
    pred_role="BOT",
)

mmlu_redux_datasets = [
    dict(
        abbr="mmlu-redux_zero-eval",
        type=MMLUReduxZeroEvalDataset,
        path="scitix/mmlu-redux_zero-eval",
        reader_cfg=mmlu_redux_reader_cfg,
        infer_cfg=mmlu_redux_infer_cfg,
        eval_cfg=mmlu_redux_eval_cfg,
    )
]
