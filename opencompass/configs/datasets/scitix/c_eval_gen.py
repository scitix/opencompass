from opencompass.datasets.scitix import CEvalDataset, CEvalEvaluator
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

c_eval_reader_cfg = dict(
    input_columns=["dialog"],  # use only one column for chat-based custom inputs
    output_column="answer",
)
c_eval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="",
    ),  # empty template as a placeholder for chat-based custom inputs
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ChatInferencer),
)
c_eval_cfg = dict(
    evaluator=dict(type=CEvalEvaluator),
    pred_role="BOT",
)

c_eval_datasets = [
    dict(
        abbr="c-eval",
        type=CEvalDataset,
        path="scitix/c-eval",
        reader_cfg=c_eval_reader_cfg,
        infer_cfg=c_eval_infer_cfg,
        eval_cfg=c_eval_cfg,
    )
]
