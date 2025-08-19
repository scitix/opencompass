from opencompass.datasets.scitix import IFEvalDataset, IFEvalEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

ifeval_reader_cfg = dict(
    input_columns=["key", "prompt", "instruction_id_list", "kwargs"],
    output_column=None,  # outputs will be generated at runtime
)
ifeval_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt="{prompt}"),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
ifeval_eval_cfg = dict(
    evaluator=dict(type=IFEvalEvaluator),
    pred_role="BOT",
)

ifeval_datasets = [
    dict(
        abbr="ifeval",
        type=IFEvalDataset,
        path="scitix/ifeval",
        reader_cfg=ifeval_reader_cfg,
        infer_cfg=ifeval_infer_cfg,
        eval_cfg=ifeval_eval_cfg,
    )
]
