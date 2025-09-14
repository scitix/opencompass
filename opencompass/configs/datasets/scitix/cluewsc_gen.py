from opencompass.datasets.scitix import CLUEWSCDataset, CLUEWSCEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# QUERY_TEMPLATE = """
# {text}
# 此处，“{span2}”是否指代“{span1}”？
# A. 是
# B. 否
# 请只回答 A 或 B。
# 答：
# """.strip()

QUERY_TEMPLATE = """
{text}
此处，“{span2}”是否指代“{span1}”？
A. 是
B. 否
请从“A”，“B”中进行选择。
答：
""".strip()

cluewsc_reader_cfg = dict(
    input_columns=["text", "span1", "span2"],
    output_column="answer",
)
cluewsc_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt=QUERY_TEMPLATE),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
cluewsc_cfg = dict(
    evaluator=dict(type=CLUEWSCEvaluator),
    pred_role="BOT",
)

cluewsc_datasets = [
    dict(
        abbr="cluewsc",
        type=CLUEWSCDataset,
        path="scitix/cluewsc",
        reader_cfg=cluewsc_reader_cfg,
        infer_cfg=cluewsc_infer_cfg,
        eval_cfg=cluewsc_cfg,
    )
]
