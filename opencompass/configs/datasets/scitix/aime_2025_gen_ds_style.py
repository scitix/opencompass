from opencompass.datasets.scitix import AIME2025Dataset, AIME2025Evaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# adapted from https://github.com/deepseek-ai/DeepSeek-Math/blob/b8b0f8ce093d80bf8e9a641e44142f06d092c305/evaluation/run_subset_parallel.py
QUERY_TEMPLATE = """
{question}
Please reason step by step, and put your final answer within \\boxed{}.
""".strip()

aime_2025_reader_cfg = dict(
    input_columns=["question"],
    output_column="answer",
)
aime_2025_infer_cfg = dict(
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
