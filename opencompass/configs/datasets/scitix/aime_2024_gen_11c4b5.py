from opencompass.datasets.scitix import AIME2024Dataset, AIME2024Evaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# adapted from https://github.com/deepseek-ai/DeepSeek-Math/blob/b8b0f8ce093d80bf8e9a641e44142f06d092c305/evaluation/run_subset_parallel.py
QUERY_TEMPLATE = """
{problem}
Please reason step by step, and put your final answer within \\boxed{}.
""".strip()

aime_2024_reader_cfg = dict(
    input_columns=["problem"],
    output_column="answer",
)
aime_2024_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt=QUERY_TEMPLATE),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
aime_2024_eval_cfg = dict(
    evaluator=dict(type=AIME2024Evaluator),
    pred_role="BOT",
)

aime_2024_datasets = [
    dict(
        abbr="aime-2024",
        type=AIME2024Dataset,
        path="scitix/aime-2024",
        reader_cfg=aime_2024_reader_cfg,
        infer_cfg=aime_2024_infer_cfg,
        eval_cfg=aime_2024_eval_cfg,
    )
]
