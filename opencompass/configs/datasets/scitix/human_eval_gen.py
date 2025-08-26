import os

from opencompass.datasets.scitix import HumanEvalDataset, HumanEvalEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# adapted from https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/humaneval_eval.py
QUERY_TEMPLATE = """
Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.
{prompt}
""".strip()

human_eval_reader_cfg = dict(
    input_columns=["prompt"],
    output_column="canonical_solution",
)
human_eval_infer_cfg = dict(
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
human_eval_eval_cfg = dict(
    evaluator=dict(
        type=HumanEvalEvaluator,
        api=os.getenv("OC_EVAL_API", "http://localhost:11451/evaluations"),
        num_workers=4,
        timeout=5.0,
        k=1,  # pass@k
    ),
    pred_role="BOT",
)

human_eval_datasets = [
    dict(
        abbr="human-eval",
        type=HumanEvalDataset,
        path="scitix/human-eval",
        reader_cfg=human_eval_reader_cfg,
        infer_cfg=human_eval_infer_cfg,
        eval_cfg=human_eval_eval_cfg,
    )
]
