from opencompass.datasets.scitix import AIME2025Dataset, AIME2025Evaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# adapted from https://github.com/huggingface/lighteval/blob/161d47cc1c10e3254d9b4144086d6650c1e9da70/src/lighteval/tasks/default_prompts.py
QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{question}
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
