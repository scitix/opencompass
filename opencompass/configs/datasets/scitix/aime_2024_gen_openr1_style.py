from opencompass.datasets.scitix import AIME2024Dataset, AIME2024Evaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# adapted from https://github.com/huggingface/lighteval/blob/161d47cc1c10e3254d9b4144086d6650c1e9da70/src/lighteval/tasks/default_prompts.py
QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{problem}
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
            ]
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
