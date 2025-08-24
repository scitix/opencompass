from opencompass.datasets.scitix import CNMO2024Dataset, CNMO2024Evaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/math_eval.py
QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()

cnmo_2024_reader_cfg = dict(
    input_columns=["question"],
    output_column="answer",
)
cnmo_2024_infer_cfg = dict(
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
cnmo_2024_eval_cfg = dict(
    evaluator=dict(type=CNMO2024Evaluator),
    pred_role="BOT",
)

cnmo_2024_datasets = [
    dict(
        abbr="cnmo-2024",
        type=CNMO2024Dataset,
        path="scitix/cnmo-2024",
        reader_cfg=cnmo_2024_reader_cfg,
        infer_cfg=cnmo_2024_infer_cfg,
        eval_cfg=cnmo_2024_eval_cfg,
    )
]
