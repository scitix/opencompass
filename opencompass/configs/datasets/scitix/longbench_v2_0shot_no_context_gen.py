from opencompass.datasets.scitix.longbench_v2 import (
    LongBenchV2Dataset,
    LongBenchV2Evaluator,
)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# adapted from https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/prompts/0shot_no_context.txt
QUERY_TEMPLATE = """
What is the correct answer to this question: {question}
Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

What is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)".
""".strip()


longbench_v2_reader_cfg = dict(
    input_columns=[
        "question",
        "choice_A",
        "choice_B",
        "choice_C",
        "choice_D",
        "context",
    ],
    output_column="answer",
)
longbench_v2_infer_cfg = dict(
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
longbench_v2_eval_cfg = dict(
    evaluator=dict(type=LongBenchV2Evaluator),
    pred_role="BOT",
)

longbench_v2_datasets = [
    dict(
        abbr="longbench-v2",
        type=LongBenchV2Dataset,
        path="scitix/longbench-v2",
        reader_cfg=longbench_v2_reader_cfg,
        infer_cfg=longbench_v2_infer_cfg,
        eval_cfg=longbench_v2_eval_cfg,
    )
]
