from opencompass.datasets.scitix import (
    CompassBenchMathDataset,
    CompassBenchMathEvaluator,
)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# adapted from https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/common.py
QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

compass_bench_v1_3_math_reader_cfg = dict(
    input_columns=["question", "A", "B", "C", "D"],
    output_column="answer",
)
compass_bench_v1_3_math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt=QUERY_TEMPLATE_MULTICHOICE),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
compass_bench_v1_3_math_eval_cfg = dict(
    evaluator=dict(type=CompassBenchMathEvaluator),
    pred_role="BOT",
)

compass_bench_v1_3_math_datasets = [
    dict(
        abbr="compass-bench-v1.3-math",
        type=CompassBenchMathDataset,
        path="scitix/CompassBench-v1.3",
        reader_cfg=compass_bench_v1_3_math_reader_cfg,
        infer_cfg=compass_bench_v1_3_math_infer_cfg,
        eval_cfg=compass_bench_v1_3_math_eval_cfg,
    )
]
