from opencompass.datasets.scitix import MMLUProDataset, MMLUProEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.utils.text_postprocessors import match_answer_pattern

QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.

Question:\n
{question}

Options:\n
{options_str}

""".strip()


mmlu_pro_reader_cfg = dict(
    input_columns=["question", "options_str"],
    output_column="answer",
)
mmlu_pro_infer_cfg = dict(
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

mmlu_pro_eval_cfg = dict(
    evaluator=dict(type=MMLUProEvaluator),
    pred_postprocessor=dict(
        type=match_answer_pattern,
        answer_pattern=r"(?i)ANSWER\s*:\s*([A-P])",
    ),
)

mmlu_pro_datasets = [
    dict(
        abbr="mmlu-pro",
        type=MMLUProDataset,
        path="scitix/mmlu-pro",
        reader_cfg=mmlu_pro_reader_cfg,
        infer_cfg=mmlu_pro_infer_cfg,
        eval_cfg=mmlu_pro_eval_cfg,
    )
]
