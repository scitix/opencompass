import os

from opencompass.datasets.scitix import AIME2024Dataset, aime_2024_llmjudge_postprocess
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.models import OpenAISDKStreaming
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/math_eval.py
QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{problem}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()

EQUALITY_TEMPLATE = """
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: {prediction}
    Expression 2: {answer}
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
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role="HUMAN", prompt=EQUALITY_TEMPLATE),
                ],
            ),
        ),
        dataset_cfg=dict(
            abbr="eval_aime-2024",
            type=AIME2024Dataset,
            path="scitix/aime-2024",
            reader_cfg=aime_2024_reader_cfg,
        ),
        judge_cfg=dict(
            abbr="gpt-4o-2024-05-13",
            type=OpenAISDKStreaming,
            openai_api_base=[
                os.getenv("OC_JUDGE_API_BASE", "https://api.openai.com/v1")
            ],
            key=os.getenv("OC_JUDGE_API_KEY", ""),
            path="gpt-4o-2024-05-13",
            tokenizer_path="gpt-4o-2024-05-13",
            # max_seq_len=32768,
            query_per_second=32,
            batch_size=128,
            temperature=0.0,
            # max_out_len=8192,
            # verbose=True,
        ),
        dict_postprocessor=dict(type=aime_2024_llmjudge_postprocess),
    ),
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
