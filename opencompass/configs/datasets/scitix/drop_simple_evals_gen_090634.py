from opencompass.datasets.scitix import DropSimpleEvalsDataset, DropSimpleEvalsEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import RandomRetriever

# adapted from https://github.com/openai/simple-evals/blob/ee3b0318d8d1d9d72755a4120879be65f7c07e9e/drop_eval.py
FEW_SHOT_TEMPLATE = """
---
{context} {completion}
""".strip()
QUERY_TEMPLATE = """
You will be asked to read a passage and answer a question. Some examples of passages and Q&A are provided below.

# Examples
</E>
# Your Task

---
{context} 

Think step by step, then write a line of the form "Answer: $ANSWER" at the end of your response.
""".lstrip()  # keep the trailing line break to align with the official prompt

drop_reader_cfg = dict(
    input_columns=["context", "completion"],
    output_column="ref_text",
    train_split="train",
    test_split="test",
)
drop_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=FEW_SHOT_TEMPLATE,
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=QUERY_TEMPLATE,
        ice_token="</E>",
    ),
    retriever=dict(
        type=RandomRetriever,
        ice_num=3,
        seed=42,
        ice_separator="\n\n",
    ),
    inferencer=dict(type=GenInferencer),
)
drop_eval_cfg = dict(
    evaluator=dict(type=DropSimpleEvalsEvaluator),
    pred_role="BOT",
)

drop_datasets = [
    dict(
        abbr="drop_simple-evals",
        type=DropSimpleEvalsDataset,
        path="scitix/drop_simple-evals",
        reader_cfg=drop_reader_cfg,
        infer_cfg=drop_infer_cfg,
        eval_cfg=drop_eval_cfg,
    )
]
