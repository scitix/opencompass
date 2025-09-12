import os

from opencompass.datasets.scitix import (
    LCBCodeGenerationDataset,
    LCBCodeGenerationEvaluator,
)
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# code generation
lcb_code_generation_reader_cfg = dict(
    input_columns=["dialog"],  # use only one column for chat-based custom inputs
    output_column=None,  # outputs will be generated at runtime
)

lcb_code_generation_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="",
    ),  # empty template as a placeholder for chat-based custom inputs
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ChatInferencer),
)

lcb_code_generation_eval_cfg = dict(
    evaluator=dict(
        type=LCBCodeGenerationEvaluator,
        api=os.getenv("OC_EVAL_API", "http://localhost:11451/evaluations"),
        num_workers=4,
        timeout=6.0,
        k=1,  # pass@k
    ),
    pred_role="BOT",
)

lcb_code_generation_dataset = dict(
    type=LCBCodeGenerationDataset,
    abbr="lcb-code-generation-lite_2408_2411",
    path="scitix/lcb-code-generation-lite",
    version_tag="release_v6",
    start_date="2024-08-01",
    end_date="2024-11-30",
    reader_cfg=lcb_code_generation_reader_cfg,
    infer_cfg=lcb_code_generation_infer_cfg,
    eval_cfg=lcb_code_generation_eval_cfg,
)

# all datasets
lcb_datasets = [lcb_code_generation_dataset]
