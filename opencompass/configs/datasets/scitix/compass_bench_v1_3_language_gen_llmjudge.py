from mmengine.config import read_base

from opencompass.datasets.scitix import (
    CompassBenchLanguageDataset,
    compass_bench_language_llmjudge_postprocess,
)
from opencompass.openicl.icl_evaluator import LMEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.summarizers import SubjectiveSummarizer

with read_base():
    from opencompass.datasets.scitix.compass_bench_v1_3_subjective import (
        COMPARE_TEMPLATE_EN,
    )

compass_bench_v1_3_language_reader_cfg = dict(
    input_columns=["instruction", "checklist_md"],
    output_column=None,  # outputs will be generated at runtime
)
compass_bench_v1_3_language_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt="{instruction}"),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
compass_bench_v1_3_language_eval_cfg = dict(
    evaluator=dict(
        type=LMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role="HUMAN", prompt=COMPARE_TEMPLATE_EN),
                ],
            ),
        ),
        dict_postprocessor=(dict(type=compass_bench_language_llmjudge_postprocess)),
    ),
    pred_role="BOT",
)

compass_bench_v1_3_language_datasets = [
    dict(
        abbr="compass-bench-v1.3-language",
        type=CompassBenchLanguageDataset,
        path="scitix/CompassBench-v1.3",
        lang="en",
        reader_cfg=compass_bench_v1_3_language_reader_cfg,
        infer_cfg=compass_bench_v1_3_language_infer_cfg,
        eval_cfg=compass_bench_v1_3_language_eval_cfg,
        # for llm judge partitioners
        mode="m2n",
        summarizer=dict(
            type=SubjectiveSummarizer,
            function="subjective",
        ),
    )
]
