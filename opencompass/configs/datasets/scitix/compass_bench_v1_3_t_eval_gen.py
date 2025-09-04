from opencompass.datasets.scitix import (
    CompassBenchTEvalAfterCallingDataset,
    CompassBenchTEvalAfterCallingEvaluator,
    CompassBenchTEvalAtCallingDataset,
    CompassBenchTEvalAtCallingEvaluator,
    CompassBenchTEvalBeforeCallingDataset,
    CompassBenchTEvalBeforeCallingEvaluator,
    CompassBenchTEvalPlanDataset,
    CompassBenchTEvalPlanEvaluator,
)
from opencompass.openicl.icl_inferencer import ChatInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# before all callings - plan
compass_bench_v1_3_t_eval_plan_reader_cfg = dict(
    input_columns=["dialog"],  # use only one column for chat-based custom inputs
    output_column="ground_truth",
)
compass_bench_v1_3_t_eval_plan_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="",
    ),  # empty template as a placeholder for chat-based custom inputs
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ChatInferencer),
)
compass_bench_v1_3_t_eval_plan_eval_cfg = dict(
    evaluator=dict(
        type=CompassBenchTEvalPlanEvaluator,
        match_strategy="bertscore",
    ),
    pred_role="BOT",
)

# before calling - reason, retrieve, understand
compass_bench_v1_3_t_eval_before_calling_reader_cfg = dict(
    input_columns=["dialog"],  # use only one column for chat-based custom inputs
    output_column="ground_truth",
)
compass_bench_v1_3_t_eval_before_calling_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="",
    ),  # empty template as a placeholder for chat-based custom inputs
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ChatInferencer),
)
compass_bench_v1_3_t_eval_before_calling_eval_cfg = dict(
    evaluator=dict(type=CompassBenchTEvalBeforeCallingEvaluator),
    pred_role="BOT",
)

# at calling - instruct
compass_bench_v1_3_t_eval_at_calling_reader_cfg = dict(
    input_columns=["dialog"],  # use only one column for chat-based custom inputs
    output_column="ground_truth",
)
compass_bench_v1_3_t_eval_at_calling_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="",
    ),  # empty template as a placeholder for chat-based custom inputs
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ChatInferencer),
)
compass_bench_v1_3_t_eval_at_calling_eval_cfg = dict(
    evaluator=dict(type=CompassBenchTEvalAtCallingEvaluator),
    pred_role="BOT",
)

# after calling - review
compass_bench_v1_3_t_eval_after_calling_reader_cfg = dict(
    input_columns=["dialog"],  # use only one column for chat-based custom inputs
    output_column="ground_truth",
)
compass_bench_v1_3_t_eval_after_calling_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="",
    ),  # empty template as a placeholder for chat-based custom inputs
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=ChatInferencer),
)
compass_bench_v1_3_t_eval_after_calling_eval_cfg = dict(
    evaluator=dict(type=CompassBenchTEvalAfterCallingEvaluator),
    pred_role="BOT",
)

compass_bench_v1_3_t_eval_datasets = [
    dict(
        abbr="compass-bench-v1.3-t-eval-plan",
        type=CompassBenchTEvalPlanDataset,
        path="scitix/T-Eval",
        reader_cfg=compass_bench_v1_3_t_eval_plan_reader_cfg,
        infer_cfg=compass_bench_v1_3_t_eval_plan_infer_cfg,
        eval_cfg=compass_bench_v1_3_t_eval_plan_eval_cfg,
    ),
    dict(
        abbr="compass-bench-v1.3-t-eval-before-calling",
        type=CompassBenchTEvalBeforeCallingDataset,
        path="scitix/T-Eval",
        reader_cfg=compass_bench_v1_3_t_eval_before_calling_reader_cfg,
        infer_cfg=compass_bench_v1_3_t_eval_before_calling_infer_cfg,
        eval_cfg=compass_bench_v1_3_t_eval_before_calling_eval_cfg,
    ),
    dict(
        abbr="compass-bench-v1.3-t-eval-at-calling",
        type=CompassBenchTEvalAtCallingDataset,
        path="scitix/T-Eval",
        reader_cfg=compass_bench_v1_3_t_eval_at_calling_reader_cfg,
        infer_cfg=compass_bench_v1_3_t_eval_at_calling_infer_cfg,
        eval_cfg=compass_bench_v1_3_t_eval_at_calling_eval_cfg,
    ),
    dict(
        abbr="compass-bench-v1.3-t-eval-after-calling",
        type=CompassBenchTEvalAfterCallingDataset,
        path="scitix/T-Eval",
        reader_cfg=compass_bench_v1_3_t_eval_after_calling_reader_cfg,
        infer_cfg=compass_bench_v1_3_t_eval_after_calling_infer_cfg,
        eval_cfg=compass_bench_v1_3_t_eval_after_calling_eval_cfg,
    ),
]
