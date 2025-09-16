import os

from opencompass.datasets.scitix import FRAMESDataset, frames_llmjudge_postprocess
from opencompass.evaluator import GenericLLMEvaluator
from opencompass.models import OpenAISDKStreaming
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

# adapted from https://github.com/codelion/optillm/blob/770bf09baa1652c591e4a84cbab3dcffb12ef285/scripts/eval_frames_benchmark.py
GRADER_TEMPLATE = """===Task===
I need your help in evaluating an answer provided by an LLM against a ground
truth answer. Your task is to determine if the ground truth answer is present in the LLM's
response. Please analyze the provided data and make a decision.
===Instructions===
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answer".
2. Consider the substance of the answers - look for equivalent information or correct answers.
Do not focus on exact wording unless the exact wording is crucial to the meaning.
3. Your final decision should be based on whether the meaning and the vital facts of the
"Ground Truth Answer" are present in the "Predicted Answer:"
===Input Data===
- Question: {question}
- Predicted Answer: {prediction}
- Ground Truth Answer: {Answer}
===Output Format===
Provide your final evaluation in the following format:
"Explanation:" (How you made the decision?)
"Decision:" ("TRUE" or "FALSE" )
Please proceed with the evaluation."""

frames_reader_cfg = dict(
    input_columns=["prompt", "question"],  # `question` is only for evaluator
    output_column="Answer",
)
frames_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role="SYSTEM", prompt="You are a helpful assistant."),
                dict(role="HUMAN", prompt="{prompt}"),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)
frames_eval_cfg = dict(
    evaluator=dict(
        type=GenericLLMEvaluator,
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role="SYSTEM", prompt="You are a helpful assistant."),
                    dict(role="HUMAN", prompt=GRADER_TEMPLATE),
                ],
            ),
        ),
        dataset_cfg=dict(
            abbr="eval_frames",
            type=FRAMESDataset,
            path="scitix/frames",
            wiki_path="scitix/frames-wiki",
            reader_cfg=frames_reader_cfg,
        ),
        judge_cfg=dict(
            abbr="gpt-4o-mini",
            type=OpenAISDKStreaming,
            meta_template=dict(
                round=[
                    dict(role="SYSTEM", api_role="SYSTEM"),
                    dict(role="HUMAN", api_role="HUMAN"),
                    dict(role="BOT", api_role="BOT", generate=True),
                ]
            ),
            openai_api_base=[
                os.getenv("OC_JUDGE_API_BASE", "https://api.openai.com/v1")
            ],
            key=os.getenv("OC_JUDGE_API_KEY", ""),
            path="gpt-4o-mini",
            tokenizer_path="gpt-4o-mini",
            # max_seq_len=32768,
            query_per_second=16,
            batch_size=128,
            temperature=0.3,
            max_out_len=300,
            # verbose=True,
        ),
        dict_postprocessor=dict(type=frames_llmjudge_postprocess),
    ),
    pred_role="BOT",
)

frames_datasets = [
    dict(
        abbr="frames",
        type=FRAMESDataset,
        path="scitix/frames",
        wiki_path="scitix/frames-wiki",
        reader_cfg=frames_reader_cfg,
        infer_cfg=frames_infer_cfg,
        eval_cfg=frames_eval_cfg,
    )
]
