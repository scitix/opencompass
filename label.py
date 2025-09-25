import json
from typing import Generic, TypeVar

from datasets import DatasetDict
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from opencompass.datasets.cibench import CIBenchDataset
from opencompass.datasets.scitix import (
    AIME2024Dataset,
    AIME2025Dataset,
    CEvalDataset,
    CLUEWSCDataset,
    CNMO2024Dataset,
    CompassBenchTEvalPlanDataset,
    CSimpleQADataset,
    DROPSimpleEvalsDataset,
    FRAMESDataset,
    GPQADiamondSimpleEvalsDataset,
    HLEDataset,
    HumanEvalDataset,
    IFEvalDataset,
    LCBCodeGenerationDataset,
    LongBenchV2Dataset,
    MATH500Dataset,
    MMLUProDataset,
    MMLUReduxZeroEvalDataset,
    MMLUSimpleEvalsDataset,
    SimpleQASimpleEvalsDataset,
)

# prompt templates
TBaseModel = TypeVar("TBaseModel", bound=BaseModel)


class PydanticOutputParser(Generic[TBaseModel]):
    _PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema}
```"""

    def __init__(self, pydantic_object):
        self.pydantic_object = pydantic_object

    def get_format_instructions(self) -> str:
        # Copy schema to avoid altering original Pydantic schema.
        schema = dict(self.pydantic_object.model_json_schema().items())

        # Remove extraneous fields.
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        schema_str = json.dumps(reduced_schema, ensure_ascii=False)

        return self._PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)


class Ability(BaseModel):
    ability: str = Field(
        ...,
        description="考察的能力，格式为'一级标签-二级标签-三级标签'，如'语言-传统 NLP-情感分析'、'知识-常识'等",
    )
    reason: str


class ClassificationResult(BaseModel):
    abilities: list[Ability] = Field(
        ..., description="分类结果，按照匹配度从高到低排序"
    )


# QUERY_TEMPLATE = """
# 根据以下分类体系，指出给定的**数据集**考察了模型的哪些能力，并简要说明理由。

# # 分类体系
# - 语言
#   - 传统 NLP：分类、提取和结构化
#     - 情感分析
#     - 意图识别
#     - 多语言翻译
#     - 内容总结
#   - 生成式 NLP：理解、生成和创造
#     - 语义理解
#     - 内容评价
#     - 创作
#     - 对话
#   - 指令遵循
# - 知识
#   - STEM
#   - 人文
#   - 社科
#   - 多学科（同时包含 >=2 个学科的题目）
#   - 常识
# - 逻辑（按难度）
#   - 基础逻辑：常识、通用推理（如演绎、归纳、溯因）
#   - 复杂逻辑：逻辑谜题推理、学科推理
#   - 文本关联：多文本、跨文本关联分析
# - 数学（按难度）
#   - 初等数学：小学、初中、高中、数值计算
#   - 高等数学
#   - 应用数学：推理计算
#   - 竞赛
# - 代码
#   - 代码续写（如代码补全、代码编辑）、代码问答
#   - 代码生成
#   - 竞赛、面试
# - Agent
#   - 简单工具调用
#   - 多轮、多种工具调用
#   - 数据分析 agent
#   - 软件工程 agent
#   - Search Agent
# - 长文本
# - 多轮对话
# - 垂直领域
#   - Med

# # 返回格式
# {format_instructions}

# # 数据集
# {dataset_brief}
# """.strip()
QUERY_TEMPLATE = """
根据以下分类体系，指出给定的**数据集**最主要考察了模型的哪个能力，并简要说明理由。

# 分类体系
- 语言
  - 传统 NLP：分类、提取和结构化
    - 情感分析
    - 意图识别
    - 多语言翻译
    - 内容总结
  - 生成式 NLP：理解、生成和创造
    - 语义理解
    - 内容评价
    - 创作
    - 对话
  - 指令遵循
- 知识
  - STEM
  - 人文
  - 社科
  - 多学科（同时包含 >=2 个学科的题目）
  - 常识
- 逻辑（按难度）
  - 基础逻辑：常识、通用推理（如演绎、归纳、溯因）
  - 复杂逻辑：逻辑谜题推理、学科推理
  - 文本关联：多文本、跨文本关联分析
- 数学（按难度）
  - 初等数学：小学、初中、高中、数值计算
  - 高等数学
  - 应用数学：推理计算
  - 竞赛
- 代码
  - 代码续写（如代码补全、代码编辑）、代码问答
  - 代码生成
  - 竞赛、面试
- Agent
  - 简单工具调用
  - 多轮、多种工具调用
  - 数据分析 agent
  - 软件工程 agent
  - Search Agent
- 长文本
- 多轮对话
- 垂直领域
  - Med

# 返回格式
{format_instructions}

# 数据集
{dataset_brief}
""".strip()


# name -> (dataset_class, data_column_in_prompt, data_loader_kwargs)
DS_MAP = {
    "scitix/ifeval": (
        IFEvalDataset,
        ["prompt"],
        {},
    ),
    "scitix/cluewsc": (
        CLUEWSCDataset,
        ["text", "span1", "span2"],
        {},
    ),
    "scitix/mmlu_simple-evals": (
        MMLUSimpleEvalsDataset,
        ["Question", "A", "B", "C", "D"],
        {},
    ),
    "scitix/mmlu-redux_zero-eval": (
        MMLUReduxZeroEvalDataset,
        ["question", "choices"],
        {},
    ),
    "scitix/mmlu-pro": (
        MMLUProDataset,
        ["question", "options_str"],
        {},
    ),
    "scitix/c-eval": (
        CEvalDataset,
        ["dialog"],
        {},
    ),
    "scitix/c-simpleqa": (
        CSimpleQADataset,
        ["question"],
        {},
    ),
    "scitix/gpqa-diamond_simple-evals": (
        GPQADiamondSimpleEvalsDataset,
        ["Question", "A", "B", "C", "D"],
        {"n_repeats": 1},
    ),
    "scitix/simpleqa_simple-evals": (
        SimpleQASimpleEvalsDataset,
        ["problem"],
        {},
    ),
    "scitix/hle": (
        HLEDataset,
        ["question"],
        {"text_only": True},
    ),
    "scitix/drop_simple-evals": (
        DROPSimpleEvalsDataset,
        ["context", "completion"],
        {},
    ),
    "scitix/aime-2024": (
        AIME2024Dataset,
        ["problem"],
        {},
    ),
    "scitix/aime-2025": (
        AIME2025Dataset,
        ["question"],
        {},
    ),
    "scitix/cnmo-2024": (
        CNMO2024Dataset,
        ["question"],
        {},
    ),
    "scitix/math-500": (
        MATH500Dataset,
        ["problem"],
        {},
    ),
    "scitix/human-eval": (
        HumanEvalDataset,
        ["prompt"],
        {},
    ),
    "scitix/lcb-code-generation-lite": (
        LCBCodeGenerationDataset,
        ["dialog"],
        {},
    ),
    "scitix/frames": (
        FRAMESDataset,
        ["prompt"],
        {"wiki_path": "scitix/frames-wiki"},
    ),
    "scitix/longbench-v2": (
        LongBenchV2Dataset,
        ["question", "choice_A", "choice_B", "choice_C", "choice_D", "context"],
        {},
    ),
    "scitix/T-Eval": (
        CompassBenchTEvalPlanDataset,
        ["dialog"],
        {},
    ),
    "./data/cibench_dataset/cibench_generation/matplotlib": (
        CIBenchDataset,
        ["questions"],
        {},
    ),
}


# api
client = OpenAI()
model = "simaas-gpt-oss-120b-v1"
# model = "gpt-4.1-2025-04-14"
# model = "gemini-2.5-pro"
# model = "claude-sonnet-4-20250514"


# with open(f"labels_{model}.txt", "w") as fh:
with open(f"label_{model}.txt", "w") as fh:
    for ds_name, (ds_cls, ds_cols, data_loader_kwargs) in tqdm(DS_MAP.items()):
        ds_brief = f"""## Name
{ds_name}

## Samples
"""

        ds = ds_cls.load(ds_name, num_examples=5, seed=3407, **data_loader_kwargs)
        # retrieve test split if possible
        if isinstance(ds, DatasetDict):
            ds = ds["test"]
        if ds_cols != "ALL":
            ds = ds.select_columns(ds_cols)

        for _, sample in enumerate(ds):
            for k, v in sample.items():
                if isinstance(v, str) and len(v) > 2048:
                    sample[k] = v[:2048] + f"...(**{len(v) - 2048}** chars left)"
            ds_brief += f"- {sample}\n"
        logger.debug(ds_brief)

        format_instructions = PydanticOutputParser(
            # ClassificationResult
            Ability
        ).get_format_instructions()
        query = QUERY_TEMPLATE.format(
            format_instructions=format_instructions, dataset_brief=ds_brief
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )
        print(ds_name)
        print(resp.choices[0].message.content)
        fh.write(ds_name + "\n")
        fh.write(resp.choices[0].message.content + "\n")
        fh.write("---\n")
