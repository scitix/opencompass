import asyncio
import json
import os

import json_repair
from openai import AsyncOpenAI
from tqdm import tqdm

from opencompass.datasets.scitix import (
    CompassBenchTEvalBeforeCallingDataset,
    CompassBenchTEvalBeforeCallingEvaluator,
)

CONCURRENCY = 32
# MODEL_NAME = "/models/preset/deepseek-ai/DeepSeek-V3.1/v1.0/"
MODEL_NAME = "simaas-qwen2-5-7b-instruct-v1"
# MODEL_NAME = "simaas-qwen2-5-72b-instruct-v1"
# MODEL_NAME = "simaas-deepseek-v3-v1"
# MODEL_NAME = "gpt-4.1"
# MODEL_NAME = "gemini-2.5-pro"


async def get_prediction(
    item: dict, model_client: AsyncOpenAI, semaphore: asyncio.Semaphore
):
    async with semaphore:
        try:
            resp = await model_client.chat.completions.create(
                model=MODEL_NAME,
                messages=json.loads(item["messages"]),
                tools=json.loads(item["tools"]),
                tool_choice="auto",
                max_tokens=8192,
                temperature=0.6,
            )
            return resp
        except Exception as e:
            return e


async def run_single(
    index: int, item: dict, model_client: AsyncOpenAI, semaphore: asyncio.Semaphore
):
    resp = await get_prediction(item, model_client, semaphore)
    return index, resp


async def main():
    dataset = CompassBenchTEvalBeforeCallingDataset.load(
        "scitix/T-Eval",
        norm_tool_name=True,
        # lang="en",
        # num_examples=5,
    )
    evaluator = CompassBenchTEvalBeforeCallingEvaluator()

    model_client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_API_BASE", "http://172.18.42.98:8000/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    )

    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        asyncio.create_task(run_single(i, item, model_client, semaphore))
        for i, item in enumerate(dataset)
    ]

    resps = [None] * len(dataset)
    with tqdm(total=len(dataset)) as pbar:
        for fut in asyncio.as_completed(tasks):
            index, resp = await fut
            resps[index] = resp
            pbar.update(1)
    await model_client.close()

    preds = [None] * len(dataset)
    refs = [item["ground_truth"] for item in dataset]
    for i, resp in enumerate(resps):
        if isinstance(resp, Exception):
            preds[i] = f"ERROR: {repr(resp)}"
            continue
        if resp.choices[0].message.tool_calls:
            if len(resp.choices[0].message.tool_calls) > 1:
                preds[i] = "Only one step is allowed."
            else:
                try:
                    args_json = json.loads(
                        resp.choices[0].message.tool_calls[0].function.arguments
                    )
                except Exception:
                    args_json = resp.choices[0].message.tool_calls[0].function.arguments
                preds[i] = {
                    "thought": resp.choices[0].message.content,
                    "name": resp.choices[0].message.tool_calls[0].function.name,
                    "args": args_json,
                }
        else:
            preds[i] = json_repair.repair_json(resp.choices[0].message.content or "")

    metrics = evaluator.score(preds, refs, test_set=dataset)
    if "details" in metrics and len(metrics["details"]) == len(resps):
        for i, detail in enumerate(metrics["details"]):
            raw_resp = resps[i]
            if isinstance(raw_resp, Exception):
                detail["resp"] = f"ERROR: {repr(raw_resp)}"
            else:
                detail["resp"] = raw_resp.model_dump_json(exclude_none=True)
            detail["answer"] = json.loads(detail["answer"])
            detail["messages"] = dataset[i]["messages"]
            detail["tools"] = dataset[i]["tools"]
    metrics["model"] = MODEL_NAME

    print(f"Model: {MODEL_NAME}")
    print("+----------------+----------+")
    print(f"| {'Metric':<14} | {'Score':<8} |")
    print("+----------------+----------+")
    print(f"| {'Parse Rate':<14} | {metrics['parse_rate']:>6.2f}% |")
    print(f"| {'Thought':<14} | {metrics['thought']:>6.2f}% |")
    print(f"| {'Name':<14} | {metrics['name']:>6.2f}% |")
    print(f"| {'Args F1':<14} | {metrics['args_f1_score']:>6.2f}% |")
    print("+----------------+----------+")

    out_file = f"t-eval2-{MODEL_NAME}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
