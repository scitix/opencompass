import asyncio
import json
import os
from openai import AsyncOpenAI
from tqdm import tqdm
import json_repair

CONCURRENCY = 32
MODEL_NAME = "gpt-4.1"
GRADER_TEMPLATE = """
===Role===
You are an expert LLM evaluator. 
Your task is to determine whether a predicted tool call correctly fulfills 
the user's intent, based on the task description, conversation context, 
and tool definitions.

===Context Information===
[User Task]
{user_task}

[Conversation History]
{history_messages}

[Available Tool Definitions]
{tools_definitions}

===Input Data===
Predicted Tool Call: {prediction}
Ground Truth Tool Call: {ground_truth}

===Evaluation Criteria===
1. Understand the user's intent from the task and conversation.
2. Examine the tool definitions to interpret tool names and parameters.
3. Compare the Predicted Tool Call with the Ground Truth Tool Call:
   - If they are semantically equivalent (tool name, arguments, and purpose match), output TRUE.
   - If any key aspect differs or fails to fulfill the intent, output FALSE.
4. Ignore superficial differences such as argument order or formatting.

===Output Format===
Output a single JSON object:
{{
  "explanation": "Briefly explain your reasoning",
  "decision": "TRUE" or "FALSE"
}}

Please proceed with the evaluation.
""".strip()


async def get_prediction(
    item: dict, model_client: AsyncOpenAI, semaphore: asyncio.Semaphore
):
    async with semaphore:
        try:
            history_messages = json.loads(item["messages"])
            history_messages_str = "\n".join(str(msg) for msg in history_messages)
            tools_definitions = json.loads(item["tools"])
            tools_definitions_str = "\n".join(str(td) for td in tools_definitions)
            user_task = history_messages[1]["content"]

            resp = json.loads(item["resp"])
            msg = resp["choices"][0]["message"]
            prediction = {}
            if "content" in msg:
                prediction["content"] = msg["content"]
            if "tool_calls" in msg and msg["tool_calls"]:
                tool_calls = []
                for tc in msg["tool_calls"]:
                    tool_calls.append(
                        {
                            "name": tc["function"]["name"],
                            "args": tc["function"]["arguments"],
                        }
                    )
                prediction["tool_calls"] = tool_calls
            ground_truth = item["answer"]

            grade_msg = GRADER_TEMPLATE.format(
                user_task=user_task,
                history_messages=history_messages_str,
                tools_definitions=tools_definitions_str,
                prediction=prediction,
                ground_truth=ground_truth,
            )

            resp = await model_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": grade_msg}],
                max_tokens=8192,
                temperature=0.6,
            )
            judge_raw = resp.choices[0].message.content
            judge = json_repair.loads(judge_raw)
            # 标准化 decision
            judge["decision"] = str(judge.get("decision", "")).upper() == "TRUE"
            # 补充一些原始信息
            judge["args_f1"] = item.get("args_f1_score")
            judge["match"] = (judge.get("args_f1") == 1) and judge["decision"]
            judge["raw_response"] = judge_raw
            return judge
        except Exception as e:
            return {"error": str(e)}


async def run_single(
    index: int, item: dict, model_client: AsyncOpenAI, semaphore: asyncio.Semaphore
):
    resp = await get_prediction(item, model_client, semaphore)
    return index, resp


async def main(exp):
    with open(f"{exp}.json", "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    details = obj.get("details", [])

    model_client = AsyncOpenAI(
        base_url=os.getenv("OPENAI_API_BASE", "http://172.18.42.98:8000/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    )

    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        asyncio.create_task(run_single(i, item, model_client, semaphore))
        for i, item in enumerate(details)
    ]

    resps = [None] * len(details)
    with tqdm(total=len(details)) as pbar:
        for fut in asyncio.as_completed(tasks):
            index, resp = await fut
            resps[index] = resp
            pbar.update(1)
    await model_client.close()

    total = len(resps)
    valid_results = []
    for i, resp in enumerate(resps):
        if isinstance(resp, dict) and ("decision" in resp or "error" in resp):
            if "decision" in resp:
                valid_results.append(resp)
            else:
                print(f"Error in item {i}: {resp.get('error')}")

    total = len(valid_results)
    true_count = sum(1 for r in valid_results if r.get("decision") is True)
    accuracy = true_count / total if total else 0.0
    print(f"Parse Rate: {len(valid_results)}/{len(resps)}")
    print(f"Decision TRUE count: {true_count}")
    print(f"Total counted: {total}")
    print(f"Accuracy: {accuracy:.4f}")

    # 保存结果
    output_path = f"{exp}.results.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "source_file": f"{exp}.json",
                "judge": MODEL_NAME,
                "num_evaluated": total,
                "accuracy": accuracy,
                "results": valid_results,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    # asyncio.run(main("t-eval2-gemini-2.5-pro"))
    # asyncio.run(main("t-eval2-gpt-4.1"))
    # asyncio.run(main("t-eval2-dsv31"))
    # asyncio.run(main("t-eval2-simaas-deepseek-v3-v1"))
    # asyncio.run(main("t-eval2-simaas-qwen2-5-72b-instruct-v1"))
    # asyncio.run(main("t-eval2-simaas-qwen2-5-7b-instruct-v1"))
    # asyncio.run(main("t-eval2-gemini-2.5-pro-en"))
    # asyncio.run(main("t-eval2-gpt-4.1-en"))
    # asyncio.run(main("t-eval2-dsv31-en"))
    # asyncio.run(main("t-eval2-simaas-deepseek-v3-v1-en"))
    asyncio.run(main("t-eval2-simaas-qwen2-5-72b-instruct-v1-en"))
    # asyncio.run(main("t-eval2-simaas-qwen2-5-7b-instruct-v1-en"))
