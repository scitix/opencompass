import argparse
import json
import os

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from opencompass.datasets.scitix.optillm.plugins.memory_plugin import (
    run as run_memory_plugin,
)
from opencompass.datasets.scitix.optillm.plugins.readurls_plugin import (
    run as run_readurls_plugin,
)

model_client = OpenAI(
    base_url=os.getenv("MODEL_BASE_URL", ""),
    api_key=os.getenv("MODEL_KEY", ""),
)
judge_client = OpenAI(
    base_url=os.getenv("JUDGE_BASE_URL", ""),
    api_key=os.getenv("JUDGE_KEY", ""),
)


def load_existing_results(filename: str) -> list[dict]:
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_result(filename: str, result: dict):
    results = load_existing_results(filename)
    results.append(result)
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)


def get_last_processed_index(results: list[dict]) -> int:
    if not results:
        return -1
    return max(int(r.get("index", -1)) for r in results)


def evaluate_response(
    question: str, llm_response: str, ground_truth: str, model: str
) -> dict[str, str]:
    evaluation_prompt = f"""===Task===
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
- Predicted Answer: {llm_response}
- Ground Truth Answer: {ground_truth}
===Output Format===
Provide your final evaluation in the following format:
"Explanation:" (How you made the decision?)
"Decision:" ("TRUE" or "FALSE" )
Please proceed with the evaluation."""

    evaluation_response = judge_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": evaluation_prompt},
        ],
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.3,
    )

    evaluation_text = evaluation_response.choices[0].message.content.strip()

    # Extract the decision and explanation
    decision_prefixes = ["Decision:", "**Decision:**", '"Decision:"']
    explanation_prefixes = ["Explanation:", "**Explanation:**", '"Explanation:"']

    decision = "FALSE"
    explanation = ""
    for line in evaluation_text.splitlines():
        for prefix in decision_prefixes:
            if line.startswith(prefix):
                decision = line.split(prefix)[1].strip().upper()
                break
        for prefix in explanation_prefixes:
            if line.startswith(prefix):
                explanation = line.split(prefix)[1].strip()
                break

    return {"decision": decision, "explanation": explanation}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--judge", type=str, required=True)
    args = parser.parse_args()

    model = args.model
    judge = args.judge
    filename = f"./eval_frames_{model.replace('/', '_')}__judgedby_{judge.replace('/', '_')}.json"
    existing_results = load_existing_results(filename)
    last_processed_index = get_last_processed_index(existing_results)

    frames_ds = load_dataset("./data/frames", split="test")
    for i, sample in tqdm(enumerate(frames_ds), total=len(frames_ds)):
        if i <= last_processed_index:
            continue

        question = sample["Prompt"]
        answer = sample["Answer"]
        # init
        wiki_links = sample["wiki_links"]
        initial_query = (
            "Here are the relevant Wikipedia articles:\n"
            f"{wiki_links}\n\n"
            "Based on all the information, answer the query.\n\n"
            f"Query: {question}\n\n"
        )
        # readurls
        modified_query, _ = run_readurls_plugin(
            system_prompt="You are a helpful assistant.",
            initial_query=initial_query,
            client=model_client,
            model=model,
        )
        # memory
        llm_response, _ = run_memory_plugin(
            system_prompt="You are a helpful assistant.",
            initial_query=modified_query,
            client=model_client,
            model=model,
        )
        # judge
        evaluation = evaluate_response(question, llm_response, answer, judge)

        result = {
            "index": i,
            "prompt": question,
            "ground_truth": answer,
            "llm_response": llm_response,
            "evaluation_decision": evaluation["decision"],
            "evaluation_explanation": evaluation["explanation"],
            "reasoning_type": sample["reasoning_types"],
        }
        save_result(filename, result)

    # Calculate and print summary statistics
    results = load_existing_results(filename)
    total_samples = len(results)
    correct_answers = sum(1 for r in results if r["evaluation_decision"] == "TRUE")
    accuracy = correct_answers / total_samples

    print(f"Model: {model}")
    print(f"Total samples: {total_samples}")
    print(f"Correct answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.2%}")

    # Print accuracy by reasoning type
    reasoning_types = set(r["reasoning_type"] for r in results)
    for rt in reasoning_types:
        rt_samples = [r for r in results if r["reasoning_type"] == rt]
        rt_correct = sum(1 for r in rt_samples if r["evaluation_decision"] == "TRUE")
        rt_accuracy = rt_correct / len(rt_samples)
        print(f"Accuracy for {rt}: {rt_accuracy:.2%}")
