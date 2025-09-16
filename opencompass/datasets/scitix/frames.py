from urllib.parse import urlparse

from datasets import Dataset, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .optillm.plugins.readurls_plugin import extract_urls


# adapted from https://github.com/codelion/optillm/blob/770bf09baa1652c591e4a84cbab3dcffb12ef285/scripts/eval_frames_benchmark.py
@LOAD_DATASET.register_module()
class FRAMESDataset(BaseDataset):
    @staticmethod
    def load(
        path: str,
        wiki_path: str,
        n_repeats: int = 1,
        num_examples: int | None = None,
        seed: int = 0,
    ) -> Dataset:
        path = get_data_path(path)
        dataset = load_dataset(path, split="test")

        # restrict to a subset of the data for debugging
        if num_examples is not None:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            shuffled_dataset = dataset.shuffle(seed=seed)
            dataset = shuffled_dataset.select(
                range(min(num_examples, len(shuffled_dataset)))
            )

        # repeat examples
        if n_repeats > 1:
            original_indices = list(range(len(dataset)))
            repeated_indices = original_indices * n_repeats
            dataset = dataset.select(repeated_indices)

        # preload wiki contents
        wiki_path = get_data_path(wiki_path)
        wiki_ds = load_dataset(wiki_path, split="train")
        wiki_url2content = {w["link"]: w["text"] for w in wiki_ds}

        def _preprocess_sample(sample):
            sample["question"] = sample["Prompt"]
            # prompt
            chunks = []
            for url in extract_urls(sample["wiki_links"]):
                if url in wiki_url2content:
                    domain = urlparse(url).netloc
                    content = wiki_url2content[url]
                    chunks.append(f"{url} [Content from {domain}: {content}]")
            context = "\n".join(chunks)
            prompt = (
                "Here are the relevant Wikipedia articles:\n"
                f"{context}\n\n"
                "Based on all the information, answer the query.\n\n"
                f"Query: {sample['question']}\n"
            )
            sample["prompt"] = prompt
            return sample

        return dataset.map(_preprocess_sample)


def frames_llmjudge_postprocess(output: dict, output_path: str) -> dict:
    details = []

    for k, v in output.items():
        # 'prediction' here is the judge's raw output
        judgement = v.get("prediction", "")

        decision_prefixes = ["Decision:", "**Decision:**", '"Decision:"']
        explanation_prefixes = ["Explanation:", "**Explanation:**", '"Explanation:"']

        decision = "FALSE"
        explanation = ""
        for line in judgement.splitlines():
            for prefix in decision_prefixes:
                if line.startswith(prefix):
                    decision = line.split(prefix)[1].strip().upper()
                    break
            for prefix in explanation_prefixes:
                if line.startswith(prefix):
                    explanation = line.split(prefix)[1].strip()
                    break

        details.append(
            {
                "judgement": judgement,
                "answer": v.get("gold", ""),
                "decision": decision,
                "explanation": explanation,
            }
        )

    total_samples = len(details)
    correct_answers = sum(1 for d in details if d["decision"] == "TRUE")
    accuracy = correct_answers / total_samples if total_samples > 0 else 0
    return {"accuracy": accuracy * 100, "details": details}
