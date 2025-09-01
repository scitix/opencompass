import os

from datasets import Dataset, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .compass_bench_v1_3_subjective import compass_bench_llmjudge_postprocess


@LOAD_DATASET.register_module()
class CompassBenchCodeDataset(BaseDataset):
    @staticmethod
    def load(
        path: str,
        lang: str = "cn",
        n_repeats: int = 1,
        num_examples: int | None = None,
        seed: int = 3407,
        *args,
        **kwargs,
    ) -> Dataset:
        data_dir = get_data_path(path)
        path = os.path.join(data_dir, "code", f"compass_bench_coding_{lang}_val.json")
        data_files = {"test": path}
        dataset = load_dataset("json", data_files=data_files, split="test")

        def preprocess_sample(sample):
            checklist_md = ""
            for item in sample["checklist"]:
                checklist_md += f"- {item}\n"
            sample["checklist_md"] = checklist_md
            return sample

        dataset = dataset.map(preprocess_sample, num_proc=os.cpu_count())

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

        return dataset


def compass_bench_code_llmjudge_postprocess(output: dict, output_path: str) -> dict:
    return compass_bench_llmjudge_postprocess(output, output_path)
