from datasets import Dataset, load_dataset

from opencompass.datasets.base import BaseDataset
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path


@LOAD_DATASET.register_module()
class HLEDataset(BaseDataset):
    @staticmethod
    def load(
        path: str,
        text_only: bool = False,
        n_repeats: int = 1,
        num_examples: int | None = None,
        seed: int = 3407,
    ) -> Dataset:
        path = get_data_path(path)
        dataset = load_dataset(path, split="test")

        if text_only:
            dataset = dataset.filter(lambda x: x["image"] == "")

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
