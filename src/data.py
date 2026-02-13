"""
Data loading, splitting, and dataset utilities for few-shot learning.
"""

import torch
from torch.utils.data import Dataset
from torchvision.datasets import Flowers102

from src.constants import CLASS_NAMES


class MappedDataset(Dataset):
    """Dataset wrapper that remaps labels to contiguous indices.

    When splitting a dataset into base/novel subsets, original label indices
    may be non-contiguous. This wrapper maps them to ``[0, N)`` for use with
    standard classification losses.

    Args:
        dataset: Source dataset (or ``Subset``) to wrap.
        label_mapping: Dictionary mapping original labels to new contiguous indices.
    """

    def __init__(self, dataset, label_mapping):
        self.dataset = dataset
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        mapped_label = self.label_mapping[label]
        return image, mapped_label


def get_data(data_dir="./data", transform=None):
    """Load the Oxford Flowers-102 train / val / test splits.

    Args:
        data_dir: Root directory for dataset download.
        transform: Optional torchvision transform applied to images.

    Returns:
        Tuple of ``(train, val, test)`` Flowers102 datasets.
    """
    train = Flowers102(root=data_dir, split="train", download=True, transform=transform)
    val = Flowers102(root=data_dir, split="val", download=True, transform=transform)
    test = Flowers102(root=data_dir, split="test", download=True, transform=transform)
    return train, val, test


def base_novel_categories(dataset):
    """Split dataset classes into base (first half) and novel (second half).

    Args:
        dataset: A Flowers102 dataset instance.

    Returns:
        Tuple of ``(base_classes, novel_classes)`` as lists of class indices.
    """
    all_classes = set(dataset._labels)
    num_classes = len(all_classes)
    base_classes = list(range(num_classes))[:num_classes // 2]
    novel_classes = list(range(num_classes))[num_classes // 2:]
    return base_classes, novel_classes


def split_data(dataset, base_classes):
    """Create base and novel subsets with contiguous label mappings.

    Args:
        dataset: Full dataset with CLIP preprocessing applied.
        base_classes: List of class indices considered as *base*.

    Returns:
        Tuple of ``(base_dataset, novel_dataset)`` as :class:`MappedDataset`.
    """
    base_samples = []
    novel_samples = []

    base_set = set(base_classes)
    novel_classes = [i for i in range(len(CLASS_NAMES)) if i not in base_set]

    for sample_id, label in enumerate(dataset._labels):
        if label in base_set:
            base_samples.append(sample_id)
        else:
            novel_samples.append(sample_id)

    base_subset = torch.utils.data.Subset(dataset, base_samples)
    novel_subset = torch.utils.data.Subset(dataset, novel_samples)

    base_label_mapping = {old: new for new, old in enumerate(base_classes)}
    novel_label_mapping = {old: new for new, old in enumerate(novel_classes)}

    base_dataset = MappedDataset(base_subset, base_label_mapping)
    novel_dataset = MappedDataset(novel_subset, novel_label_mapping)

    return base_dataset, novel_dataset
