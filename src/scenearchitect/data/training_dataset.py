"""Metadata-backed SDXL training dataset with deterministic aspect buckets."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Sampler


class SDXLMetadataDataset(Dataset[dict[str, Any]]):
    """Load and crop processed images; augmentation is deterministic per epoch/item."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        tokenizer_one: Any,
        tokenizer_two: Any,
        *,
        seed: int,
        center_crop: bool,
        random_flip: bool,
    ) -> None:
        self.records = [dict(record) for record in records]
        self.seed = seed
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.epoch = 0
        captions = [str(record["caption"]) for record in self.records]
        self.input_ids_one = tokenizer_one(
            captions,
            max_length=tokenizer_one.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        self.input_ids_two = tokenizer_two(
            captions,
            max_length=tokenizer_two.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def bucket(self, index: int) -> tuple[int, int]:
        record = self.records[index]
        return int(record["bucket_width"]), int(record["bucket_height"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        target_width, target_height = self.bucket(index)
        with Image.open(record["_image_path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        original_width, original_height = image.size
        scale = max(target_width / original_width, target_height / original_height)
        resized_width = max(target_width, round(original_width * scale))
        resized_height = max(target_height, round(original_height * scale))
        image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        if self.center_crop:
            crop_x = max(0, (resized_width - target_width) // 2)
            crop_y = max(0, (resized_height - target_height) // 2)
        else:
            crop_x = rng.randint(0, max(0, resized_width - target_width))
            crop_y = rng.randint(0, max(0, resized_height - target_height))
        image = image.crop((crop_x, crop_y, crop_x + target_width, crop_y + target_height))
        if self.random_flip and rng.random() < 0.5:
            image = ImageOps.mirror(image)

        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        pixel_values = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return {
            "pixel_values": pixel_values,
            "input_ids_one": self.input_ids_one[index],
            "input_ids_two": self.input_ids_two[index],
            "original_size": (original_height, original_width),
            "crop_top_left": (crop_y, crop_x),
            "target_size": (target_height, target_width),
        }


class AspectBucketBatchSampler(Sampler[list[int]]):
    """Shuffle within compatible buckets and yield fixed-shape mini-batches."""

    def __init__(
        self,
        dataset: SDXLMetadataDataset,
        batch_size: int,
        *,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self.groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index in range(len(dataset)):
            self.groups[dataset.bucket(index)].append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.dataset.set_epoch(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for indices in self.groups.values():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return sum(len(indices) // self.batch_size for indices in self.groups.values())
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.groups.values())


def collate_sdxl(examples: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in examples]),
        "input_ids_one": torch.stack([item["input_ids_one"] for item in examples]),
        "input_ids_two": torch.stack([item["input_ids_two"] for item in examples]),
        "original_sizes": torch.tensor([item["original_size"] for item in examples]),
        "crop_top_lefts": torch.tensor([item["crop_top_left"] for item in examples]),
        "target_sizes": torch.tensor([item["target_size"] for item in examples]),
    }
