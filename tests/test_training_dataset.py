from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from scenearchitect.data.metadata import load_metadata
from scenearchitect.data.training_dataset import AspectBucketBatchSampler, SDXLMetadataDataset


class FakeTokenizer:
    model_max_length = 8

    def __call__(self, captions: list[str], **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(input_ids=torch.zeros((len(captions), 8), dtype=torch.long))


def test_load_metadata_resolves_repository_relative_images(tmp_path: Path) -> None:
    image = tmp_path / "data" / "processed" / "images" / "one.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (64, 64), "white").save(image)
    metadata = tmp_path / "data" / "processed" / "metadata" / "train.jsonl"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps(
            {
                "image": "data/processed/images/one.jpg",
                "caption": "anime street",
                "bucket_width": 64,
                "bucket_height": 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_metadata(metadata, tmp_path)
    assert records[0]["_image_path"] == str(image.resolve())


def test_bucket_dataset_produces_fixed_shape_batches(tmp_path: Path) -> None:
    records = []
    for index, size in enumerate(((96, 64), (80, 64), (64, 96))):
        image = tmp_path / f"{index}.png"
        Image.new("RGB", size, "white").save(image)
        bucket = (64, 64) if index < 2 else (64, 96)
        records.append(
            {
                "image": str(image),
                "_image_path": str(image),
                "caption": "anime architecture",
                "bucket_width": bucket[0],
                "bucket_height": bucket[1],
            }
        )
    dataset = SDXLMetadataDataset(
        records,
        FakeTokenizer(),
        FakeTokenizer(),
        seed=42,
        center_crop=False,
        random_flip=True,
    )
    assert dataset[0]["pixel_values"].shape == (3, 64, 64)
    sampler = AspectBucketBatchSampler(dataset, batch_size=2, seed=42)
    batches = list(sampler)
    assert sorted(len(batch) for batch in batches) == [1, 2]
    assert all(len({dataset.bucket(index) for index in batch}) == 1 for batch in batches)
