import logging
import os

import datasets as ds
import pytest

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)


@pytest.fixture
def dataset_path() -> str:
    return "cocostuff.py"


@pytest.mark.skipif(
    bool(os.environ.get("CI", False)),
    reason="Because this test downloads a large data set, we will skip running it on CI.",
)
def test_load_stuff_thing_dataset(dataset_path: str):
    dataset = ds.load_dataset(path=dataset_path, name="stuff-thing")

    expected_features = [
        "image",
        "image_id",
        "image_filename",
        "width",
        "height",
        "stuff_map",
        "objects",
    ]
    for expected_feature in expected_features:
        assert expected_feature in dataset["train"].features.keys()  # type: ignore
        assert expected_feature in dataset["validation"].features.keys()  # type: ignore

    assert dataset["train"].num_rows == 118280  # type: ignore
    assert dataset["validation"].num_rows == 5000  # type: ignore


@pytest.mark.skipif(
    bool(os.environ.get("CI", False)),
    reason="Because this test downloads a large data set, we will skip running it on CI.",
)
def test_load_stuff_only_dataset(dataset_path: str):
    dataset = ds.load_dataset(path=dataset_path, name="stuff-only")

    expected_features = [
        "image",
        "image_id",
        "image_filename",
        "width",
        "height",
        "objects",
    ]
    for expected_feature in expected_features:
        assert expected_feature in dataset["train"].features.keys()  # type: ignore
        assert expected_feature in dataset["validation"].features.keys()  # type: ignore

    assert dataset["train"].num_rows == 118280  # type: ignore
    assert dataset["validation"].num_rows == 5000  # type: ignore
