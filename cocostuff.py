import copy
import json
import logging
import os
from collections import defaultdict
from typing import Dict, TypedDict

import datasets as ds

logger = logging.getLogger(__name__)

_CITATION = """\
@INPROCEEDINGS{caesar2018cvpr,
  title={COCO-Stuff: Thing and stuff classes in context},
  author={Caesar, Holger and Uijlings, Jasper and Ferrari, Vittorio},
  booktitle={Computer vision and pattern recognition (CVPR), 2018 IEEE conference on},
  organization={IEEE},
  year={2018}
}
"""

_DESCRIPTION = """\
COCO-Stuff augments all 164K images of the popular COCO dataset with pixel-level stuff annotations. These annotations can be used for scene understanding tasks like semantic segmentation, object detection and image captioning.
"""

_HOMEPAGE = "https://github.com/nightrome/cocostuff"

_LICENSE = """\
COCO-Stuff is a derivative work of the COCO dataset. The authors of COCO do not in any form endorse this work. Different licenses apply:
- COCO images: Flickr Terms of use
- COCO annotations: Creative Commons Attribution 4.0 License
- COCO-Stuff annotations & code: Creative Commons Attribution 4.0 License
"""


class URLs(TypedDict):
    train: str
    val: str
    stuffthingmaps_trainval: str
    stuff_trainval: str
    labels: str


_URLS: URLs = {
    "train": "http://images.cocodataset.org/zips/train2017.zip",
    "val": "http://images.cocodataset.org/zips/val2017.zip",
    "stuffthingmaps_trainval": "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip",
    "stuff_trainval": "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuff_trainval2017.zip",
    "labels": "https://raw.githubusercontent.com/nightrome/cocostuff/master/labels.txt",
}


class GenerateExamplesArguments(TypedDict):
    image_dirpath: str
    stuff_dirpath: str
    stuff_thing_maps_dirpath: str
    labels_path: str
    split: str


def _load_json(json_path: str):
    logger.info(f"Load json from {json_path}")
    with open(json_path, "r") as rf:
        json_data = json.load(rf)
    return json_data


def _load_labels(labels_path: str) -> Dict[int, str]:
    label_id_to_label_name: Dict[int, str] = {}

    logger.info(f"Load labels from {labels_path}")
    with open(labels_path, "r") as rf:
        for line in rf:
            label_id_str, label_name = line.strip().split(": ")
            label_id = int(label_id_str)

            # correspondence between .png annotation & category_id · Issue #17 · nightrome/cocostuff https://github.com/nightrome/cocostuff/issues/17
            # Label matching, 182 or 183 labels? · Issue #8 · nightrome/cocostuff https://github.com/nightrome/cocostuff/issues/8
            if label_id == 0:
                # for unlabeled class
                assert label_name == "unlabeled", label_name
                label_id_to_label_name[183] = label_name
            else:
                label_id_to_label_name[label_id] = label_name

    assert len(label_id_to_label_name) == 183

    return label_id_to_label_name


class CocoStuffDataset(ds.GeneratorBasedBuilder):

    VERSION = ds.Version("1.0.0")  # type: ignore

    BUILDER_CONFIGS = [
        ds.BuilderConfig(
            name="stuff-thing",
            version=VERSION,  # type: ignore
            description="Stuff+thing PNG-style annotations on COCO 2017 trainval",
        ),
        ds.BuilderConfig(
            name="stuff-only",
            version=VERSION,  # type: ignore
            description="Stuff-only COCO-style annotations on COCO 2017 trainval",
        ),
    ]

    def _info(self) -> ds.DatasetInfo:
        if self.config.name == "stuff-thing":
            features = ds.Features(
                {
                    "image": ds.Image(),
                    "image_id": ds.Value("int32"),
                    "image_filename": ds.Value("string"),
                    "width": ds.Value("int32"),
                    "height": ds.Value("int32"),
                    "stuff_map": ds.Image(),
                    "objects": [
                        {
                            "object_id": ds.Value("string"),
                            "x": ds.Value("int32"),
                            "y": ds.Value("int32"),
                            "w": ds.Value("int32"),
                            "h": ds.Value("int32"),
                            "name": ds.Value("string"),
                        }
                    ],
                }
            )
        elif self.config.name == "stuff-only":
            features = ds.Features(
                {
                    "image": ds.Image(),
                    "image_id": ds.Value("int32"),
                    "image_filename": ds.Value("string"),
                    "width": ds.Value("int32"),
                    "height": ds.Value("int32"),
                    "objects": [
                        {
                            "object_id": ds.Value("int32"),
                            "x": ds.Value("int32"),
                            "y": ds.Value("int32"),
                            "w": ds.Value("int32"),
                            "h": ds.Value("int32"),
                            "name": ds.Value("string"),
                        }
                    ],
                }
            )
        else:
            raise ValueError(f"Invalid dataset name: {self.config.name}")

        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def load_stuff_json(self, stuff_dirpath: str, split: str):
        return _load_json(
            json_path=os.path.join(stuff_dirpath, f"stuff_{split}2017.json")
        )

    def get_image_id_to_image_infos(self, images):

        image_id_to_image_infos = {}
        for img_dict in images:
            image_id = img_dict.pop("id")
            image_id_to_image_infos[image_id] = img_dict

        image_id_to_image_infos = dict(sorted(image_id_to_image_infos.items()))
        return image_id_to_image_infos

    def get_image_id_to_annotations(self, annotations):

        image_id_to_annotations = defaultdict(list)
        for ann_dict in annotations:
            image_id = ann_dict.pop("image_id")
            image_id_to_annotations[image_id].append(ann_dict)

        image_id_to_annotations = dict(sorted(image_id_to_annotations.items()))
        return image_id_to_annotations

    def _split_generators(self, dl_manager: ds.DownloadManager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        tng_image_dirpath = os.path.join(downloaded_files["train"], "train2017")
        val_image_dirpath = os.path.join(downloaded_files["val"], "val2017")

        stuff_dirpath = downloaded_files["stuff_trainval"]
        stuff_things_maps_dirpath = downloaded_files["stuffthingmaps_trainval"]
        labels_path = downloaded_files["labels"]

        tng_gen_kwargs: GenerateExamplesArguments = {
            "image_dirpath": tng_image_dirpath,
            "stuff_dirpath": stuff_dirpath,
            "stuff_thing_maps_dirpath": stuff_things_maps_dirpath,
            "labels_path": labels_path,
            "split": "train",
        }
        val_gen_kwargs: GenerateExamplesArguments = {
            "image_dirpath": val_image_dirpath,
            "stuff_dirpath": stuff_dirpath,
            "stuff_thing_maps_dirpath": stuff_things_maps_dirpath,
            "labels_path": labels_path,
            "split": "val",
        }
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                gen_kwargs=tng_gen_kwargs,  # type: ignore
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,  # type: ignore
                gen_kwargs=val_gen_kwargs,  # type: ignore
            ),
        ]

    def _generate_examples_for_stuff_thing(
        self,
        image_dirpath: str,
        stuff_dirpath: str,
        stuff_thing_maps_dirpath: str,
        labels_path: str,
        split: str,
    ):
        id_to_label = _load_labels(labels_path=labels_path)
        stuff_json = self.load_stuff_json(stuff_dirpath=stuff_dirpath, split=split)

        image_id_to_image_infos = self.get_image_id_to_image_infos(
            images=copy.deepcopy(stuff_json["images"])
        )
        image_id_to_stuff_annotations = self.get_image_id_to_annotations(
            annotations=copy.deepcopy(stuff_json["annotations"])
        )

        assert len(image_id_to_image_infos.keys()) >= len(
            image_id_to_stuff_annotations.keys()
        )

        for image_id in image_id_to_stuff_annotations.keys():

            img_info = image_id_to_image_infos[image_id]
            image_filename = img_info["file_name"]
            image_filepath = os.path.join(image_dirpath, image_filename)
            img_example_dict = {
                "image": image_filepath,
                "image_id": image_id,
                "image_filename": image_filename,
                "width": img_info["width"],
                "height": img_info["height"],
            }

            img_anns = image_id_to_stuff_annotations[image_id]
            bboxes = [list(map(int, ann["bbox"])) for ann in img_anns]
            category_ids = [ann["category_id"] for ann in img_anns]
            category_labels = list(map(lambda cid: id_to_label[cid], category_ids))

            assert len(bboxes) == len(category_ids) == len(category_labels)
            zip_it = zip(bboxes, category_ids, category_labels)
            objects_example = [
                {
                    "object_id": category_id,
                    "x": bbox[0],
                    "y": bbox[1],
                    "w": bbox[2],
                    "h": bbox[3],
                    "name": category_label,
                }
                for bbox, category_id, category_label in zip_it
            ]

            root, _ = os.path.splitext(img_example_dict["image_filename"])
            stuff_map_filepath = os.path.join(
                stuff_thing_maps_dirpath, f"{split}2017", f"{root}.png"
            )

            example_dict = {
                **img_example_dict,
                "objects": objects_example,
                "stuff_map": stuff_map_filepath,
            }
            yield image_id, example_dict

    def _generate_examples_for_stuff_only(
        self,
        image_dirpath: str,
        stuff_dirpath: str,
        labels_path: str,
        split: str,
    ):
        id_to_label = _load_labels(labels_path=labels_path)
        stuff_json = self.load_stuff_json(stuff_dirpath=stuff_dirpath, split=split)

        image_id_to_image_infos = self.get_image_id_to_image_infos(
            images=copy.deepcopy(stuff_json["images"])
        )
        image_id_to_stuff_annotations = self.get_image_id_to_annotations(
            annotations=copy.deepcopy(stuff_json["annotations"])
        )

        assert len(image_id_to_image_infos.keys()) >= len(
            image_id_to_stuff_annotations.keys()
        )

        for image_id in image_id_to_stuff_annotations.keys():

            img_info = image_id_to_image_infos[image_id]
            image_filename = img_info["file_name"]
            image_filepath = os.path.join(image_dirpath, image_filename)
            img_example_dict = {
                "image": image_filepath,
                "image_id": image_id,
                "image_filename": image_filename,
                "width": img_info["width"],
                "height": img_info["height"],
            }

            img_anns = image_id_to_stuff_annotations[image_id]
            bboxes = [list(map(int, ann["bbox"])) for ann in img_anns]
            category_ids = [ann["category_id"] for ann in img_anns]
            category_labels = list(map(lambda cid: id_to_label[cid], category_ids))

            assert len(bboxes) == len(category_ids) == len(category_labels)
            zip_it = zip(bboxes, category_ids, category_labels)
            objects_example = [
                {
                    "object_id": category_id,
                    "x": bbox[0],
                    "y": bbox[1],
                    "w": bbox[2],
                    "h": bbox[3],
                    "name": category_label,
                }
                for bbox, category_id, category_label in zip_it
            ]

            example_dict = {
                **img_example_dict,
                "objects": objects_example,
            }
            yield image_id, example_dict

    def _generate_examples(  # type: ignore
        self,
        image_dirpath: str,
        stuff_dirpath: str,
        stuff_thing_maps_dirpath: str,
        labels_path: str,
        split: str,
    ):
        logger.info(f"Generating examples for {split}.")

        if "stuff-thing" in self.config.name:
            return self._generate_examples_for_stuff_thing(
                image_dirpath=image_dirpath,
                stuff_dirpath=stuff_dirpath,
                stuff_thing_maps_dirpath=stuff_thing_maps_dirpath,
                labels_path=labels_path,
                split=split,
            )
        elif "stuff-only" in self.config.name:
            return self._generate_examples_for_stuff_only(
                image_dirpath=image_dirpath,
                stuff_dirpath=stuff_dirpath,
                labels_path=labels_path,
                split=split,
            )
        else:
            raise ValueError(f"Invalid dataset name: {self.config.name}")
