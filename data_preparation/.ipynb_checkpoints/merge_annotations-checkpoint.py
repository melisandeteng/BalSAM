# pip install pycocotools
import glob
import json
import os

from pycocotools.coco import COCO


def add_index_to_annots():  # only run onceà
    # add id to annotations  (some version of geodataset tilerizer did not add ids to the annotations )
    annots = glob.glob("/network/projects/trees-co2/final_tiles/*/*.json")
    for file in annots:
        with open(file, "r") as f:
            annot = json.load(f)
        for i, elem in enumerate(annot["annotations"]):
            elem["id"] = i
        with open(file, "w") as f:
            json.dump(annot, f)


def merge_coco_json(json_files, output_file):
    merged_annotations = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    image_id_offset = 0
    annotation_id_offset = 0
    category_id_offset = 0
    existing_category_ids = set()

    for idx, file in enumerate(json_files):
        coco = COCO(file)

        # Update image IDs to avoid conflicts
        for image in coco.dataset["images"]:
            image["id"] += image_id_offset
            image["file_name"] = os.path.join(
                os.path.dirname(file), "tiles", image["file_name"]
            )
            merged_annotations["images"].append(image)

        # Update annotation IDs to avoid conflicts
        for annotation in coco.dataset["annotations"]:
            annotation["id"] += annotation_id_offset
            annotation["image_id"] += image_id_offset
            merged_annotations["annotations"].append(annotation)

        # Update categories and their IDs to avoid conflicts
        for category in coco.dataset["categories"]:
            if category["id"] not in existing_category_ids:
                category["id"] += category_id_offset
                merged_annotations["categories"].append(category)
                existing_category_ids.add(category["id"])

        image_id_offset = len(merged_annotations["images"])
        annotation_id_offset = len(merged_annotations["annotations"])
        category_id_offset = len(merged_annotations["categories"])

    with open(output_file, "w") as f:
        json.dump(merged_annotations, f)


def add_dsm_info(annot_file, out_file):
    with open(annot_file, "r") as f:
        annot = json.load(f)
    for elem in annot["images"]:
        elem["dsm_path"] = (
            elem["file_name"]
            .replace("rgb", "dsm_highdis")
            .replace("final_tiles", "final_dsm_tiles_v2")
        )
    with open(out_file, "w") as fo:
        json.dump(annot, fo)


def main():
    list_train_files = glob.glob(
        "/network/projects/trees-co2/final_tiles/*/*_train.json"
    )
    list_val_files = glob.glob("/network/projects/trees-co2/final_tiles/*/*_val.json")
    list_test_files = glob.glob("/network/projects/trees-co2/final_tiles/*/*_test.json")
    merge_coco_json(
        list_train_files,
        "/network/projects/trees-co2/final_tiles/merged_annots_train_new.json",
    )
    merge_coco_json(
        list_val_files,
        "/network/projects/trees-co2/final_tiles/merged_annots_val_new.json",
    )
    merge_coco_json(
        list_test_files,
        "/network/projects/trees-co2/final_tiles/merged_annots_test_new.json",
    )

    add_dsm_info(
        "/network/projects/trees-co2/final_tiles/merged_annots_test_new.json",
        "/network/projects/trees-co2/final_tiles/merged_annots_dsm_test_new.json",
    )
    add_dsm_info(
        "/network/projects/trees-co2/final_tiles/merged_annots_train_new.json",
        "/network/projects/trees-co2/final_tiles/merged_annots_dsm_train_new.json",
    )
    add_dsm_info(
        "/network/projects/trees-co2/final_tiles/merged_annots_val_new.json",
        "/network/projects/trees-co2/final_tiles/merged_annots_dsm_val_new.json",
    )


if __name__ == "__main__":
    main()
