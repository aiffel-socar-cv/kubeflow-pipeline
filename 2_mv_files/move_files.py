import os
import argparse
import json
from google.cloud import storage


def mv_blob(bucket_name, blob_name, new_bucket_name, new_blob_name):
    storage_client = storage.Client.from_service_account_json("/.gcp/aiffel-gn-3-c8c200820331.json")
    # storage_client = storage.Client.from_service_account_json("aiffel-gn-3-c8c200820331.json")
    source_bucket = storage_client.get_bucket(bucket_name)
    source_blob = source_bucket.blob(blob_name)
    destination_bucket = storage_client.get_bucket(new_bucket_name)

    # copy to new destination
    new_blob = source_bucket.copy_blob(source_blob, destination_bucket, new_blob_name)

    # delete in old destination
    source_blob.delete()

    print(f"File moved from {source_blob} to {new_blob_name}")


def list_blobs(bucket_name, prefix):
    """return blob names in the bucket"""

    storage_client = storage.Client.from_service_account_json("/.gcp/aiffel-gn-3-c8c200820331.json")
    # storage_client = storage.Client.from_service_account_json("aiffel-gn-3-c8c200820331.json")

    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    blob_list = []
    for blob in blobs:
        file_name, ext = os.path.splitext(blob.name)
        if ext in [".jpg", ".jpeg", ".png"]:
            blob_list.append(blob.name)

    return blob_list


def move_files(from_bucket, to_bucket, data_type):
    image_list = list_blobs(from_bucket, f"{data_type}/train/images")
    mask_list = list_blobs(from_bucket, f"{data_type}/train/masks")

    for image, mask in zip(image_list, mask_list):
        mv_blob(from_bucket, image, to_bucket, image)
        mv_blob(from_bucket, mask, to_bucket, mask)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument("--json_file", type=str, help="Input JSON file path")

    args = argument_parser.parse_args()
    json_file = args.json_file

    file_num_dict = eval(json_file)  # str to dict

    print("FILE NUM DICT: ", file_num_dict)
    print("FILE NUM TYPE: ", type(file_num_dict))  # dict type
    print("=" * 30)

    data_status_dict = {"dent": False, "scratch": False, "spacing": False}

    INFERRED_BUCKET = "images-annotated"
    RETRAIN_BUCKET = "images-retrain"
    THRESHOLD = 10

    if file_num_dict["dent"] > THRESHOLD:
        move_files(INFERRED_BUCKET, RETRAIN_BUCKET, "dent")
        print("Dent files are moved.")
        data_status_dict["dent"] = True
    if file_num_dict["scratch"] > THRESHOLD:
        move_files(INFERRED_BUCKET, RETRAIN_BUCKET, "scratch")
        print("Scratch files are moved.")
        data_status_dict["scratch"] = True
    if file_num_dict["spacing"] > THRESHOLD:
        move_files(INFERRED_BUCKET, RETRAIN_BUCKET, "spacing")
        print("Spacing files are moved.")
        data_status_dict["spacing"] = True

    with open("data_status.json", "w") as f:
        json.dump(data_status_dict, f)
