import os
import argparse
import json
from google.cloud import storage


def mv_blob(bucket_name, blob_name, new_bucket_name, new_blob_name):
    """
    Function for moving files between directories or buckets. it will use GCP's copy
    function then delete the blob from the old location.

    inputs
    -----
    bucket_name: name of bucket
    blob_name: str, name of file
        ex. 'data/some_location/file_name'
    new_bucket_name: name of bucket (can be same as original if we're just moving around directories)
    new_blob_name: str, name of file in new directory in target bucket
        ex. 'data/destination/file_name'
    """
    storage_client = storage.Client.from_service_account_json("aiffel-gn-3-c8c200820331.json")
    source_bucket = storage_client.get_bucket(bucket_name)
    source_blob = source_bucket.blob(blob_name)
    destination_bucket = storage_client.get_bucket(new_bucket_name)

    # copy to new destination
    new_blob = source_bucket.copy_blob(source_blob, destination_bucket, new_blob_name)
    # delete in old destination
    source_blob.delete()

    print(f"File moved from {source_blob} to {new_blob_name}")


def list_blobs(bucket_name):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"

    # storage_client = storage.Client()
    storage_client = storage.Client.from_service_account_json("aiffel-gn-3-c8c200820331.json")

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix="dent/images")

    for blob in blobs:
        print(blob.name)


if __name__ == "__main__":
    # argument_parser = argparse.ArgumentParser()

    # argument_parser.add_argument("--json_file", type=str, help="Input JSON file path")

    # args = argument_parser.parse_args()
    # json_file = args.json_file
    # # print("JSON FILE: ", json_file)
    # # print("JSON FILE TYPE: ", type(json_file))  # str type

    # file_num_dict = eval(json_file)  # str to dict

    # print("FILE NUM DICT: ", file_num_dict)
    # print("FILE NUM TYPE: ", type(file_num_dict))  # dict type
    # print("=" * 30)

    # ret_dict = {"dent": False, "scratch": False, "spacing": False}

    INFERRED_BUCKET = "images-annotated"
    RETRAIN_BUCKET = "images-retrain"

    # if file_num_dict["dent"] > 10:

    #     print("Dent files are moved.")
    # if file_num_dict["scratch"] > 10:
    #     print("Scratch files are moved.")
    # if file_num_dict["spacing"] > 10:
    #     print("Spacing files are moved.")
    # mv_blob(INFERRED_BUCKET, "dent/images", RETRAIN_BUCKET, "dent/images")
    list_blobs(INFERRED_BUCKET)
