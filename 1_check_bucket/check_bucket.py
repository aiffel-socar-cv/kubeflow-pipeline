import os
import json
from google.cloud import storage


def file_counter(blob):
    cnt = -1
    for _ in blob:
        cnt += 1
    return cnt


if __name__ == "__main__":
    BUCKET_NAME = "images-annotated"
    PREFIX_1 = "masks-dent"
    PREFIX_2 = "masks-scratch"
    PREFIX_3 = "masks-spacing"

    client = storage.Client.from_service_account_json("/.gcp/aiffel-gn-3-c8c200820331.json")
    # client = storage.Client.from_service_account_json("aiffel-gn-3-c8c200820331.json")

    dent_images = client.list_blobs(BUCKET_NAME, prefix="dent/train/images")
    dent_masks = client.list_blobs(BUCKET_NAME, prefix="dent/train/masks")
    scratch_images = client.list_blobs(BUCKET_NAME, prefix="scratch/train/images")
    scratch_masks = client.list_blobs(BUCKET_NAME, prefix="scratch/train/masks")
    spacing_images = client.list_blobs(BUCKET_NAME, prefix="spacing/train/images")
    spacing_masks = client.list_blobs(BUCKET_NAME, prefix="spacing/train/masks")

    num_dent_images = file_counter(dent_images)
    num_dent_masks = file_counter(dent_masks)
    num_scratch_images = file_counter(scratch_images)
    num_scratch_masks = file_counter(scratch_masks)
    num_spacing_images = file_counter(spacing_images)
    num_spacing_masks = file_counter(spacing_masks)

    data_dict = {}
    print(f"Dent Images: {num_dent_images}")
    print(f"Dent Masks: {num_dent_masks}")
    print(f"Scratch Images: {num_scratch_images}")
    print(f"Scratch Masks: {num_scratch_masks}")
    print(f"Spacing Images: {num_spacing_images}")
    print(f"Spacing Masks: {num_spacing_masks}")
    print("=" * 30)

    if num_dent_images == num_dent_masks:
        data_dict["dent"] = num_dent_images
    else:
        print("The number of dent images and masks are not matched.")
        data_dict["dent"] = -1

    if num_scratch_images == num_scratch_masks:
        data_dict["scratch"] = num_scratch_images
    else:
        print("The number of scratch images and masks are not matched.")
        data_dict["scratch"] = -1

    if num_spacing_images == num_spacing_masks:
        data_dict["spacing"] = num_spacing_images
    else:
        print("The number of spacing images and masks are not matched.")
        data_dict["spacing"] = -1

    with open("file_nums.json", "w") as f:
        json.dump(data_dict, f)
