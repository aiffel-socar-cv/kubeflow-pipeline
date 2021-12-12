import os
import json
from google.cloud import storage


def file_counter(blob):
    cnt = 0
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

    dent_images = client.list_blobs(BUCKET_NAME, prefix=os.path.join(PREFIX_1, "images"))
    dent_masks = client.list_blobs(BUCKET_NAME, prefix=os.path.join(PREFIX_1, "masks"))
    scratch_images = client.list_blobs(BUCKET_NAME, prefix=os.path.join(PREFIX_2, "images"))
    scratch_masks = client.list_blobs(BUCKET_NAME, prefix=os.path.join(PREFIX_2, "masks"))
    spacing_images = client.list_blobs(BUCKET_NAME, prefix=os.path.join(PREFIX_3, "images"))
    spacing_masks = client.list_blobs(BUCKET_NAME, prefix=os.path.join(PREFIX_3, "masks"))

    num_dent_images = file_counter(dent_images)
    num_dent_masks = file_counter(dent_masks)
    num_scratch_images = file_counter(scratch_images)
    num_scratch_masks = file_counter(scratch_masks)
    num_spacing_images = file_counter(spacing_images)
    num_spacing_masks = file_counter(spacing_masks)

    data_dict = {}

    if num_dent_images == num_dent_masks:
        data_dict["dent"] = num_dent_images
    else:
        data_dict["dent"] = -1

    if num_scratch_images == num_scratch_masks:
        data_dict["scratch"] = num_scratch_images
    else:
        data_dict["scratch"] = -1

    if num_spacing_images == num_spacing_masks:
        data_dict["spacing"] = num_spacing_images
    else:
        data_dict["spacing"] = -1

    with open("/file_nums.json", "w") as f:
        json.dump(data_dict, f)
