import os
import argparse
import json
from io import StringIO
from google.cloud import storage


def json2dict(json_file_path):
    # json_file_path = StringIO(json_file_path)
    # with open(json_file_path, "r") as f:
    file_num_dict = json.load(json_file_path)
    return file_num_dict


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument("--json_path", type=str, help="Input JSON file path")

    args = argument_parser.parse_args()
    json_file_path = args.json_path
    print("JSON FILE PATH: ", json_file_path)

    if json_file_path:
        file_num = json2dict(json_file_path)

    print("=" * 30)
    print(file_num)

    if file_num["dent"] > 100:
        print("Dent files are moved.")
    if file_num["scratch"] > 100:
        print("Scratch files are moved.")
    if file_num["spacing"] > 100:
        print("Spacing files are moved.")
