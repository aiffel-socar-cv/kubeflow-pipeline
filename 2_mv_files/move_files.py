import os
import argparse
import json
from io import StringIO
from google.cloud import storage


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument("--json_file", type=str, help="Input JSON file path")

    args = argument_parser.parse_args()
    json_file = args.json_file
    print("JSON FILE: ", json_file)
    print("JSON FILE TYPE: ", type(json_file))  # str type

    file_num_dict = eval(json_file)  # str to dict

    print("=" * 30)
    print("FILE NUM DICT: ", file_num_dict)
    print("FILE NUM TYPE: ", type(file_num_dict))  # dict type

    # if file_num["dent"] > 100:
    #     print("Dent files are moved.")
    # if file_num["scratch"] > 100:
    #     print("Scratch files are moved.")
    # if file_num["spacing"] > 100:
    #     print("Spacing files are moved.")
