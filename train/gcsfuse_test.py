import os
import subprocess
import time


class Colors:
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("CREATED FOLDER: ", Colors.GREEN, directory, Colors.RESET)
        else:  # TODO: delete already exist folder
            print(Colors.RED, "The directory already exists!!!", Colors.RESET)
    except OSError:
        print(Colors.RED, "Craeting directory failed.", Colors.RESET)


def main():
    WORK_DIR = os.getcwd()

    CREDENTIAL_FILE = (
        "/home/t1won/Documents/Github/kubeflow-pipeline/aiffel-gn-3-c8c200820331.json"
    )
    IMAGE_FOLDER = os.path.join(WORK_DIR, "images")
    MASK_FOLDER = os.path.join(WORK_DIR, "masks")
    WRITE_FOLDER = os.path.join(WORK_DIR, "write_folder")
    BUCKET_NAME = "test_bucket_aiffel"

    create_folder(IMAGE_FOLDER)
    create_folder(MASK_FOLDER)
    create_folder(WRITE_FOLDER)

    # export credential file
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIAL_FILE

    # mount
    subprocess.run(["gcsfuse", "--only-dir", "train/images", BUCKET_NAME, IMAGE_FOLDER])
    subprocess.run(["gcsfuse", "--only-dir", "train/masks", BUCKET_NAME, MASK_FOLDER])
    subprocess.run(
        ["gcsfuse", "--only-dir", "train/write_folder", BUCKET_NAME, WRITE_FOLDER]
    )

    # work
    # print(os.listdir(os.path.join(TEST_FOLDER, "dent_images")))
    for root, dirnames, filenames in os.walk(IMAGE_FOLDER):
        print("IMAGE", len(filenames))
    for root, dirnames, filenames in os.walk(MASK_FOLDER):
        print("MASK", len(filenames))

    with open(os.path.join(WRITE_FOLDER, "test.txt"), "w") as f:
        f.write("TEST MESSAGE")

    # unmount
    time.sleep(10)  # wait

    def unmount_fuse(folder):
        ret_code = 1
        while ret_code == 1:
            ret = subprocess.run(["fusermount", "-u", folder])
            ret_code = ret.returncode
        print(Colors.GREEN, f"Unmount f{folder} done!", Colors.RESET)

    folders = [IMAGE_FOLDER, MASK_FOLDER, WRITE_FOLDER]
    for f in folders:
        unmount_fuse(f)


if __name__ == "__main__":
    main()
