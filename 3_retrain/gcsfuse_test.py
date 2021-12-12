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
            print("CREATING FOLDER: ", Colors.GREEN, directory, Colors.RESET)
            os.makedirs(directory)
        else:
            print(Colors.RED, "The directory already exists!!!", Colors.RESET)
            os.rmdir(directory)
            print("RECREATING FOLDER: ", Colors.GREEN, directory, Colors.RESET)
            os.makedirs(directory)

    except OSError:
        print(Colors.RED, "Creating  directory failed.", Colors.RESET)


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
            if ret_code == 1:
                print(Colors.RED, "Retry unmount", Colors.RESET)
                time.sleep(1)
        print(Colors.GREEN, f"Unmount f{folder} done!", Colors.RESET)

    folders = [IMAGE_FOLDER, MASK_FOLDER, WRITE_FOLDER]
    for f in folders:
        unmount_fuse(f)


if __name__ == "__main__":
    main()
