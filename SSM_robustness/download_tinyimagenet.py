import os
import urllib.request
import zipfile
def download_tinyimagenet(destination="datasets"):
    dataset_dir = os.path.join(destination, "tiny-imagenet-200")
    zip_path = os.path.join(destination, "tiny-imagenet-200.zip")
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    if not os.path.exists(dataset_dir):
        os.makedirs(destination, exist_ok=True)
        print("Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(url, zip_path)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(destination)

        os.remove(zip_path)
        print(f"Tiny ImageNet downloaded and extracted to {dataset_dir}")
    else:
        print("Tiny ImageNet dataset already exists.")

download_tinyimagenet()