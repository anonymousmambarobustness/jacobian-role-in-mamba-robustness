import os
import requests
import tarfile


def download_imagenet_r(destination_dir="./imagenet-r"):
    """Downloads and extracts the ImageNet-R dataset."""
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
    tar_path = os.path.join(destination_dir, "imagenet-r.tar")

    os.makedirs(destination_dir, exist_ok=True)

    # Download the dataset
    print(f"Downloading ImageNet-R to {tar_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we stop if there's an error

    with open(tar_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)

    # Extract the dataset
    print("Extracting ImageNet-R...")
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=destination_dir)

    # Remove tar file after extraction
    os.remove(tar_path)
    print(f"ImageNet-R downloaded and extracted to {destination_dir}.")


if __name__ == "__main__":
    download_imagenet_r()
