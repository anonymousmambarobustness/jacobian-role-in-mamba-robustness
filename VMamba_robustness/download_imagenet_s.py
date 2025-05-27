import os
import tempfile
import gdown
import zipfile
from tqdm import tqdm


def download_imagenet_s(destination_dir="imagenet-s"):
    """Download with temp directory for cookies"""
    # Set temp directory for cookies
    temp_cache = os.path.join(tempfile.gettempdir(), "gdown_cache")
    os.makedirs(temp_cache, exist_ok=True)
    os.environ["GDOWN_CACHE_DIR"] = temp_cache

    # Google Drive file ID
    file_id = "1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA"
    url = f"https://drive.google.com/uc?id={file_id}"
    zip_path = os.path.join(destination_dir, "imagenet-s.zip")

    os.makedirs(destination_dir, exist_ok=True)

    print("Downloading ImageNet-S...")
    try:
        gdown.download(url, zip_path, quiet=False)
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying direct download...")
        return direct_download(destination_dir)

    verify_and_extract(zip_path, destination_dir)


def direct_download(destination_dir):
    """Fallback direct download method"""
    import requests
    url = "https://drive.usercontent.google.com/download?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA&export=download&confirm=t"
    zip_path = os.path.join(destination_dir, "imagenet-s.zip")

    print("Downloading directly...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(zip_path, 'wb') as f, tqdm(
            unit='B', unit_scale=True,
            total=int(response.headers.get('content-length', 0))
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    verify_and_extract(zip_path, destination_dir)


def verify_and_extract(zip_path, destination_dir):
    """Verify and extract the dataset"""
    print("Verifying...")
    if not zipfile.is_zipfile(zip_path):
        os.remove(zip_path)
        raise ValueError("Invalid zip file")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc='Extracting'):
            zip_ref.extract(member, destination_dir)

    os.remove(zip_path)
    print(f"Dataset ready at: {destination_dir}")


if __name__ == "__main__":
    download_imagenet_s()