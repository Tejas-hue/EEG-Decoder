#!/usr/bin/env python3
import argparse
import os
import sys
import zipfile
import shutil
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from tqdm.auto import tqdm

DEFAULT_URL = "https://physionet.org/content/eegmmidb/get-zip/1.0.0/"


def stream_download(url: str, out_path: str, chunk_size: int = 1024 * 1024) -> None:
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading EEGMMI") as pbar:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
    except (URLError, HTTPError) as e:
        raise RuntimeError(f"Download failed: {e}")


def extract_zip(zip_path: str, extract_to: str) -> str:
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    return extract_to


def find_top_level_dir(extract_root: str) -> str:
    entries = [os.path.join(extract_root, e) for e in os.listdir(extract_root)]
    dirs = [e for e in entries if os.path.isdir(e)]
    # If there is a single directory, assume it is the root of extracted dataset
    if len(dirs) == 1:
        return dirs[0]
    return extract_root


def maybe_move_contents(src_root: str, dst_root: str) -> None:
    os.makedirs(dst_root, exist_ok=True)
    # Move all subject folders/files into dst_root if not already there
    for name in os.listdir(src_root):
        src = os.path.join(src_root, name)
        dst = os.path.join(dst_root, name)
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if os.path.exists(dst):
            # Skip if already exists
            continue
        shutil.move(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare PhysioNet EEG Motor Movement/Imagery dataset")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Dataset zip URL")
    parser.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "physionet_raw"), help="Target directory for extracted data")
    parser.add_argument("--keep-zip", action="store_true", help="Keep downloaded zip file")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp_dir = os.path.join(repo_root, "data", "_tmp_download")
    os.makedirs(tmp_dir, exist_ok=True)
    zip_path = os.path.join(tmp_dir, "eegmmidb_1.0.0.zip")

    print("Downloading from:", args.url)
    print("Download to:", zip_path)
    stream_download(args.url, zip_path)

    print("Extracting to:", tmp_dir)
    extracted_root = extract_zip(zip_path, tmp_dir)
    top = find_top_level_dir(extracted_root)

    print("Preparing destination:", args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    maybe_move_contents(top, args.out_dir)

    if not args.keep_zip:
        try:
            os.remove(zip_path)
        except Exception:
            pass
    # Clean empty tmp dir if possible
    try:
        for name in os.listdir(tmp_dir):
            path = os.path.join(tmp_dir, name)
            if os.path.isdir(path) and not os.listdir(path):
                os.rmdir(path)
        if not os.listdir(tmp_dir):
            os.rmdir(tmp_dir)
    except Exception:
        pass

    print("Done. EDF files should be under:", args.out_dir)
    print("Expected structure: data/physionet_raw/Sxxx/SxxxRyy.edf")
    print()
    print("Acknowledgement:")
    print("EEG Motor Movement/Imagery Dataset (Sept. 9, 2009, midnight)")
    print("Schalk et al., IEEE T-BME 51(6):1034-1043, 2004. PhysioNet citation: Goldberger et al., Circulation 101(23), e215–e220, 2000.")


if __name__ == "__main__":
    sys.exit(main())