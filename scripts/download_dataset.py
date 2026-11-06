# -*- coding: utf-8 -*-
"""
================================================================================
Bovine Pose Estimation — Reproducibility Audit & Data Provenance Benchmark
================================================================================
File:         scripts/download_dataset.py
Description:  Automated dataset downloader and extractor for Dairy Campus
              bovine pose estimation dataset from Zenodo (10.5281/zenodo.23931569).
Author:       Bert Klandermans (Data Scientist)
              ORCID: https://orcid.org/0009-0009-2173-9422
Project Lead: Dr. Claudia Kamphuis
              ORCID: https://orcid.org/0000-0001-5552-036X
Affiliation:  Animal Breeding and Genomics, Wageningen Livestock Research (WUR)
Facility:     Dairy Campus, Leeuwarden, the Netherlands
Dataset DOI:  https://doi.org/10.5281/zenodo.23931569
Repository:   https://github.com/dairycampus/
              dairycampus-cow-pose-estimation-klandermans
License:      CC-BY-4.0 / Apache-2.0
Standards:    Black-formatted, PEP 8 / Flake8 compliant
================================================================================
"""

import argparse
import hashlib
import os
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from typing import Dict, Optional

ZENODO_RECORD_ID: str = "23931569"
ZENODO_PREVIEW_URL: str = f"https://zenodo.org/records/{ZENODO_RECORD_ID}?preview=1"
EXPECTED_MD5: str = "1c23da9befeb16e772d6076adaf1f1c2"
EXPECTED_SIZE: int = 209056207  # 209 MB


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for dataset downloader."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    parser = argparse.ArgumentParser(
        description=(
            "Download and extract Dairy Campus Pose Estimation Dataset " "from Zenodo"
        )
    )
    parser.add_argument(
        "--record-id",
        type=str,
        default=ZENODO_RECORD_ID,
        help="Zenodo Record ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(repo_root, "data"),
        help="Target data directory",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("ZENODO_TOKEN", None),
        help="Zenodo Personal Access Token (required while record is draft)",
    )
    parser.add_argument(
        "--zip-file",
        type=str,
        default=None,
        help="Path to locally downloaded Archive.zip",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip MD5 integrity check",
    )
    parser.add_argument(
        "--prepare-combined",
        action="store_true",
        help="Automatically prepare 320-image combined evaluation dataset",
    )
    return parser.parse_args()


def verify_md5(file_path: str, expected_md5: str) -> bool:
    """Verify MD5 checksum of a file."""
    print(f"Verifying MD5 checksum for {file_path}...")
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    if actual == expected_md5:
        print(f"✓ Checksum verified: {actual}")
        return True
    print(
        f"Warning: MD5 mismatch!\n"
        f"  Expected: {expected_md5}\n"
        f"  Actual:   {actual}"
    )
    return False


def download_with_progress(
    url: str, dest_path: str, headers: Optional[Dict[str, str]] = None
) -> None:
    """Download a remote URL to local file with a terminal progress bar."""
    headers = headers or {}
    req = urllib.request.Request(url, headers=headers)
    print(f"Connecting to: {url}")

    try:
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB

            with open(dest_path, "wb") as out_f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        mb_done = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        sys.stdout.write(
                            f"\rDownloading: {mb_done:.1f}/{mb_total:.1f} MB "
                            f"[{pct:.1f}%]"
                        )
                        sys.stdout.flush()
            print("\nDownload finished.")
    except urllib.error.HTTPError as err:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise err


def extract_archive(zip_path: str, extract_to: str) -> None:
    """Extract zip archive while ignoring macOS resource fork metadata."""
    print(f"Extracting {zip_path} into {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        # Ignore macOS resource fork files during extraction
        members = [
            m
            for m in z.namelist()
            if not m.startswith("__MACOSX/")
            and not os.path.basename(m).startswith("._")
        ]
        total = len(members)
        for i, member in enumerate(members):
            z.extract(member, extract_to)
            if i % 100 == 0 or i == total - 1:
                sys.stdout.write(f"\rUnpacking: {i+1}/{total} files")
                sys.stdout.flush()
    print("\nExtraction complete.")


def run_prepare_combined(dataset_root: str, output_dir: str) -> None:
    """Run prepare_combined_dataset.py script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prep_script = os.path.join(script_dir, "prepare_combined_dataset.py")
    if os.path.exists(prep_script):
        print("\nBuilding 320-image combined dataset...")
        cmd = [
            sys.executable,
            prep_script,
            "--dataset-root",
            dataset_root,
            "--output-dir",
            output_dir,
        ]
        subprocess.run(cmd, check=True)


def main() -> None:
    """Main execution entry point."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    expected_kpt_dir = os.path.join(args.output_dir, "cow_barn_kpt_it2_corr_300_20")
    train_ann = os.path.join(
        expected_kpt_dir, "train", "annotations", "instances_default.json"
    )
    if os.path.exists(train_ann):
        print(f"Dataset already present in {expected_kpt_dir}.")
        if args.prepare_combined:
            run_prepare_combined(
                expected_kpt_dir, os.path.join(args.output_dir, "cow_kpt_320")
            )
        return

    local_zip = args.zip_file
    default_zip_target = os.path.join(args.output_dir, "Archive.zip")

    if not local_zip and os.path.exists(default_zip_target):
        local_zip = default_zip_target
        print(f"Found existing archive at {local_zip}")

    if not local_zip or not os.path.exists(local_zip):
        download_success = False

        # 1. Try public Zenodo endpoint (works when record is published)
        base_zenodo = f"https://zenodo.org/records/{args.record_id}/files"
        base_api = f"https://zenodo.org/api/records/{args.record_id}/files"
        public_urls = [
            f"{base_zenodo}/Archive.zip/content",
            f"{base_api}/Archive.zip/content",
        ]

        for url in public_urls:
            try:
                print("Attempting public download from Zenodo...")
                download_with_progress(url, default_zip_target)
                download_success = True
                local_zip = default_zip_target
                break
            except urllib.error.HTTPError as err:
                if err.code in (403, 404):
                    continue
                print(f"HTTP error during download: {err}")

        # 2. Try draft endpoint with token if public failed
        if not download_success:
            if args.token:
                draft_url = (
                    f"https://zenodo.org/api/records/{args.record_id}"
                    "/draft/files/Archive.zip/content"
                )
                headers = {"Authorization": f"Bearer {args.token}"}
                try:
                    print("Attempting draft download with Zenodo Access Token...")
                    download_with_progress(
                        draft_url, default_zip_target, headers=headers
                    )
                    download_success = True
                    local_zip = default_zip_target
                except urllib.error.HTTPError as err:
                    print(f"Draft API error: {err}")
            else:
                print("\n" + "=" * 70)
                print("   ZENODO RECORD IN PREVIEW / RESTRICTED MODE")
                print("=" * 70)
                print(
                    f"The Zenodo dataset record ({args.record_id}) is "
                    "currently in preview/draft mode:"
                )
                print(f"  {ZENODO_PREVIEW_URL}\n")
                print(
                    "To extract and prepare the dataset, choose one of "
                    "these options:"
                )
                print(
                    "1. In your browser, open the preview link above while "
                    "logged in,"
                )
                print("   download 'Archive.zip', and run:")
                print(
                    "   python3 scripts/download_dataset.py --zip-file "
                    "/path/to/Archive.zip\n"
                )
                print("2. Provide your Zenodo Personal Access Token:")
                print(
                    "   python3 scripts/download_dataset.py --token "
                    "<YOUR_ZENODO_TOKEN>\n"
                )
                print(
                    "3. Once the record is officially published on Zenodo, " "running:"
                )
                print("   python3 scripts/download_dataset.py")
                print(
                    "   will automatically download and unpack the dataset "
                    "publicly without tokens."
                )
                print("=" * 70 + "\n")
                sys.exit(1)

    # Check integrity
    if not args.skip_checksum:
        verify_md5(local_zip, EXPECTED_MD5)

    # Unpack archive
    extract_archive(local_zip, args.output_dir)

    # Auto-adjust structure if Archive.zip extracts under a specific subfolder
    if os.path.exists(expected_kpt_dir):
        print(f"✓ Keypoint dataset structure ready at: {expected_kpt_dir}")
    else:
        print(f"Notice: Archive unpacked into {args.output_dir}.")

    if args.prepare_combined:
        run_prepare_combined(
            expected_kpt_dir, os.path.join(args.output_dir, "cow_kpt_320")
        )


if __name__ == "__main__":
    main()
