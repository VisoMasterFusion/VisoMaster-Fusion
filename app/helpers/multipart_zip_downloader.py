import os
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from app.helpers.downloader import download_file
from app.helpers.integrity_checker import check_file_integrity


def _copy_stream(src, dst) -> None:
    for chunk in iter(lambda: src.read(1024 * 1024), b""):
        dst.write(chunk)


def download_multipart_zip_model(
    model_data: dict, skip_hash_check: bool = False
) -> bool:
    multipart_zip = model_data.get("multipart_zip")
    if not multipart_zip:
        return False

    target_path = Path(model_data["local_path"])
    target_hash = model_data["hash"]

    if target_path.is_file():
        if skip_hash_check:
            print(
                f"\n[INFO] Skipping '{model_data['model_name']}': file exists "
                "(hash check skipped - optimized models mode)."
            )
            return True
        if check_file_integrity(str(target_path), target_hash):
            print(
                f"\n[INFO] Skipping '{model_data['model_name']}': file already exists and is valid."
            )
            return True
        print(f"\n[WARN] File '{target_path}' exists but is corrupt. Rebuilding...")
        target_path.unlink()

    target_path.parent.mkdir(parents=True, exist_ok=True)

    part_paths: list[Path] = []
    for part in multipart_zip["parts"]:
        ok = download_file(
            part["model_name"],
            part["local_path"],
            part["hash"],
            part["url"],
            skip_hash_check=skip_hash_check,
        )
        if not ok:
            return False
        part_paths.append(Path(part["local_path"]))

    zip_path = target_path.with_name(f"{target_path.name}.zip.tmp")
    extracted_tmp_path = target_path.with_name(f"{target_path.name}.tmp")

    try:
        with zip_path.open("wb") as archive:
            for part_path in part_paths:
                with part_path.open("rb") as part_file:
                    _copy_stream(part_file, archive)

        zip_hash = multipart_zip.get("hash")
        if zip_hash and not check_file_integrity(str(zip_path), zip_hash):
            raise RuntimeError(f"Multipart ZIP hash mismatch for {zip_path}")

        member_name = multipart_zip.get("member", target_path.name)
        with ZipFile(zip_path) as zf:
            try:
                member_info = zf.getinfo(member_name)
            except KeyError as exc:
                raise RuntimeError(
                    f"Multipart ZIP does not contain expected member '{member_name}'"
                ) from exc
            with zf.open(member_info) as member, extracted_tmp_path.open("wb") as out:
                _copy_stream(member, out)

        if not check_file_integrity(str(extracted_tmp_path), target_hash):
            raise RuntimeError(f"Extracted model hash mismatch for {target_path}")

        os.replace(extracted_tmp_path, target_path)
        print(f"\n[INFO] Extracted '{model_data['model_name']}' to: {target_path}")
        return True
    except (BadZipFile, OSError, RuntimeError) as exc:
        print(
            f"\n[ERROR] Failed to assemble multipart ZIP for {model_data['model_name']}: {exc}"
        )
        return False
    finally:
        zip_path.unlink(missing_ok=True)
        extracted_tmp_path.unlink(missing_ok=True)
