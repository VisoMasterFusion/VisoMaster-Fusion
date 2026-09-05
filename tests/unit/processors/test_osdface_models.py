from pathlib import Path
import hashlib
import zipfile

from app.helpers import multipart_zip_downloader
from app.processors.face_restorers import FaceRestorers
from app.processors.models_data import fp16_safe_models_list, models_list
from app.ui.widgets.models_toggle_data import MODELS_TOGGLE_MAP


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def test_osdface_assets_are_registered():
    entries = {item["model_name"]: item for item in models_list}

    expected = {
        "OSDFacePromptEncoder": (
            "osdface/prompt_encoder.onnx",
            "83187cb142963151ff8abb7454e119e0a7e248e17c03c8deaa9a14bd6ba8f2a9",
            "prompt_encoder.onnx",
        ),
        "OSDFaceVAEEncoder": (
            "osdface/vae_encoder.onnx",
            "95f6d278737a864b02f99e51ac8cd00bdfb6c0b515b6d157470689fe0257dfa7",
            "vae_encoder.onnx",
        ),
        "OSDFaceUNet": (
            "osdface/unet.onnx",
            "b14bdaa36274da7f80f1a628aecbf7a9029239dad6223918432a16430e15c9e4",
            "unet.onnx",
        ),
        "OSDFaceUNetData": (
            "osdface/unet.onnx.data",
            "89a8ed18e13a5254874d567f6fc3607363af0103b249679951f167b6580bcc2c",
            "unet.onnx.data",
        ),
        "OSDFaceVAEDecoder": (
            "osdface/vae_decoder.onnx",
            "d3ac89733f86e445b3d106e801860455cb5f44f289c12cc4673c0dfccff4e051",
            "vae_decoder.onnx",
        ),
        "OSDFaceScheduler": (
            "osdface/scheduler.json",
            "fa7586cae137df656b274b2fdbfb551e95bc9fe42334354007c2ba13db6fd0c7",
            "scheduler.json",
        ),
    }

    for model_name, (local_suffix, sha256, asset_name) in expected.items():
        assert (
            entries[model_name]["local_path"].replace("\\", "/").endswith(local_suffix)
        )
        assert entries[model_name]["hash"] == sha256
        if model_name == "OSDFaceUNetData":
            assert "url" not in entries[model_name]
        else:
            assert entries[model_name]["url"] == (
                f"https://github.com/Glat0s/OSDFace-onnx/releases/download/v0.0.1/{asset_name}"
            )


def test_osdface_unet_external_data_sidecar_is_registered():
    entries = {item["model_name"]: item for item in models_list}

    assert "external_data" not in entries["OSDFaceUNet"]
    assert (
        entries["OSDFaceUNetData"]["local_path"]
        .replace("\\", "/")
        .endswith("osdface/unet.onnx.data")
    )
    assert entries["OSDFaceUNetData"]["hash"] == (
        "89a8ed18e13a5254874d567f6fc3607363af0103b249679951f167b6580bcc2c"
    )
    multipart_zip = entries["OSDFaceUNetData"]["multipart_zip"]
    assert multipart_zip["member"] == "unet.onnx.data"
    assert multipart_zip["hash"] == (
        "ced4b8a667a54f92b1ac149f8f5006c7b7eb4770d8301f82aaea34ec281b51cf"
    )
    assert [part["model_name"] for part in multipart_zip["parts"]] == [
        "OSDFaceUNetDataZip001",
        "OSDFaceUNetDataZip002",
    ]
    assert [part["hash"] for part in multipart_zip["parts"]] == [
        "e23d1a35c93ede05d00a78359381b5804368b7951596244a502dd92856e9eea3",
        "7a404682246d84ed3d5f0e3ab8b24f062bc3d723c79e701bcec440473fb4b9d1",
    ]


def test_osdface_is_exposed_as_face_restorer():
    layout_source = Path("app/ui/widgets/common_layout_data.py").read_text(
        encoding="utf-8"
    )

    assert '"requiredSelectionValue": "OSDFace"' in layout_source
    assert '"OSDFaceTimestepSlider"' in layout_source
    assert '"OSDFaceLatentStrengthDecimalSlider"' in layout_source
    assert '"OSDFaceTimestep2Slider"' in layout_source
    assert '"OSDFaceLatentStrength2DecimalSlider"' in layout_source
    assert "OSDFacePromptEncoder" in MODELS_TOGGLE_MAP
    assert "OSDFacePromptEncoder" in fp16_safe_models_list
    assert "OSDFaceVAEEncoder" in fp16_safe_models_list
    assert "OSDFaceUNet" not in fp16_safe_models_list
    assert "OSDFaceVAEDecoder" not in fp16_safe_models_list
    assert FaceRestorers.osdface_model_names == (
        "OSDFacePromptEncoder",
        "OSDFaceVAEEncoder",
        "OSDFaceUNet",
        "OSDFaceVAEDecoder",
    )


def test_multipart_zip_helper_reconstructs_and_extracts_model(tmp_path, monkeypatch):
    source_model = tmp_path / "unet.onnx.data"
    source_model.write_bytes(b"osdface model sidecar")

    archive_path = tmp_path / "unet.onnx.data.zip"
    with zipfile.ZipFile(archive_path, "w", allowZip64=True) as zf:
        zf.write(source_model, arcname="unet.onnx.data")

    archive_bytes = archive_path.read_bytes()
    part_1 = tmp_path / "unet.onnx.data.zip.001"
    part_2 = tmp_path / "unet.onnx.data.zip.002"
    midpoint = len(archive_bytes) // 2
    part_1.write_bytes(archive_bytes[:midpoint])
    part_2.write_bytes(archive_bytes[midpoint:])
    source_model.unlink()

    def fake_download_file(*args, **kwargs):
        return True

    monkeypatch.setattr(multipart_zip_downloader, "download_file", fake_download_file)

    model_data = {
        "model_name": "OSDFaceUNetData",
        "local_path": str(source_model),
        "hash": hashlib.sha256(b"osdface model sidecar").hexdigest(),
        "multipart_zip": {
            "member": "unet.onnx.data",
            "hash": _sha256(archive_path),
            "parts": [
                {
                    "model_name": "OSDFaceUNetDataZip001",
                    "local_path": str(part_1),
                    "hash": _sha256(part_1),
                    "url": "https://example.invalid/001",
                },
                {
                    "model_name": "OSDFaceUNetDataZip002",
                    "local_path": str(part_2),
                    "hash": _sha256(part_2),
                    "url": "https://example.invalid/002",
                },
            ],
        },
    }

    assert multipart_zip_downloader.download_multipart_zip_model(model_data)
    assert source_model.read_bytes() == b"osdface model sidecar"
