from app.helpers.downloader import download_file
from app.processors.models_data import models_list

for model_data in models_list:
    download_file(model_data['model_name'], model_data['local_path'], model_data['hash'], model_data['url'])

# Download ReF-LDM models
print("\nChecking for ReF-LDM models...")
import os
from app.helpers.miscellaneous import is_file_exists

base_path = "model_assets/ref-ldm_embedding"
configs_path = os.path.join(base_path, "configs")
ckpts_path = os.path.join(base_path, "ckpts")
os.makedirs(configs_path, exist_ok=True)
os.makedirs(ckpts_path, exist_ok=True)

ref_ldm_files = {
    "configs/ldm.yaml": "https://raw.githubusercontent.com/Glat0s/ref-ldm-onnx/slim-fast/configs/ldm.yaml",
    "configs/refldm.yaml": "https://raw.githubusercontent.com/Glat0s/ref-ldm-onnx/slim-fast/configs/refldm.yaml",
    "configs/vqgan.yaml": "https://raw.githubusercontent.com/Glat0s/ref-ldm-onnx/slim-fast/configs/vqgan.yaml",
    "ckpts/refldm.ckpt": "https://github.com/ChiWeiHsiao/ref-ldm/releases/download/1.0.0/refldm.ckpt",
    "ckpts/vqgan.ckpt": "https://github.com/ChiWeiHsiao/ref-ldm/releases/download/1.0.0/vqgan.ckpt",
}

for rel_path, url in ref_ldm_files.items():
    full_path = os.path.join(base_path, rel_path)
    if not is_file_exists(full_path):
        print(f"Downloading ReF-LDM file: {os.path.basename(full_path)}...")
        download_file(os.path.basename(full_path), full_path, None, url)
    else:
        print(f"ReF-LDM file already exists: {os.path.basename(full_path)}")

print("ReF-LDM model check complete.")