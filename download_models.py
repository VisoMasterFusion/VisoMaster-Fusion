from pathlib import Path
from app.helpers.downloader import download_file
from app.processors.models_data import models_list

# When USE_OPTIMIZED_MODELS=true is set in portable.cfg (written by the
# launcher after running "Optimize Models (onnxsim)"), skip the hash check
# for existing files so optimized models are not re-downloaded.
_cfg_path = Path(__file__).resolve().parent.parent / "portable.cfg"
_skip_hash = False
if _cfg_path.is_file():
    for _line in _cfg_path.read_text(encoding="utf-8").splitlines():
        if _line.strip().upper() == "USE_OPTIMIZED_MODELS=TRUE":
            _skip_hash = True
            break

for model_data in models_list:
    download_file(
        model_data["model_name"],
        model_data["local_path"],
        model_data["hash"],
        model_data["url"],
        skip_hash_check=_skip_hash,
    )
