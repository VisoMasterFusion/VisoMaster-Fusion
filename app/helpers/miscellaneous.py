import os
import shutil
import cv2
import time
from bisect import bisect_left, bisect_right
from collections import UserDict, OrderedDict
import hashlib
import numpy as np
from functools import wraps
from datetime import datetime
from pathlib import Path
from torchvision.transforms import v2
from typing import Dict, Mapping, Tuple, Optional, Any, Collection, Sequence
import threading
import subprocess
import json

import torch
from PIL import Image
from skimage import transform as trans

# --- Global Scope ---

# Scaling transforms cache — bounded LRU so long sessions with many interpolation
# setting changes cannot grow this indefinitely.  In practice 2–5 entries are used.
_transform_cache: OrderedDict = OrderedDict()
_TRANSFORM_CACHE_MAX = 32
image_extensions = (
    ".jpg",
    ".jpeg",
    ".jpe",
    ".png",
    ".webp",
    ".tif",
    ".tiff",
    ".jp2",
    ".exr",
    ".hdr",
    ".ras",
    ".pnm",
    ".ppm",
    ".pgm",
    ".pbm",
    ".pfm",
)
video_extensions = (
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".3gp",
    ".gif",
)

# --- Class Definitions ---


class ThumbnailManager:
    """
    Manages the creation, storage, and retrieval of media file thumbnails.

    This class encapsulates all thumbnail-related logic, such as hashing filenames,
    managing the thumbnail storage directory, and generating thumbnail images from
    video frames or images.
    """

    def __init__(self, thumbnail_dir: str = ".thumbnails"):
        """
        Initializes the ThumbnailManager.

        Args:
            thumbnail_dir (str): The name of the directory to store thumbnails,
                                 created in the current working directory.
        """
        self.thumbnail_dir = os.path.join(os.getcwd(), thumbnail_dir)
        self._lock = threading.Lock()
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """
        Ensures that the thumbnail storage directory exists.
        This is a private method called during initialization.
        """
        os.makedirs(self.thumbnail_dir, exist_ok=True)

    def _get_file_hash(self, file_path: str) -> str:
        """
        Generates a unique hash for a file based on its name and size.

        Args:
            file_path (str): The absolute path to the file.

        Returns:
            str: A unique MD5 hash string for the file.
        """
        name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        hash_input = f"{name}_{file_size}"
        return hashlib.md5(hash_input.encode("utf-8")).hexdigest()

    def get_thumbnail_path(self, file_path: str) -> Tuple[str, str]:
        """
        Generates the potential paths for a thumbnail (PNG and JPG).

        Args:
            file_path (str): The path to the original media file.

        Returns:
            tuple[str, str]: A tuple containing the ideal PNG path and the fallback JPG path.
        """
        file_hash = self._get_file_hash(file_path)
        png_path = os.path.join(self.thumbnail_dir, f"{file_hash}.png")
        jpg_path = os.path.join(self.thumbnail_dir, f"{file_hash}.jpg")
        return png_path, jpg_path

    def find_existing_thumbnail(self, file_path: str) -> str | None:
        """
        Checks for an existing thumbnail file (PNG or JPG) and returns its path.

        Args:
            file_path (str): The path to the original media file.

        Returns:
            str | None: The path to the existing thumbnail, or None if it doesn't exist.
        """
        png_path, jpg_path = self.get_thumbnail_path(file_path)
        with self._lock:
            if os.path.exists(png_path):
                return png_path
            if os.path.exists(jpg_path):
                return jpg_path
        return None

    def create_thumbnail(self, frame: np.ndarray, file_path: str) -> None:
        """
        Saves a given frame as an optimized thumbnail image.

        It tries to save as a high-quality PNG. If the PNG is too large,
        it falls back to an optimized JPEG.

        Args:
            frame (np.ndarray): The image frame (from OpenCV) to save.
            file_path (str): The path of the *original media file* to generate the thumbnail name.
        """
        png_path, jpg_path = self.get_thumbnail_path(file_path)

        # Color format conversion to avoid errors
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        height, width, _ = frame.shape
        width, height = get_scaled_resolution(
            media_width=width, media_height=height, max_height=140, max_width=140
        )

        resized_frame = cv2.resize(
            frame, (width, height), interpolation=cv2.INTER_LANCZOS4
        )

        try:
            with self._lock:
                cv2.imwrite(png_path, resized_frame)
            if os.path.getsize(png_path) > 30 * 1024:  # If PNG is > 30KB
                os.remove(png_path)
                raise Exception("PNG file too large, falling back to JPEG.")
        except Exception:
            jpeg_params = [
                cv2.IMWRITE_JPEG_QUALITY,
                98,
                cv2.IMWRITE_JPEG_OPTIMIZE,
                1,
                cv2.IMWRITE_JPEG_PROGRESSIVE,
                1,
            ]
            with self._lock:
                cv2.imwrite(jpg_path, resized_frame, jpeg_params)


class DFMModelManager:
    """
    Manages the discovery and retrieval of DeepFace Model (DFM) files.

    This class scans a specified directory for .dfm and .onnx model files,
    making them available for use in the application, for example, in UI dropdowns.
    """

    def __init__(self, models_path: str = "./model_assets/dfm_models"):
        """
        Initializes the DFMModelManager.

        Args:
            models_path (str): The path to the directory containing DFM model files.
        """
        self.models_path = models_path
        self.models_data: Dict[str, str] = {}
        self.refresh_models()

    def refresh_models(self) -> None:
        """
        Scans the model directory and updates the internal dictionary of found models.
        """
        self.models_data.clear()
        if not os.path.isdir(self.models_path):
            print(f"[WARN] DFM models directory not found at: {self.models_path}")
            return

        for dfm_file in os.listdir(self.models_path):
            if dfm_file.endswith((".dfm", ".onnx")):
                self.models_data[dfm_file] = os.path.join(self.models_path, dfm_file)

    def get_models_data(self) -> dict:
        """Returns the dictionary mapping model filenames to their full paths."""
        return self.models_data

    def get_selection_values(self) -> list:
        """Returns a list of model filenames for use in selection widgets."""
        return list(self.models_data.keys())

    def get_default_value(self) -> str:
        """Returns the filename of the first model found, or an empty string."""
        dfm_values = self.get_selection_values()
        return dfm_values[0] if dfm_values else ""


# Datatype used for storing parameter values
# Major use case for subclassing this is to fallback to a default value, when trying to access value from a non-existing key
# Helps when saving/importing workspace or parameters from external file after a future update including new Parameter widgets
class ParametersDict(UserDict):
    def __init__(self, parameters, default_parameters: dict):
        super().__init__(parameters)
        self._default_parameters = default_parameters

    def __getitem__(self, key):
        try:
            return self.data[key]
        except KeyError:
            self.__setitem__(key, self._default_parameters[key])
            return self._default_parameters[key]


def copy_mapping_data(value: object) -> dict[str, Any]:
    """Return a plain dict copy when *value* is any mapping-like object.

    This preserves `ParametersDict` and other `Mapping` subclasses while keeping
    non-mapping inputs on a safe empty-dict fallback.
    """
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def is_detected_face_eligible_for_matching(
    kps: np.ndarray | None,
    bbox: np.ndarray | None,
    min_face_pixels: int,
) -> bool:
    """Return True when a detected face is valid enough for matching.

    This mirrors the standard frame-worker gate used before recognition and
    matching: keypoints must exist and be finite, and the shortest bbox side
    must meet the minimum face-size threshold.
    """
    if kps is None or kps.size == 0:
        return False
    if np.any(np.isnan(kps)) or np.any(np.isinf(kps)):
        return False
    if bbox is None or bbox.size < 4:
        return False

    shortest_side = min(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]))
    return shortest_side >= float(min_face_pixels)


def find_best_target_match(
    detected_embedding: np.ndarray,
    models_processor: Any,
    target_faces: Mapping[object, Any],
    face_parameters: Mapping[str, object],
    default_params: Mapping[str, Any],
    recognition_model: str,
) -> tuple[Any | None, ParametersDict | None, float]:
    """Return the best matching target face for a detected embedding.

    The caller supplies the current face-parameter mapping and default params so
    this helper can apply the same per-face threshold logic everywhere it is
    used, including playback/render and issue scans.
    """
    best_target = None
    best_params_pd = None
    highest_sim = -1.0
    default_params_dict = dict(default_params)

    for target_id, target_face in target_faces.items():
        face_id_str = str(getattr(target_face, "face_id", target_id))
        face_specific_params = copy_mapping_data(face_parameters.get(face_id_str))
        current_params_pd = ParametersDict(face_specific_params, default_params_dict)
        target_embedding = target_face.get_embedding(recognition_model)
        if not isinstance(target_embedding, np.ndarray) or target_embedding.size == 0:
            continue

        sim = models_processor.findCosineDistance(detected_embedding, target_embedding)
        if sim >= current_params_pd["SimilarityThresholdSlider"] and sim > highest_sim:
            highest_sim = sim
            best_target = target_face
            best_params_pd = current_params_pd

    return best_target, best_params_pd, highest_sim


def count_issue_scan_frames(
    scan_ranges: Sequence[tuple[int, int]],
    dropped_frames: Collection[int],
) -> int:
    """Count scan frames after excluding dropped render frames.

    This keeps issue-scan progress and summary stats aligned with the frames that
    render/output will actually keep.
    """
    normalized_ranges = normalize_issue_scan_ranges(scan_ranges)
    normalized_dropped = sorted({int(frame) for frame in dropped_frames})
    total_frames = 0

    for start_frame, end_frame in normalized_ranges:
        normalized_start = int(start_frame)
        normalized_end = int(end_frame)
        if normalized_end < normalized_start:
            continue

        dropped_in_range = bisect_right(
            normalized_dropped, normalized_end
        ) - bisect_left(normalized_dropped, normalized_start)
        total_frames += (normalized_end - normalized_start + 1) - dropped_in_range

    return total_frames


def normalize_issue_scan_ranges(
    scan_ranges: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return chronologically sorted, overlap-merged scan ranges."""
    normalized: list[tuple[int, int]] = []

    for start_frame, end_frame in scan_ranges:
        normalized_start = int(start_frame)
        normalized_end = int(end_frame)
        if normalized_end < normalized_start:
            continue
        normalized.append((normalized_start, normalized_end))

    if not normalized:
        return []

    normalized.sort()
    merged_ranges: list[tuple[int, int]] = [normalized[0]]

    for start_frame, end_frame in normalized[1:]:
        previous_start, previous_end = merged_ranges[-1]
        if start_frame <= previous_end:
            merged_ranges[-1] = (previous_start, max(previous_end, end_frame))
        else:
            merged_ranges.append((start_frame, end_frame))

    return merged_ranges


# --- Function Definitions ---


def get_scaling_transforms(control_params: dict) -> tuple:
    """
    Creates and caches a set of image scaling transformations based on control parameters.

    This function acts as a performance optimization. Creating transform objects can be
    resource-intensive. This function generates a unique key based on the current
    interpolation settings, checks if the transforms for these settings already exist
    in a cache (`_transform_cache`), and returns them if so. Otherwise, it creates
    the new set of transforms, caches them, and then returns them.

    Args:
        control_params (dict): A dictionary containing user-configurable settings,
                               including various interpolation mode selections.

    Returns:
        tuple: A large tuple containing various configured `torchvision.transforms.v2.Resize`
               objects and interpolation mode enums for different parts of the image
               processing pipeline.
    """
    # A unique key is created from all relevant control parameters.
    # This key represents a specific combination of user settings.
    config_key = (
        control_params.get("get_cropped_face_kpsTypeSelection", "BILINEAR"),
        control_params.get("original_face_128_384TypeSelection", "BILINEAR"),
        control_params.get("original_face_512TypeSelection", "BILINEAR"),
        control_params.get("UntransformTypeSelection", "BILINEAR"),
        control_params.get("ScalebackFrameTypeSelection", "BILINEAR"),
        control_params.get("expression_faceeditor_t256TypeSelection", "BILINEAR"),
        control_params.get("expression_faceeditor_backTypeSelection", "BILINEAR"),
        control_params.get("block_shiftTypeSelection", "NEAREST"),
        control_params.get("AntialiasTypeSelection", "False"),
    )

    # Performance check: If this exact configuration is already in the cache, return it immediately.
    if config_key in _transform_cache:
        _transform_cache.move_to_end(config_key)  # refresh LRU position
        return _transform_cache[config_key]

    # --- If not cached, create the new set of transforms ---

    # Map user-friendly string names to the actual PyTorch interpolation objects.
    interpolation_map = {
        "NEAREST": v2.InterpolationMode.NEAREST,
        "BILINEAR": v2.InterpolationMode.BILINEAR,
        "BICUBIC": v2.InterpolationMode.BICUBIC,
    }
    interpolation_get_cropped_face_kps = interpolation_map.get(
        control_params.get("get_cropped_face_kpsTypeSelection", "BILINEAR")
    )
    interpolation_original_face_128_384 = interpolation_map.get(
        control_params.get("original_face_128_384TypeSelection", "BILINEAR")
    )
    interpolation_original_face_512 = interpolation_map.get(
        control_params.get("original_face_512TypeSelection", "BILINEAR")
    )
    interpolation_Untransform = interpolation_map.get(
        control_params.get("UntransformTypeSelection", "BILINEAR")
    )
    interpolation_scaleback = interpolation_map.get(
        control_params.get("ScalebackFrameTypeSelection", "BILINEAR")
    )
    interpolation_expression_faceeditor_t256 = interpolation_map.get(
        control_params.get("expression_faceeditor_t256TypeSelection", "BILINEAR")
    )
    interpolation_expression_faceeditor_back = interpolation_map.get(
        control_params.get("expression_faceeditor_backTypeSelection", "BILINEAR")
    )

    interpolation_block_shift_map = {
        "NEAREST": "nearest",
        "BILINEAR": "bilinear",
        "BICUBIC": "bicubic",
    }
    interpolation_block_shift = interpolation_block_shift_map.get(
        control_params.get("block_shiftTypeSelection", "NEAREST")
    )

    antialias_method = control_params.get("AntialiasTypeSelection", "False") == "True"

    # Create the specific Resize transform objects with the selected settings.
    t256_face = v2.Resize(
        (256, 256),
        interpolation=interpolation_expression_faceeditor_t256,
        antialias=antialias_method,
    )
    t512 = v2.Resize(
        (512, 512),
        interpolation=interpolation_original_face_512,
        antialias=antialias_method,
    )
    t384 = v2.Resize(
        (384, 384),
        interpolation=interpolation_original_face_128_384,
        antialias=antialias_method,
    )
    t256 = v2.Resize(
        (256, 256),
        interpolation=interpolation_original_face_128_384,
        antialias=antialias_method,
    )
    t128 = v2.Resize(
        (128, 128),
        interpolation=interpolation_original_face_128_384,
        antialias=antialias_method,
    )

    # Store the entire collection of new transforms in a tuple.
    result = (
        t512,
        t384,
        t256,
        t128,
        interpolation_get_cropped_face_kps,
        interpolation_original_face_128_384,
        interpolation_original_face_512,
        interpolation_Untransform,
        interpolation_scaleback,
        t256_face,
        interpolation_expression_faceeditor_back,
        interpolation_block_shift,
    )

    # Save the result in the cache before returning it.
    # Evict the oldest entry first if the cache is at capacity (LRU).
    if len(_transform_cache) >= _TRANSFORM_CACHE_MAX:
        _transform_cache.popitem(last=False)
    _transform_cache[config_key] = result

    return result


def absoluteFilePaths(directory: str, include_subfolders=False):
    if include_subfolders:
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                yield os.path.abspath(os.path.join(dirpath, f))
    else:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                yield file_path


def truncate_text(text):
    if len(text) >= 35:
        return f"{text[:32]}..."
    return text


def get_video_files(folder_name, include_subfolders=False):
    return [
        f
        for f in absoluteFilePaths(folder_name, include_subfolders)
        if f.lower().endswith(video_extensions)
    ]


def get_image_files(folder_name, include_subfolders=False):
    return [
        f
        for f in absoluteFilePaths(folder_name, include_subfolders)
        if f.lower().endswith(image_extensions)
    ]


def is_image_file(file_name: str):
    return file_name.lower().endswith(image_extensions)


def is_video_file(file_name: str):
    return file_name.lower().endswith(video_extensions)


def is_file_exists(file_path: str) -> bool:
    if not file_path:
        return False
    return Path(file_path).is_file()


def get_file_type(file_name):
    if is_image_file(file_name):
        return "image"
    if is_video_file(file_name):
        return "video"
    return None


def get_scaled_resolution(
    media_width: Optional[int] = None,
    media_height: Optional[int] = None,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    media_capture: Optional[cv2.VideoCapture] = None,
) -> tuple[int, int]:
    """
    Calculates scaled dimensions for media to fit within given bounds while maintaining aspect ratio.

    This function can determine the source dimensions in two ways:
    1. Directly from the `media_width` and `media_height` arguments.
    2. By extracting them from a `cv2.VideoCapture` object if the dimensions are not provided.

    If the original dimensions are larger than the bounds (`max_width`, `max_height`),
    it scales them down proportionally.

    Args:
        media_width (int, optional): The original width of the media. Defaults to None.
        media_height (int, optional): The original height of the media. Defaults to None.
        max_width (int, optional): The maximum allowed width. Defaults to 1920.
        max_height (int, optional): The maximum allowed height. Defaults to 1080.
        media_capture (cv2.VideoCapture, optional): A video capture object to get dimensions from if they are not provided. Defaults to None.

    Returns:
        tuple[int, int]: A tuple containing the new scaled (width, height).
    """
    # Set default maximum bounds if not provided.
    if max_width is None:
        max_width = 1920
    if max_height is None:
        max_height = 1080

    # If dimensions are not provided, try to get them from the video capture object.
    if (
        (media_width is None or media_height is None)
        and media_capture
        and media_capture.isOpened()
    ):
        media_width = media_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        media_height = media_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # If dimensions are still not available, we cannot proceed.
    if (
        media_width is None
        or media_height is None
        or media_width == 0
        or media_height == 0
    ):
        return 0, 0  # Return a zero size if dimensions are invalid.

    # Check if the media dimensions exceed the maximum bounds.
    if media_width > max_width or media_height > max_height:
        # Calculate the scaling ratio for width and height.
        width_scale = max_width / media_width
        height_scale = max_height / media_height

        # Use the smaller ratio to ensure the media fits entirely within the bounds.
        scale = min(width_scale, height_scale)

        # Apply the scaling factor to the dimensions.
        scaled_width = media_width * scale
        scaled_height = media_height * scale

        return int(scaled_width), int(scaled_height)

    # If the media is already within bounds, return its original dimensions.
    return int(media_width), int(media_height)


def get_video_rotation(media_path: str) -> int:
    """
    Uses ffprobe to retrieve the video rotation metadata using a recursive search strategy.
    This is robust against variations in JSON structure (tags vs side_data_list).
    Returns 0, 90, 180, or 270.
    """

    # If OpenCV (>= 4.8.0) supports auto-rotation, we let it handle the rotation natively.
    # Returning 0 bypasses the slow ffprobe subprocess entirely, which massively
    # speeds up directory scanning and thumbnail generation!
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        return 0

    print(
        f"[INFO] Checking video rotation metadata for: {os.path.basename(media_path)}..."
    )

    if not is_ffmpeg_in_path():
        return 0

    try:
        # We select only the first video stream (v:0) to avoid getting audio rotation metadata
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "v:0",
            str(media_path),
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        stdout_data, stderr_data = process.communicate(timeout=10)

        if process.returncode != 0:
            print(f"[ERROR] ffprobe failed. Error: {stderr_data}")
            return 0

        data = json.loads(stdout_data)

        # --- Helper: Recursive Search ---
        def find_rotation_value(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k.lower() == "rotation":
                        return v
                    # Recursive call for nested dicts
                    result = find_rotation_value(v)
                    if result is not None:
                        return result
            elif isinstance(obj, list):
                for item in obj:
                    # Recursive call for items in lists
                    result = find_rotation_value(item)
                    if result is not None:
                        return result
            return None

        # Search for 'rotation' anywhere in the JSON
        rotation_raw = find_rotation_value(data)

        if rotation_raw is not None:
            try:
                rotation_angle = int(float(rotation_raw))

                # Normalize angle
                if rotation_angle < 0:
                    rotation_angle += 360
                rotation_angle = rotation_angle % 360

                # Align to standard angles
                if 85 <= rotation_angle <= 95:
                    print("[INFO] Detected video rotation: 90°")
                    return 90
                elif 175 <= rotation_angle <= 185:
                    print("[INFO] Detected video rotation: 180°")
                    return 180
                elif 265 <= rotation_angle <= 275:
                    print("[INFO] Detected video rotation: 270°")
                    return 270
                elif rotation_angle != 0:
                    print(
                        f"[INFO] Found rotation '{rotation_angle}°', but ignoring non-standard angle."
                    )

            except (ValueError, TypeError):
                pass  # Found the key but value wasn't a number

    except Exception as e:
        print(f"[ERROR] Video rotation check failed: {e}")

    print("[INFO] No rotation metadata applied (returning 0).")
    return 0


def _apply_frame_rotation(frame: np.ndarray, angle: int) -> np.ndarray:
    """Applies OpenCV rotation to a frame based on a metadata angle."""
    # The 'rotation: 90' tag typically implies a counter-clockwise rotation
    # to turn landscape (1920x1080) into portrait (1080x1920).
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def check_and_warn_vfr(file_path: str) -> bool:
    """
    Samples the first 200 frames using ffprobe to accurately detect Variable Frame Rate (VFR).
    Headers are often inaccurate, so analyzing actual packet durations is the safest method.

    Args:
        file_path (str): The absolute path to the video file.

    Returns:
        bool: True if VFR is detected, False otherwise.
    """
    if not file_path or not os.path.isfile(file_path):
        return False

    try:
        # We read the packet duration of the first 200 frames.
        # This is virtually instantaneous as it only reads container metadata, not pixel data.
        args = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=pkt_duration_time",
            "-read_intervals",
            "%+#200",  # Read only the first 200 frames
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            file_path,
        ]
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            return False

        durations = set()
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line and line != "N/A":
                try:
                    # Round to 3 decimal places to ignore floating point inaccuracies
                    durations.add(round(float(line), 3))
                except ValueError:
                    pass

        # If we have more than one distinct frame duration, the video is Variable Frame Rate.
        is_vfr = len(durations) > 1

        if is_vfr:
            print(
                "[WARN] -------------------------------------------------------------"
            )
            print("[WARN] VARIABLE FRAME RATE (VFR) DETECTED IN SOURCE VIDEO!")
            print("[WARN] The original media does not maintain a constant framerate.")
            print(
                "[WARN] Audio sync drift may occur during long recordings. For flawless"
            )
            print("[WARN] results, please transcode your video to Constant Frame Rate")
            print("[WARN] (CFR) using a tool like Handbrake before processing it here.")
            print(
                "[WARN] -------------------------------------------------------------"
            )
        else:
            print(
                "[INFO] Video framerate is Constant (CFR). Audio sync should be perfect."
            )

        return is_vfr

    except Exception as e:
        print(f"[WARN] Could not probe VFR status for {file_path}: {e}")
        return False


def benchmark(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Record the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.perf_counter()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(
            f"[INFO] Function '{func.__name__}' executed in {elapsed_time:.6f} seconds."
        )
        return result  # Return the result of the original function

    return wrapper


# --- OPTIMIZED MULTI-THREADING VIDEO LOCKS ---
_capture_locks: Dict[int, threading.Lock] = {}
_locks_mutex = threading.Lock()


def _get_capture_lock(capture_obj: cv2.VideoCapture) -> threading.Lock:
    """Retrieves or creates a unique thread lock for a specific VideoCapture instance."""
    obj_id = id(capture_obj)
    with _locks_mutex:
        if obj_id not in _capture_locks:
            _capture_locks[obj_id] = threading.Lock()
        return _capture_locks[obj_id]


def read_frame(
    capture_obj: cv2.VideoCapture,
    media_rotation: int = 0,
    preview_target_height: Optional[int] = None,
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Reads a single frame from the video capture object in a thread-safe manner
    and applies rotation.

    The 'lock' (Point 5) is critical as 'capture_obj' is a shared resource.
    It prevents race conditions between the feeder thread and seek operations.
    """
    capture_lock = _get_capture_lock(capture_obj)

    with capture_lock:
        ret, frame = capture_obj.read()

    if not ret:
        return False, None  # Return immediately if read fails

    # 1. Apply rotation (if necessary)
    if media_rotation != 0:
        frame = _apply_frame_rotation(frame, media_rotation)

    # 2. Apply resizing (if necessary)
    # This is done *after* the lock to avoid holding it during resizing.
    if ret and preview_target_height is not None:
        try:
            original_height, original_width = frame.shape[:2]
            if original_height == 0:
                return ret, frame  # Avoid division by zero

            # Use the specified target height
            target_height = preview_target_height
            aspect_ratio = original_width / original_height
            target_width = int(target_height * aspect_ratio)

            # Ensure width is even (good practice for some video operations)
            if target_width % 2 != 0:
                target_width += 1

            # cv2.INTER_AREA is generally the fastest and best for downscaling
            frame = cv2.resize(
                frame, (target_width, target_height), interpolation=cv2.INTER_AREA
            )
        except Exception as e:
            print(f"[ERROR] Failed to resize frame in preview_mode: {e}")
            # Fallback: return the original (rotated) frame if resize fails
            return ret, frame

    # Return the (potentially rotated and resized) frame
    return ret, frame


def seek_frame(capture_obj: cv2.VideoCapture, frame_number: int) -> bool:
    """
    Seeks a video capture object to a specific frame number in a thread-safe manner.
    Uses the same global lock as read_frame to prevent deadlocks.

    Args:
        capture_obj (cv2.VideoCapture): The shared OpenCV capture object.
        frame_number (int): The frame number to seek to.

    Returns:
        bool: The result of capture_obj.set().
    """
    capture_lock = _get_capture_lock(capture_obj)
    with capture_lock:
        return capture_obj.set(cv2.CAP_PROP_POS_FRAMES, frame_number)


def release_capture(capture_obj: cv2.VideoCapture):
    """
    Releases the OpenCV capture object in a thread-safe manner.
    Uses the same global lock as read_frame to prevent deadlocks.
    """
    if capture_obj is None:
        return

    obj_id = id(capture_obj)
    capture_lock = _get_capture_lock(capture_obj)

    with capture_lock:
        if capture_obj.isOpened():
            capture_obj.release()

    with _locks_mutex:
        _capture_locks.pop(obj_id, None)


def read_image_file(image_path):
    try:
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Always load as BGR
    except Exception as e:
        print(f"[ERROR] Failed to load {image_path}: {e}")
        return None

    if img is None:
        print("[ERROR] Failed to decode:", image_path)
        return None

    return img  # Return BGR format


def get_output_file_path(
    original_media_path: str,
    output_folder: str,
    media_type: str = "video",
    job_name: Optional[str] = None,
    use_job_name_for_output: bool = False,
    output_file_name: Optional[str] = None,
) -> str:
    """
    Determines the full output path for a processed media file based on a priority system.

    The base name for the output file is determined by the following priorities:
    1. `output_file_name`: If provided and `use_job_name_for_output` is False.
    2. `job_name`: If provided and `use_job_name_for_output` is True.
    3. Fallback: A combination of the original filename and a current timestamp.

    The file extension is determined by the `media_type`.

    Args:
        original_media_path (str): The path of the original input media.
        output_folder (str): The directory where the output file will be saved.
        media_type (str): The type of media ('video' or 'image'), used to determine the extension.
        job_name (str, optional): The name of the current job, used if `use_job_name_for_output` is True.
        use_job_name_for_output (bool): Flag to indicate if the job name should be used for the output filename.
        output_file_name (str, optional): A specific name for the output file.

    Returns:
        str: The fully constructed, absolute path for the output file.
    """
    date_and_time = datetime.now().strftime(r"%Y_%m_%d_%H_%M_%S")
    input_filename = os.path.basename(original_media_path)
    temp_path = Path(input_filename)

    output_base_name = None

    # --- Filename Priority Logic ---
    # Priority 1: Use the specific `output_file_name` if provided and not overridden by the job name flag.
    if not use_job_name_for_output and output_file_name:
        output_base_name = output_file_name
    # Priority 2: Use the `job_name` if the corresponding flag is checked.
    elif use_job_name_for_output and job_name:
        output_base_name = job_name
    # Priority 3 (Fallback): Use the original filename with a timestamp to ensure uniqueness.
    else:
        output_base_name = f"{temp_path.stem}_{date_and_time}"

    # --- Extension Logic ---
    if media_type == "video":
        extension = ".mp4"
    elif media_type == "image":
        extension = ".png"  # Default to PNG for processed images.
    elif media_type == "jpegimage":
        extension = ".jpg"  # Default to PNG for processed images.
    else:
        # If media type is unknown, try to preserve the original extension or default to nothing.
        extension = temp_path.suffix if temp_path.suffix else ""

    # --- Final Path Construction ---
    output_filename = f"{output_base_name}{extension}"
    output_file_path = os.path.join(output_folder, output_filename)
    return output_file_path


def is_ffmpeg_in_path():
    if not cmd_exist("ffmpeg"):
        print("[ERROR] FFMPEG Not found in your system!")
        return False
    return True


def cmd_exist(cmd):
    try:
        return shutil.which(cmd) is not None
    except ImportError:
        return any(
            os.access(os.path.join(path, cmd), os.X_OK)
            for path in os.environ["PATH"].split(os.pathsep)
        )


def get_dir_of_file(file_path):
    if file_path:
        return os.path.dirname(file_path)
    return os.path.curdir


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Converts a PyTorch tensor to a PIL Image.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
        tensor = (tensor * 255).clamp(0, 255).byte()
    tensor = tensor.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(tensor)


def keypoints_adjustments(
    kps_5: np.ndarray,
    parameters: Mapping[str, Any],
    source_kps: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Adjusts facial keypoints for morphing and manual alignments.
    Uses a Local Anisotropic Alignment strategy. This separates horizontal (Yaw)
    and vertical (Pitch) perspective compression by flat-rotating the face,
    scaling axes independently, and rotating back. It perfectly preserves
    inter-ocular distance while allowing jaw compression.
    """
    kps_5_adj = kps_5.copy()

    if (
        parameters.get("FaceKeypointsReplaceEnableToggle", False)
        and source_kps is not None
    ):
        morph_amount = parameters.get("FaceKeypointsReplaceDecimalSlider", 0.0)

        if morph_amount > 0.0:
            try:
                # 1. Isolate Translation: Center the keypoints
                tgt_centroid = np.mean(kps_5_adj, axis=0)
                src_centroid = np.mean(source_kps, axis=0)

                tgt_centered = kps_5_adj - tgt_centroid
                src_centered = source_kps - src_centroid

                # 2. Find Roll Angles based strictly on the eyes (Indices 0 and 1)
                # This gives us the true intrinsic tilt axis of the face
                angle_tgt = np.arctan2(
                    kps_5_adj[1, 1] - kps_5_adj[0, 1], kps_5_adj[1, 0] - kps_5_adj[0, 0]
                )
                angle_src = np.arctan2(
                    source_kps[1, 1] - source_kps[0, 1],
                    source_kps[1, 0] - source_kps[0, 0],
                )

                # 3. Rotate both faces to be perfectly upright/horizontal (Roll = 0)
                cos_tgt, sin_tgt = np.cos(-angle_tgt), np.sin(-angle_tgt)
                R_flat_tgt = np.array([[cos_tgt, -sin_tgt], [sin_tgt, cos_tgt]])
                tgt_flat = tgt_centered @ R_flat_tgt.T

                cos_src, sin_src = np.cos(-angle_src), np.sin(-angle_src)
                R_flat_src = np.array([[cos_src, -sin_src], [sin_src, cos_src]])
                src_flat = src_centered @ R_flat_src.T

                # 4. Local Anisotropic Scale (Independent X and Y)
                # X std-dev is primarily dictated by eye distance (Width).
                # Y std-dev is primarily dictated by eye-to-mouth distance (Height).
                eps = 1e-6
                std_tgt = np.std(tgt_flat, axis=0) + eps
                std_src = np.std(src_flat, axis=0) + eps

                scale_x = std_tgt[0] / std_src[0]
                scale_y = std_tgt[1] / std_src[1]

                # 5. Apply Independent Scaling
                # If target looks down -> scale_y compresses, scale_x is kept intact!
                src_flat_scaled = src_flat * np.array([scale_x, scale_y])

                # 6. Rotate back to the Target's original Roll angle
                cos_inv, sin_inv = np.cos(angle_tgt), np.sin(angle_tgt)
                R_unflat = np.array([[cos_inv, -sin_inv], [sin_inv, cos_inv]])

                src_aligned = src_flat_scaled @ R_unflat.T

                # 7. Final translation
                source_kps_aligned = src_aligned + tgt_centroid

                # 8. Apply linear interpolation (Morphing)
                kps_5_adj = (
                    kps_5_adj + morph_amount * (source_kps_aligned - kps_5_adj)
                ).astype(np.float32)

            except Exception as e:
                print(f"[WARNING] Face Keypoints Morphing bypassed: {e}")

    # --- MANUAL ALIGNMENTS (Sliders) ---
    if parameters.get("FaceAdjEnableToggle", False):
        kps_5_adj[:, 0] += parameters["KpsXSlider"]
        kps_5_adj[:, 1] += parameters["KpsYSlider"]
        kps_5_adj[:, 0] -= 255
        kps_5_adj[:, 0] *= 1 + parameters["KpsScaleSlider"] / 100.0
        kps_5_adj[:, 0] += 255
        kps_5_adj[:, 1] -= 255
        kps_5_adj[:, 1] *= 1 + parameters["KpsScaleSlider"] / 100.0
        kps_5_adj[:, 1] += 255

    if (
        parameters.get("LandmarksPositionAdjEnableToggle", False)
        and kps_5_adj.shape[0] >= 5
    ):
        kps_5_adj[0][0] += parameters["EyeLeftXAmountSlider"]
        kps_5_adj[0][1] += parameters["EyeLeftYAmountSlider"]
        kps_5_adj[1][0] += parameters["EyeRightXAmountSlider"]
        kps_5_adj[1][1] += parameters["EyeRightYAmountSlider"]
        kps_5_adj[2][0] += parameters["NoseXAmountSlider"]
        kps_5_adj[2][1] += parameters["NoseYAmountSlider"]
        kps_5_adj[3][0] += parameters["MouthLeftXAmountSlider"]
        kps_5_adj[3][1] += parameters["MouthLeftYAmountSlider"]
        kps_5_adj[4][0] += parameters["MouthRightXAmountSlider"]
        kps_5_adj[4][1] += parameters["MouthRightYAmountSlider"]

    return kps_5_adj


# Cache for static target grids to prevent massive VRAM reallocation per frame
_static_grid_cache: Dict[
    Tuple[int, int, torch.device], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def get_grid_for_pasting(
    tform_target_to_source: trans.SimilarityTransform,
    target_h: int,
    target_w: int,
    source_h: int,
    source_w: int,
    device: torch.device,
):
    """
    ULTRA-OPTIMIZED: Generates a sampling grid for grid_sample.
    Uses 1D tensor broadcasting and caches the static target grids to save
    massive amounts of VRAM allocation/deallocation overhead per face/frame.
    """
    grid_key = (target_h, target_w, device)

    # Fetch from cache or create 1D coordinate tensors and target grid once
    if grid_key not in _static_grid_cache:
        y = torch.arange(target_h, device=device, dtype=torch.float32).view(-1, 1)
        x = torch.arange(target_w, device=device, dtype=torch.float32).view(1, -1)

        grid_y = y.expand(target_h, target_w)
        grid_x = x.expand(target_h, target_w)
        target_grid_yx_pixels = torch.stack((grid_y, grid_x), dim=-1).unsqueeze(0)

        _static_grid_cache[grid_key] = (x, y, target_grid_yx_pixels)

    x, y, target_grid_yx_pixels = _static_grid_cache[grid_key]

    # Transformation matrix from tform_target_to_source (2x3)
    M = torch.tensor(
        tform_target_to_source.params[0:2, :], dtype=torch.float32, device=device
    )

    # Apply affine transformation using automatic broadcasting -> results in (H, W)
    src_x = x * M[0, 0] + y * M[0, 1] + M[0, 2]
    src_y = x * M[1, 0] + y * M[1, 1] + M[1, 2]

    # Normalize source coordinates directly for grid_sample [-1, 1]
    src_x_norm = (src_x / (source_w - 1.0)) * 2.0 - 1.0
    src_y_norm = (src_y / (source_h - 1.0)) * 2.0 - 1.0

    # Stack to create the final normalized grid: 1 x H x W x 2
    source_grid_normalized_xy = torch.stack((src_x_norm, src_y_norm), dim=-1).unsqueeze(
        0
    )

    return target_grid_yx_pixels, source_grid_normalized_xy


def draw_bounding_boxes_on_detected_faces(
    img: torch.Tensor, det_faces_data: list, color_rgb: list | None = None
) -> torch.Tensor:
    """
    OPTIMIZED: Removed unnecessary .expand() calls.
    Relies on PyTorch's native C++ broadcasting for instant assignment.
    """
    _color = color_rgb if color_rgb is not None else [0, 255, 0]
    for i, fface in enumerate(det_faces_data):
        bbox = fface["bbox"]
        x_min, y_min, x_max, y_max = map(int, bbox)

        # Ensure bounding box is within the image dimensions
        _, h, w = img.shape
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w - 1, x_max), min(h - 1, y_max)

        # Dynamically compute thickness based on the image resolution
        max_dimension = max(img.shape[1], img.shape[2])
        thickness = max(4, max_dimension // 400)

        color_tensor_c11 = torch.tensor(
            _color, dtype=img.dtype, device=img.device
        ).view(-1, 1, 1)

        # PyTorch handles the broadcasting automatically, no need to expand()
        img[:, y_min : y_min + thickness, x_min : x_max + 1] = color_tensor_c11
        img[:, y_max - thickness + 1 : y_max + 1, x_min : x_max + 1] = color_tensor_c11
        img[:, y_min : y_max + 1, x_min : x_min + thickness] = color_tensor_c11
        img[:, y_min : y_max + 1, x_max - thickness + 1 : x_max + 1] = color_tensor_c11

    return img


def paint_landmarks_on_image(img: torch.Tensor, landmarks_data: list) -> torch.Tensor:
    """
    OPTIMIZED: Replaced deeply nested loops and per-pixel tensor allocations
    with tensor slicing and pre-allocated colors to eliminate CPU bottlenecks.
    """
    img_out_hwc = img.clone()
    p = 2

    for item in landmarks_data:
        keypoints = item["kps"]
        kcolor = item["color"]
        if keypoints is not None:
            # OPTIMIZATION: Allocate the color tensor ONCE per face, not per pixel
            kcolor_tensor = torch.tensor(kcolor, device=img.device, dtype=img.dtype)

            for kpoint in keypoints:
                kx, ky = int(kpoint[0]), int(kpoint[1])

                # OPTIMIZATION: Use direct slicing instead of nested loops
                y_min = max(0, ky - p // 2)
                y_max = min(img_out_hwc.shape[0], ky + p // 2 + 1)
                x_min = max(0, kx - p // 2)
                x_max = min(img_out_hwc.shape[1], kx + p // 2 + 1)

                if y_min < y_max and x_min < x_max:
                    img_out_hwc[y_min:y_max, x_min:x_max] = kcolor_tensor

    return img_out_hwc
