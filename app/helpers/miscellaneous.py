import os
import shutil
import cv2
import time
from collections import UserDict
import hashlib
import numpy as np
from functools import wraps
from datetime import datetime
from pathlib import Path
from torchvision.transforms import v2
import threading
lock = threading.Lock()

# --- Global Scope ---

# scaling transforms cache dictionary at the module level
_transform_cache = {}
image_extensions = ('.jpg', '.jpeg', '.jpe', '.png', '.webp', '.tif', '.tiff', '.jp2', '.exr', '.hdr', '.ras', '.pnm', '.ppm', '.pgm', '.pbm', '.pfm')
video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.gif')

# --- Class Definitions ---

class ThumbnailManager:
    """
    Manages the creation, storage, and retrieval of media file thumbnails.

    This class encapsulates all thumbnail-related logic, such as hashing filenames,
    managing the thumbnail storage directory, and generating thumbnail images from
    video frames or images.
    """
    def __init__(self, thumbnail_dir: str = '.thumbnails'):
        """
        Initializes the ThumbnailManager.

        Args:
            thumbnail_dir (str): The name of the directory to store thumbnails,
                                 created in the current working directory.
        """
        self.thumbnail_dir = os.path.join(os.getcwd(), thumbnail_dir)
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
        return hashlib.md5(hash_input.encode('utf-8')).hexdigest()

    def get_thumbnail_path(self, file_path: str) -> (str, str):
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
        if len(frame.shape) == 2: frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4: frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        height, width, _ = frame.shape
        width, height = get_scaled_resolution(media_width=width, media_height=height, max_height=140, max_width=140)
        
        resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        try:
            cv2.imwrite(png_path, resized_frame)
            if os.path.getsize(png_path) > 30 * 1024: # If PNG is > 30KB
                os.remove(png_path)
                raise Exception("PNG file too large, falling back to JPEG.")
        except Exception:
            jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 98, cv2.IMWRITE_JPEG_OPTIMIZE, 1, cv2.IMWRITE_JPEG_PROGRESSIVE, 1]
            cv2.imwrite(jpg_path, resized_frame, jpeg_params)

class DFMModelManager:
    """
    Manages the discovery and retrieval of DeepFace Model (DFM) files.
    
    This class scans a specified directory for .dfm and .onnx model files,
    making them available for use in the application, for example, in UI dropdowns.
    """
    def __init__(self, models_path: str = './model_assets/dfm_models'):
        """
        Initializes the DFMModelManager.
        
        Args:
            models_path (str): The path to the directory containing DFM model files.
        """
        self.models_path = models_path
        self.models_data = {}
        self.refresh_models()

    def refresh_models(self) -> None:
        """
        Scans the model directory and updates the internal dictionary of found models.
        """
        self.models_data.clear()
        if not os.path.isdir(self.models_path):
            print(f"[Warning] DFM models directory not found at: {self.models_path}")
            return
            
        for dfm_file in os.listdir(self.models_path):
            if dfm_file.endswith(('.dfm', '.onnx')):
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
        return dfm_values[0] if dfm_values else ''

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
        control_params.get("AntialiasTypeSelection", "False")
    )

    # Performance check: If this exact configuration is already in the cache, return it immediately.
    if config_key in _transform_cache:
        return _transform_cache[config_key]

    # --- If not cached, create the new set of transforms ---

    # Map user-friendly string names to the actual PyTorch interpolation objects.
    interpolation_map = {
        'NEAREST': v2.InterpolationMode.NEAREST,
        'BILINEAR': v2.InterpolationMode.BILINEAR,
        'BICUBIC': v2.InterpolationMode.BICUBIC
    }
    interpolation_get_cropped_face_kps = interpolation_map.get(control_params.get("get_cropped_face_kpsTypeSelection", "BILINEAR"))
    interpolation_original_face_128_384 = interpolation_map.get(control_params.get("original_face_128_384TypeSelection", "BILINEAR"))
    interpolation_original_face_512 = interpolation_map.get(control_params.get("original_face_512TypeSelection", "BILINEAR"))
    interpolation_Untransform = interpolation_map.get(control_params.get("UntransformTypeSelection", "BILINEAR"))
    interpolation_scaleback = interpolation_map.get(control_params.get("ScalebackFrameTypeSelection", "BILINEAR"))
    interpolation_expression_faceeditor_t256 = interpolation_map.get(control_params.get("expression_faceeditor_t256TypeSelection", "BILINEAR"))
    interpolation_expression_faceeditor_back = interpolation_map.get(control_params.get("expression_faceeditor_backTypeSelection", "BILINEAR"))
    
    interpolation_block_shift_map = {'NEAREST': 'nearest', 'BILINEAR': 'bilinear', 'BICUBIC': 'bicubic'}
    interpolation_block_shift = interpolation_block_shift_map.get(control_params.get("block_shiftTypeSelection", "NEAREST"))
    
    antialias_method = control_params.get("AntialiasTypeSelection", "False") == 'True'

    # Create the specific Resize transform objects with the selected settings.
    t256_face = v2.Resize((256, 256), interpolation=interpolation_expression_faceeditor_t256, antialias=antialias_method)
    t512 = v2.Resize((512, 512), interpolation=interpolation_original_face_512, antialias=antialias_method)
    t384 = v2.Resize((384, 384), interpolation=interpolation_original_face_128_384, antialias=antialias_method)
    t256 = v2.Resize((256, 256), interpolation=interpolation_original_face_128_384, antialias=antialias_method)
    t128 = v2.Resize((128, 128), interpolation=interpolation_original_face_128_384, antialias=antialias_method)
    
    # Store the entire collection of new transforms in a tuple.
    result = (t512, t384, t256, t128, interpolation_get_cropped_face_kps, interpolation_original_face_128_384, interpolation_original_face_512, interpolation_Untransform, interpolation_scaleback, t256_face, interpolation_expression_faceeditor_back, interpolation_block_shift)
    
    # Save the result in the cache before returning it.
    _transform_cache[config_key] = result
    
    return result

def absoluteFilePaths(directory: str, include_subfolders=False):
    if include_subfolders:
        for dirpath,_,filenames in os.walk(directory):
            for f in filenames:
                yield os.path.abspath(os.path.join(dirpath, f))
    else:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                yield file_path

def truncate_text(text):
    if len(text) >= 35:
        return f'{text[:32]}...'
    return text

def get_video_files(folder_name, include_subfolders=False):
    return [f for f in absoluteFilePaths(folder_name, include_subfolders) if f.lower().endswith(video_extensions)]

def get_image_files(folder_name, include_subfolders=False):
    return [f for f in absoluteFilePaths(folder_name, include_subfolders) if f.lower().endswith(image_extensions)]

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
        return 'image'
    if is_video_file(file_name):
        return 'video'
    return None

def get_scaled_resolution(media_width: int = None, media_height: int = None, max_width: int = None, max_height: int = None, media_capture: cv2.VideoCapture = None) -> tuple[int, int]:
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
    if max_width is None: max_width = 1920
    if max_height is None: max_height = 1080

    # If dimensions are not provided, try to get them from the video capture object.
    if (media_width is None or media_height is None) and media_capture and media_capture.isOpened():
        media_width = media_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        media_height = media_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # If dimensions are still not available, we cannot proceed.
    if media_width is None or media_height is None or media_width == 0 or media_height == 0:
        return 0, 0 # Return a zero size if dimensions are invalid.

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

def benchmark(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Record the start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.perf_counter()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.6f} seconds.")
        return result  # Return the result of the original function
    return wrapper

def read_frame(capture_obj: cv2.VideoCapture, preview_mode=False):
    with lock:
        ret, frame = capture_obj.read()
    if ret and preview_mode:
        pass
        # width, height = get_scaled_resolution(capture_obj)
        # frame = cv2.resize(fr2ame, dsize=(width, height), interpolation=cv2.INTER_LANCZOS4)
    return ret, frame

def read_image_file(image_path):
    try:
        img_array = np.fromfile(image_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Always load as BGR
    except Exception as e:
        print(f"Failed to load {image_path}: {e}")
        return None

    if img is None:
        print("Failed to decode:", image_path)
        return None

    return img  # Return BGR format

def get_output_file_path(original_media_path: str, output_folder: str, media_type: str = 'video', job_name: str = None, use_job_name_for_output: bool = False, output_file_name: str = None) -> str:
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
    date_and_time = datetime.now().strftime(r'%Y_%m_%d_%H_%M_%S')
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
        output_base_name = f'{temp_path.stem}_{date_and_time}'

    # --- Extension Logic ---
    if media_type == 'video':
        extension = '.mp4'
    elif media_type == 'image':
        extension = '.png'  # Default to PNG for processed images.
    else:
        # If media type is unknown, try to preserve the original extension or default to nothing.
        extension = temp_path.suffix if temp_path.suffix else '' 

    # --- Final Path Construction ---
    output_filename = f'{output_base_name}{extension}'
    output_file_path = os.path.join(output_folder, output_filename)
    return output_file_path

def is_ffmpeg_in_path():
    if not cmd_exist('ffmpeg'):
        print("FFMPEG Not found in your system!")
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