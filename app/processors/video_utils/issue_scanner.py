import copy
import cv2
import numpy
import torch
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

# Internal project imports
from app.processors.workers.frame_worker import FrameWorker
import app.helpers.miscellaneous as misc_helpers
from app.ui.widgets.actions import video_control_actions
from app.helpers.typing_helper import (
    ControlTypes,
    FacesParametersTypes,
    ParametersTypes,
)

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow
    from app.processors.video_utils.sequential_detector import SequentialDetector

# Type aliases for complex dictionary structures used in scanning
IssueScanTargetEmbeddings = dict[str, dict[str, numpy.ndarray]]
IssueScanTargetSnapshot = dict[str, dict[str, Any]]

# --- ALLOWLISTS ---
# During an offline scan, we only care about parameters that affect face detection 
# and identification. Filtering out unrelated rendering parameters (like color correction or enhancers) 
# drastically reduces dictionary cloning overhead and prevents unintended state bleed.
SCAN_CONTROL_ALLOWLIST = frozenset(
    {
        "GlobalInputResizeToggle",
        "GlobalInputResizeSizeSelection",
        "DetectorModelSelection",
        "MaxFacesToDetectSlider",
        "DetectorScoreSlider",
        "LandmarkDetectToggle",
        "LandmarkDetectModelSelection",
        "LandmarkDetectScoreSlider",
        "DetectFromPointsToggle",
        "AutoRotationToggle",
        "LandmarkMeanEyesToggle",
        "FaceTrackingEnableToggle",
        "ByteTrackTrackThreshSlider",
        "ByteTrackMatchThreshSlider",
        "ByteTrackTrackBufferSlider",
        "KPSSmoothingEnableToggle",
        "KPSEmaAlphaSlider",
        "RecognitionModelSelection",
    }
)
SCAN_FACE_PARAM_ALLOWLIST = frozenset({"SimilarityThresholdSlider"})


class IssueScanner:
    """
    Handles offline analytical scans to detect face tracking and identification issues.
    Decoupled from the real-time VideoProcessor to maintain the Single Responsibility Principle.
    
    Architectural impact: By running this in an isolated environment using state snapshots, 
    we ensure that long-running file scans do not block the UI thread or corrupt the active 
    playback state of the SequentialDetector.
    """

    def __init__(self, main_window: "MainWindow", sequential_detector: "SequentialDetector", media_path: Optional[str] = None, max_frame_number: int = 0, media_rotation: int = 0):
        self.main_window = main_window
        self.sequential_detector = sequential_detector
        self.media_path = media_path
        self.max_frame_number = max_frame_number
        self.media_rotation = media_rotation

    @staticmethod
    def _get_target_input_height_for_control(control: Mapping[str, Any] | None) -> Optional[int]:
        """Parses the UI control dictionary to determine if a global resolution downscale is requested."""
        resize_enabled = (
            bool(control.get("GlobalInputResizeToggle", False))
            if isinstance(control, Mapping)
            else False
        )
        if not resize_enabled:
            return None
        try:
            size_str = control.get("GlobalInputResizeSizeSelection", "720p") if isinstance(control, Mapping) else "720p"
            return int(str(size_str).replace("p", ""))
        except Exception as e:
            print(f"[WARN] Could not parse global input resolution for scan, defaulting to original size. Error: {e}")
            return None

    @staticmethod
    def _filter_scan_control(control: Mapping[str, Any] | None) -> ControlTypes:
        """Deep copies only the essential detection controls to prevent UI dictionary race conditions."""
        if not isinstance(control, Mapping):
            return cast(ControlTypes, {})
        return cast(
            ControlTypes,
            {
                str(key): copy.deepcopy(value)
                for key, value in control.items()
                if str(key) in SCAN_CONTROL_ALLOWLIST
            },
        )

    @staticmethod
    def _filter_scan_face_params(params: Mapping[str, Any] | None, target_face_ids: Iterable[str] | None = None) -> FacesParametersTypes:
        """Deep copies face-specific parameters (like Similarity Threshold) for safe worker thread use."""
        if not isinstance(params, Mapping):
            return cast(FacesParametersTypes, {})

        allowed_face_ids = {str(face_id) for face_id in target_face_ids} if target_face_ids is not None else None
        filtered: FacesParametersTypes = cast(FacesParametersTypes, {})

        for face_id, raw_face_params in params.items():
            face_id_str = str(face_id)
            if allowed_face_ids is not None and face_id_str not in allowed_face_ids:
                continue
            if not isinstance(raw_face_params, Mapping):
                filtered[face_id_str] = cast(ParametersTypes, {})
                continue
            filtered_face_params = {
                str(key): copy.deepcopy(value)
                for key, value in raw_face_params.items()
                if str(key) in SCAN_FACE_PARAM_ALLOWLIST
            }
            filtered[face_id_str] = cast(ParametersTypes, filtered_face_params)

        return filtered

    @staticmethod
    def _marker_control_data_for_position(markers: Mapping[Any, Any] | None, frame_number: int) -> Mapping[str, Any] | None:
        """Locates the closest preceding keyframe marker to apply local parameter overrides."""
        if not isinstance(markers, Mapping) or not markers:
            return None

        latest_key: Any = None
        latest_frame = None
        for raw_key in markers.keys():
            try:
                marker_frame = int(raw_key)
            except (TypeError, ValueError):
                continue
            if marker_frame > frame_number:
                continue
            if latest_frame is None or marker_frame > latest_frame:
                latest_frame = marker_frame
                latest_key = raw_key

        if latest_key is None:
            return None

        marker_data = markers.get(latest_key)
        if not isinstance(marker_data, Mapping):
            return None
        control_data = marker_data.get("control")
        return control_data if isinstance(control_data, Mapping) else None

    @staticmethod
    def _issue_scan_vr180_enabled(control: Mapping[str, Any] | None) -> bool:
        """Checks if VR180 spherical mapping is enabled, which alters standard detection paradigms."""
        return isinstance(control, Mapping) and bool(control.get("VR180ModeEnableToggle"))

    def get_issue_scan_unavailable_reason(
        self,
        control: Mapping[str, Any] | None,
        scan_ranges: Iterable[tuple[int, int]] | None = None,
        markers: Mapping[Any, Any] | None = None,
        fallback_control: Mapping[str, Any] | None = None,
    ) -> str | None:
        """Validates if a scan can be performed. VR180 mode currently unsupported due to complex equirectangular distorsions."""
        if scan_ranges is None:
            if self._issue_scan_vr180_enabled(control) or (
                fallback_control is not control
                and self._issue_scan_vr180_enabled(fallback_control)
            ):
                return "Issue scans are not supported while VR180 mode is enabled."
            return None

        if not isinstance(markers, Mapping) or not markers:
            if self._issue_scan_vr180_enabled(control):
                return "Issue scans are not supported while VR180 mode is enabled."
            return None

        normalized_marker_frames: list[tuple[Any, int]] = []
        for raw_key in markers.keys():
            try:
                normalized_marker_frames.append((raw_key, int(raw_key)))
            except (TypeError, ValueError):
                continue
        normalized_marker_frames.sort(key=lambda item: item[1])

        for start_frame, end_frame in scan_ranges:
            if end_frame < start_frame:
                continue

            if self._issue_scan_vr180_enabled(self._marker_control_data_for_position(markers, int(start_frame)) or control):
                return "Issue scans are not supported while VR180 mode is enabled."

            for raw_key, marker_frame in normalized_marker_frames:
                if marker_frame < start_frame:
                    continue
                if marker_frame > end_frame:
                    break
                marker_data = markers.get(raw_key)
                if not isinstance(marker_data, Mapping):
                    continue
                if self._issue_scan_vr180_enabled(cast(Mapping[str, Any] | None, marker_data.get("control"))):
                    return "Issue scans are not supported while VR180 mode is enabled."
        return None

    def _get_issue_scan_ranges(self) -> List[Tuple[int, int]]:
        """Converts user-defined UI start/end points into concrete frame tuple ranges."""
        max_frame = int(self.max_frame_number)
        scan_ranges: List[Tuple[int, int]] = []
        open_start_frame: Optional[int] = None

        for start_frame, end_frame in self.main_window.job_marker_pairs:
            if start_frame is None:
                continue
            normalized_start = int(start_frame)
            if end_frame is None:
                open_start_frame = normalized_start
                continue

            normalized_end = int(end_frame)
            if normalized_end >= normalized_start:
                scan_ranges.append((normalized_start, normalized_end))

        if open_start_frame is not None and open_start_frame <= max_frame:
            scan_ranges.append((open_start_frame, max_frame))

        if scan_ranges:
            return misc_helpers.normalize_issue_scan_ranges(scan_ranges)

        return [(0, max_frame)]

    def describe_issue_scan_scope(self, scan_ranges: Optional[List[Tuple[int, int]]] = None) -> str:
        """Returns a human-readable string defining the scope of the scan for the UI progress dialog."""
        scan_ranges = scan_ranges or self._get_issue_scan_ranges()
        max_frame = int(self.max_frame_number)
        if not getattr(self.main_window, "job_marker_pairs", []):
            return "Scanning full clip"
        if scan_ranges == [(0, max_frame)]:
            return "Scanning full clip"

        open_start_frames = [
            int(start_frame)
            for start_frame, end_frame in self.main_window.job_marker_pairs
            if start_frame is not None and end_frame is None
        ]
        has_open_start = bool(open_start_frames)
        open_start_frame = min(open_start_frames) if open_start_frames else None

        if len(scan_ranges) == 1:
            start_frame, end_frame = scan_ranges[0]
            if has_open_start and end_frame == max_frame and open_start_frame is not None:
                if start_frame < open_start_frame:
                    return f"Scanning 1 marked range and record start frame {open_start_frame} to end"
                if open_start_frame > 0:
                    return f"Scanning from record start frame {open_start_frame}"
            return "Scanning 1 marked range"

        effective_complete_segments = len(scan_ranges)
        effective_open_start_frame: Optional[int] = None
        if has_open_start and scan_ranges[-1][1] == max_frame and open_start_frame is not None:
            effective_open_start_frame = open_start_frame
            effective_complete_segments -= 1

        if effective_complete_segments and effective_open_start_frame is not None:
            range_label = "range" if effective_complete_segments == 1 else "ranges"
            return (
                f"Scanning {effective_complete_segments} marked {range_label} "
                f"and record start frame {effective_open_start_frame} to end"
            )
        if effective_complete_segments:
            range_label = "range" if effective_complete_segments == 1 else "ranges"
            return f"Scanning {effective_complete_segments} marked {range_label}"
        if effective_open_start_frame is not None:
            return f"Scanning from record start frame {effective_open_start_frame}"
        return "Scanning full clip"

    @staticmethod
    def _compute_longest_issue_run(issue_frames: list[int]) -> int:
        """Analyzes a list of failing frames to find the longest continuous dropout sequence."""
        longest_issue_run = 0
        current_run = 0
        previous_frame = None
        for frame_number in sorted(set(issue_frames)):
            if previous_frame is not None and frame_number == previous_frame + 1:
                current_run += 1
            else:
                current_run = 1
            longest_issue_run = max(longest_issue_run, current_run)
            previous_frame = frame_number
        return longest_issue_run

    def _get_issue_scan_bytetrack_config(self, control: Mapping[str, Any] | None) -> tuple[bool, int, int, int]:
        """Extracts ByteTrack configurations safely to ensure the tracker is initialized correctly per segment."""
        if not isinstance(control, Mapping):
            return (False, 40, 80, 30)
        return (
            bool(control.get("FaceTrackingEnableToggle", False)),
            int(control.get("ByteTrackTrackThreshSlider", 40)),
            int(control.get("ByteTrackMatchThreshSlider", 80)),
            int(control.get("ByteTrackTrackBufferSlider", 30)),
        )

    def _resolve_scan_state_for_frame(
        self,
        frame_number: int,
        base_control: ControlTypes,
        base_params: FacesParametersTypes,
        target_faces_snapshot: Optional[dict] = None,
        control_defaults_snapshot: Optional[ControlTypes] = None,
    ) -> tuple[ControlTypes, FacesParametersTypes]:
        """
        Calculates the active parameter/control state for a specific frame.
        Emulates playback behaviour where UI markers override the base state.
        """
        marker_data = video_control_actions._get_marker_data_for_position(self.main_window, frame_number)
        if not marker_data:
            return (
                self._filter_scan_control(copy.deepcopy(base_control)),
                self._filter_scan_face_params(copy.deepcopy(base_params)),
            )

        local_params = self._filter_scan_face_params(cast(FacesParametersTypes, marker_data.get("parameters", {})))
        local_control: ControlTypes = cast(ControlTypes, {})
        local_control.update(
            self._filter_scan_control(
                cast(ControlTypes, control_defaults_snapshot if control_defaults_snapshot is not None else {})
            )
        )

        control_data = marker_data.get("control")
        if isinstance(control_data, dict):
            local_control.update(self._filter_scan_control(cast(ControlTypes, control_data).copy()))

        active_target_faces = target_faces_snapshot if target_faces_snapshot is not None else self.main_window.target_faces
        default_scan_face_params = cast(
            ParametersTypes,
            self._filter_scan_face_params({"__default__": self.main_window.default_parameters.data}).get("__default__", {}),
        )
        for face_id in active_target_faces.keys():
            face_id_str = str(face_id)
            if face_id_str not in local_params:
                local_params[face_id_str] = cast(ParametersTypes, copy.deepcopy(default_scan_face_params))

        return self._filter_scan_control(local_control), self._filter_scan_face_params(local_params, active_target_faces.keys())

    def _build_issue_scan_state_segments(
        self,
        scan_ranges: List[Tuple[int, int]],
        base_control: ControlTypes,
        base_params: FacesParametersTypes,
        target_faces_snapshot: dict,
        control_defaults_snapshot: Optional[ControlTypes] = None,
    ) -> list[tuple[int, int, ControlTypes, FacesParametersTypes]]:
        """
        Breaks continuous scan ranges into smaller uniform segments divided by UI markers.
        Impact on performance: Drastically reduces overhead by allowing the core loop to reuse 
        the exact same parameter dictionary for thousands of frames until a marker is hit.
        """
        marker_positions = sorted(int(frame_number) for frame_number in getattr(self.main_window, "markers", {}).keys())
        segments: list[tuple[int, int, ControlTypes, FacesParametersTypes]] = []

        for start_frame, end_frame in scan_ranges:
            range_markers = [marker_frame for marker_frame in marker_positions if start_frame < marker_frame <= end_frame]
            segment_start = start_frame
            local_control, local_params = self._resolve_scan_state_for_frame(
                start_frame, base_control, base_params, target_faces_snapshot, control_defaults_snapshot
            )

            for next_marker_frame in range_markers + [end_frame + 1]:
                segment_end = next_marker_frame - 1
                if segment_end >= segment_start:
                    segments.append((segment_start, segment_end, local_control, local_params))
                if next_marker_frame <= end_frame:
                    segment_start = next_marker_frame
                    local_control, local_params = self._resolve_scan_state_for_frame(
                        next_marker_frame, base_control, base_params, target_faces_snapshot, control_defaults_snapshot
                    )

        return segments

    def _reset_issue_scan_sequential_state(self) -> None:
        """Clears the temporal tracker history. Essential when jumping across discontinuous segments."""
        self.sequential_detector.reset_state()

    def _prepare_issue_scan_match_context(
        self,
        local_control: ControlTypes,
        local_params: FacesParametersTypes,
        target_faces_snapshot: IssueScanTargetSnapshot,
    ) -> dict[str, Any]:
        """
        Preloads ArcFace embeddings into a structured context dict.
        By preparing this once per segment, we avoid hitting dictionaries or recalculating 
        thresholds during the hot-path execution of `scan_issue_frames`.
        """
        recognition_model = str(local_control.get("RecognitionModelSelection", "arcface_128"))
        similarity_type = str("Auto")
        default_params = dict(self.main_window.default_parameters.data)
        prepared_targets: list[tuple[str, float, numpy.ndarray]] = []

        for target_id, target_face_snapshot in target_faces_snapshot.items():
            face_id_str = str(target_face_snapshot.get("face_id", target_id))
            face_specific_params = misc_helpers.copy_mapping_data(local_params.get(face_id_str))
            params_pd = misc_helpers.ParametersDict(face_specific_params, default_params)
            target_embeddings = cast(IssueScanTargetEmbeddings, target_face_snapshot.get("embeddings_by_model", {}))
            target_embedding = target_embeddings.get(recognition_model, {}).get(similarity_type)
            if not isinstance(target_embedding, numpy.ndarray) or target_embedding.size == 0:
                continue
            prepared_targets.append((face_id_str, float(params_pd["SimilarityThresholdSlider"]), target_embedding))

        return {
            "recognition_model": recognition_model,
            "similarity_type": similarity_type,
            "prepared_targets": prepared_targets,
        }

    def _find_best_target_match_for_scan(self, detected_embedding: numpy.ndarray, prepared_targets: list[tuple[str, float, numpy.ndarray]]) -> str | None:
        """Computes the cosine distance between a detected face and all prepared target identities."""
        best_target = None
        highest_sim = -1.0
        for target_face_id, threshold, target_embedding in prepared_targets:
            sim = self.main_window.function_worker.findCosineDistance(detected_embedding, target_embedding)
            if sim >= threshold and sim > highest_sim:
                highest_sim = sim
                best_target = target_face_id
        return best_target

    def _build_issue_scan_target_embedding(self, target_face: Any, recognition_model: str, similarity_type: str) -> numpy.ndarray:
        """Extracts the high-dimensional feature vector (embedding) for a UI target face using TensorRT/CUDA."""
        cropped_face = getattr(target_face, "cropped_face", None)
        if not isinstance(cropped_face, numpy.ndarray) or cropped_face.size == 0:
            return numpy.array([])
        image = numpy.ascontiguousarray(cropped_face)
        image_uint8 = image if image.dtype == numpy.uint8 else image.astype("uint8", copy=False)
        image_tensor = (
            torch.from_numpy(image_uint8)
            .to(self.main_window.models_processor.device, non_blocking=True)
            .permute(2, 0, 1)
        )
        height, width = image_uint8.shape[:2]
        
        # Hardcoded central alignment points for the tightly cropped UI face image
        full_face_kps = numpy.array(
            [
                [0.3 * width, 0.35 * height],
                [0.7 * width, 0.35 * height],
                [0.5 * width, 0.55 * height],
                [0.35 * width, 0.75 * height],
                [0.65 * width, 0.75 * height],
            ],
            dtype=numpy.float32,
        )
        face_emb, _ = self.main_window.function_worker.run_recognize_direct(
            image_tensor, full_face_kps, similarity_type, recognition_model
        )
        return face_emb if isinstance(face_emb, numpy.ndarray) else numpy.array([])

    def prepare_issue_scan_target_faces_snapshot(
        self,
        scan_ranges: list[tuple[int, int]],
        base_control: ControlTypes,
        base_params: FacesParametersTypes,
        control_defaults_snapshot: Optional[ControlTypes] = None,
    ) -> IssueScanTargetSnapshot:
        """
        Deep copies the target faces and pre-calculates embeddings for all required recognition models.
        Thread Safety: Prevents crashes if the user deletes a target face in the UI mid-scan.
        """
        live_target_faces = dict(self.main_window.target_faces)
        if not live_target_faces:
            return {}

        scan_segments = self._build_issue_scan_state_segments(
            scan_ranges, base_control, base_params, live_target_faces, control_defaults_snapshot
        )
        # Determine all required embedding formats across all segments (some markers might change the model)
        required_embedding_modes = {
            (str(local_control.get("RecognitionModelSelection", "arcface_128")), str("Auto"))
            for _start_frame, _end_frame, local_control, _local_params in scan_segments
        }
        if not required_embedding_modes:
            required_embedding_modes = {("arcface_128", "Auto")}

        target_faces_snapshot: IssueScanTargetSnapshot = {}
        for target_id, target_face in live_target_faces.items():
            embeddings_by_model: IssueScanTargetEmbeddings = {}
            for recognition_model, similarity_type in sorted(required_embedding_modes):
                model_embeddings = embeddings_by_model.setdefault(recognition_model, {})
                model_embeddings[similarity_type] = self._build_issue_scan_target_embedding(
                    target_face, recognition_model, similarity_type
                )
            target_faces_snapshot[str(target_id)] = {
                "face_id": str(getattr(target_face, "face_id", target_id)),
                "embeddings_by_model": embeddings_by_model,
            }
        return target_faces_snapshot

    def scan_issue_frames(
        self,
        progress_callback=None,
        issue_found_callback=None,
        is_cancelled=None,
        scan_ranges: Optional[List[Tuple[int, int]]] = None,
        target_height: Optional[int] = None,
        base_control: Optional[dict] = None,
        base_params: Optional[dict] = None,
        target_faces_snapshot: Optional[IssueScanTargetSnapshot] = None,
        control_defaults_snapshot: Optional[dict] = None,
        reset_frame_number: Optional[int] = None,
    ) -> Optional[dict]:
        """
        The core analytical loop. Runs high-speed sequential detection on specified frame ranges
        to identify frames where target faces are lost or identification thresholds fail.
        
        VRAM Management: Explicitly handles torch.Tensor creation and destruction per frame 
        to ensure zero memory leaks over long video scans.
        """
        # --- 1. SETUP & VALIDATION ---
        scan_ranges = scan_ranges or self._get_issue_scan_ranges()
        unsupported_reason = self.get_issue_scan_unavailable_reason(
            base_control if base_control is not None else self.main_window.control,
            scan_ranges=scan_ranges,
            markers=getattr(self.main_window, "markers", None),
            fallback_control=getattr(self.main_window, "control", None),
        )
        if unsupported_reason:
            raise RuntimeError(unsupported_reason)

        if not self.media_path:
            raise RuntimeError("Media path is not provided to the scanner.")

        capture = cv2.VideoCapture(self.media_path)
        if not capture or not capture.isOpened():
            raise RuntimeError("Could not open the selected video for scanning.")

        # Ignore manually dropped frames so we don't flag them as "issues"
        dropped_frames_snapshot = {int(frame) for frame in getattr(self.main_window, "dropped_frames", set())}
        total_frames = misc_helpers.count_issue_scan_frames(scan_ranges, dropped_frames_snapshot)
        
        # Deep copy all control variables to achieve total isolation from UI modifications
        base_control = cast(ControlTypes, self._filter_scan_control(copy.deepcopy(base_control if base_control is not None else self.main_window.control)))
        base_params = cast(FacesParametersTypes, self._filter_scan_face_params(copy.deepcopy(base_params if base_params is not None else self.main_window.parameters)))
        initial_target_height = target_height if target_height is not None else self._get_target_input_height_for_control(base_control)
        
        if target_faces_snapshot is None:
            target_faces_snapshot = self.prepare_issue_scan_target_faces_snapshot(
                scan_ranges, base_control, base_params, cast(Optional[ControlTypes], control_defaults_snapshot)
            )
        else:
            target_faces_snapshot = cast(IssueScanTargetSnapshot, dict(target_faces_snapshot))

        # Snapshot the current SequentialDetector state. 
        # Since this scanner runs in a background thread, we must not permanently mutate 
        # the user's live tracking state. We will restore these in the 'finally' block.
        previous_last_detected_faces = copy.deepcopy(self.sequential_detector.last_detected_faces)
        previous_smoothed_kps = copy.deepcopy(self.sequential_detector._smoothed_kps)
        previous_smoothed_dense_kps = copy.deepcopy(self.sequential_detector._smoothed_dense_kps)
        previous_smoothed_dense_kps_203 = copy.deepcopy(self.sequential_detector._smoothed_dense_kps_203)
        is_master_edit_snapshot = self.main_window.editFacesButton.isChecked()

        total_frames_scanned = 0
        tracking_enabled = False
        issue_frames_by_face: dict[str, set[int]] = {str(face_id): set() for face_id in target_faces_snapshot.keys()}

        try:
            self._reset_issue_scan_sequential_state()
            scan_segments = self._build_issue_scan_state_segments(
                scan_ranges, base_control, base_params, target_faces_snapshot, cast(Optional[ControlTypes], control_defaults_snapshot)
            )
            
            tracking_enabled = any(
                bool(local_control.get("FaceTrackingEnableToggle", False))
                for _start_frame, _end_frame, local_control, _local_params in scan_segments
            )
            if tracking_enabled:
                self.main_window.function_worker.reset_face_tracker()
            
            previous_segment_tracking_enabled: Optional[bool] = None
            previous_segment_bytetrack_config = None

            def emit_progress(frame_number: int) -> None:
                if progress_callback:
                    progress_callback(total_frames_scanned, total_frames, frame_number)

            def emit_issue(face_id: str, frame_number: int) -> None:
                normalized_face_id = str(face_id)
                face_frames = issue_frames_by_face.setdefault(normalized_face_id, set())
                normalized_frame = int(frame_number)
                if normalized_frame in face_frames:
                    return
                face_frames.add(normalized_frame)
                if issue_found_callback:
                    issue_found_callback(normalized_face_id, normalized_frame)

            def build_result(cancelled: bool) -> dict[str, Any]:
                faces_with_issues = sum(1 for frames in issue_frames_by_face.values() if frames)
                return {
                    "issue_frames_by_face": {face_id: sorted(frames) for face_id, frames in issue_frames_by_face.items()},
                    "frames_scanned": total_frames_scanned,
                    "faces_with_issues": faces_with_issues,
                    "cancelled": cancelled,
                }

            # --- 2. CORE LOOP ---
            for start_frame, end_frame, local_control, local_params in scan_segments:
                segment_has_resize_state = any(key in local_control for key in ("GlobalInputResizeToggle", "GlobalInputResizeSizeSelection"))
                segment_target_height = self._get_target_input_height_for_control(local_control) if segment_has_resize_state else None
                if not segment_has_resize_state and segment_target_height is None:
                    segment_target_height = initial_target_height
                
                current_segment_tracking_enabled = bool(local_control.get("FaceTrackingEnableToggle", False))
                current_segment_bytetrack_config = self._get_issue_scan_bytetrack_config(local_control)
                
                # Reset ByteTrack if we switch segments and tracker configuration changes
                if (current_segment_tracking_enabled and previous_segment_tracking_enabled is False) or \
                   (current_segment_tracking_enabled and previous_segment_bytetrack_config is not None and 
                    previous_segment_bytetrack_config[0] and current_segment_bytetrack_config != previous_segment_bytetrack_config):
                    self.main_window.function_worker.reset_face_tracker()
                    self._reset_issue_scan_sequential_state()
                
                match_context = self._prepare_issue_scan_match_context(local_control, local_params, target_faces_snapshot)
                misc_helpers.seek_frame(capture, start_frame)
                frame_number = start_frame

                while frame_number <= end_frame:
                    if is_cancelled and is_cancelled():
                        return build_result(True)
                        
                    # Skip manually dropped frames to save processing time
                    if frame_number in dropped_frames_snapshot:
                        next_frame = frame_number + 1
                        while next_frame <= end_frame and next_frame in dropped_frames_snapshot:
                            next_frame += 1
                        misc_helpers.seek_frame(capture, next_frame)
                        frame_number = next_frame
                        continue

                    ret, frame_bgr = misc_helpers.read_frame(capture, self.media_rotation, preview_target_height=segment_target_height)
                    if not ret or not isinstance(frame_bgr, numpy.ndarray):
                        # Treat unreadable frames as absolute issues for all faces
                        for face_id in issue_frames_by_face:
                            emit_issue(face_id, frame_number)
                        misc_helpers.seek_frame(capture, frame_number + 1)
                        total_frames_scanned += 1
                        emit_progress(frame_number)
                        frame_number += 1
                        continue

                    # Pre-process numpy array for fast PyTorch ingestion
                    frame_rgb = numpy.ascontiguousarray(frame_bgr[..., ::-1])
                    frame_rgb_uint8 = frame_rgb if frame_rgb.dtype == numpy.uint8 else frame_rgb.astype("uint8", copy=False)
                    
                    # Memory Critical: explicitly allocate tensor on correct target device
                    frame_tensor = (
                        torch.from_numpy(frame_rgb_uint8)
                        .to(self.main_window.models_processor.device, non_blocking=True)
                        .permute(2, 0, 1)
                    )

                    # Trigger the sequential tracker
                    bboxes, kpss_5, _, _ = self.sequential_detector.run(
                        frame_rgb=frame_rgb,
                        local_control_for_worker=local_control,
                        local_params_for_worker=local_params,
                        is_master_edit_active=is_master_edit_snapshot,
                        frame_tensor=frame_tensor,
                        detector_control_override=local_control,
                        frame_number=frame_number,
                    )
                    
                    detected_embeddings: list[numpy.ndarray] = []
                    if isinstance(bboxes, numpy.ndarray) and bboxes.shape[0] > 0 and isinstance(kpss_5, numpy.ndarray) and kpss_5.shape[0] > 0:
                        max_faces = min(bboxes.shape[0], kpss_5.shape[0])
                        recognition_model = match_context["recognition_model"]
                        similarity_type = match_context["similarity_type"]
                        
                        # Generate ArcFace feature vectors for all detected targets in the frame
                        for face_index in range(max_faces):
                            face_kps = kpss_5[face_index]
                            face_bbox = bboxes[face_index]
                            if not misc_helpers.is_detected_face_eligible_for_matching(face_kps, face_bbox, FrameWorker._MIN_FACE_PIXELS):
                                continue
                            face_emb, _ = self.main_window.function_worker.run_recognize_direct(
                                frame_tensor, face_kps, similarity_type, recognition_model,
                            )
                            if isinstance(face_emb, numpy.ndarray) and face_emb.size > 0:
                                detected_embeddings.append(face_emb)
                    
                    # --- 3. VRAM CLEANUP ---
                    # Strictly delete the heavy frame tensor per frame to prevent CUDA Out of Memory errors
                    del frame_tensor

                    # --- 4. IDENTIFICATION MATH ---
                    matched_face_ids: set[str] = set()
                    prepared_targets = match_context["prepared_targets"]
                    for detected_embedding in detected_embeddings:
                        best_target_face_id = self._find_best_target_match_for_scan(detected_embedding, prepared_targets)
                        if best_target_face_id is not None:
                            matched_face_ids.add(best_target_face_id)

                    # Flag issues: if a target was required but not found in 'matched_face_ids', record it
                    for face_id in issue_frames_by_face:
                        if face_id not in matched_face_ids:
                            emit_issue(face_id, frame_number)
                    
                    total_frames_scanned += 1
                    emit_progress(frame_number)
                    frame_number += 1
                
                previous_segment_tracking_enabled = current_segment_tracking_enabled
                previous_segment_bytetrack_config = current_segment_bytetrack_config

            return build_result(False)
        
        # --- 5. TEARDOWN ---
        finally:
            # Safely restore the real-time tracking states back to the live UI instance
            self.sequential_detector.last_detected_faces = previous_last_detected_faces
            self.sequential_detector._smoothed_kps = previous_smoothed_kps
            self.sequential_detector._smoothed_dense_kps = previous_smoothed_dense_kps
            self.sequential_detector._smoothed_dense_kps_203 = previous_smoothed_dense_kps_203

            if tracking_enabled:
                self.main_window.function_worker.reset_face_tracker()
            misc_helpers.release_capture(capture)