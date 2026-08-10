import os
import math
import json
import subprocess
from typing import Dict, Any, Optional, Mapping, Tuple, List
import numpy


class FFmpegEncoder:
    """
    Handles FFmpeg subprocess lifecycle, argument generation, and raw frame encoding.
    This class isolates OS-level subprocess management from the main video processing loop
    to prevent thread blocking and simplify recording logic.
    """

    def __init__(self) -> None:
        self.recording_sp: Optional[subprocess.Popen] = None
        self.frames_written: int = 0
        self._source_metrics_cache: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _parse_ffprobe_fps(rate_text: Any) -> Optional[float]:
        """Parse ffprobe frame-rate strings such as '30000/1001' safely."""
        if rate_text is None:
            return None
        try:
            text = str(rate_text).strip()
            if not text:
                return None
            if "/" in text:
                num_s, den_s = text.split("/", 1)
                num = float(num_s)
                den = float(den_s)
                if den == 0:
                    return None
                value = num / den
            else:
                value = float(text)
            return value if value > 0 else None
        except Exception:
            return None

    def probe_source_video_metrics(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Probe source video metrics needed for quality matching.
        Returns a dictionary with keys: bit_rate, width, height, fps, codec_name.
        """
        if not file_path or not os.path.isfile(file_path):
            return None

        # Return from cache if available to prevent redundant blocking I/O calls
        if file_path in self._source_metrics_cache:
            return self._source_metrics_cache[file_path]

        try:
            args = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type,codec_name,width,height,bit_rate,avg_frame_rate,r_frame_rate:format=bit_rate",
                file_path,
            ]
            result = subprocess.run(args, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return None

            probe_data = json.loads(result.stdout)
            video_stream = next(
                (
                    s
                    for s in probe_data.get("streams", [])
                    if s.get("codec_type") == "video"
                ),
                None,
            )
            if not isinstance(video_stream, dict):
                return None

            width = int(video_stream.get("width") or 0)
            height = int(video_stream.get("height") or 0)

            bit_rate_raw = video_stream.get("bit_rate") or probe_data.get(
                "format", {}
            ).get("bit_rate")
            bit_rate = float(bit_rate_raw) if bit_rate_raw else 0.0

            fps = self._parse_ffprobe_fps(video_stream.get("avg_frame_rate"))
            if not fps:
                fps = self._parse_ffprobe_fps(video_stream.get("r_frame_rate"))

            if width <= 0 or height <= 0 or not fps or bit_rate <= 0:
                return None

            metrics = {
                "bit_rate": bit_rate,
                "width": float(width),
                "height": float(height),
                "fps": float(fps),
                "codec_name": str(video_stream.get("codec_name") or "").lower(),
            }
            self._source_metrics_cache[file_path] = metrics
            return metrics
        except Exception as e:
            print(f"[WARN] Failed to probe source metrics for {file_path}: {e}")
            return None

    @staticmethod
    def _source_codec_to_hevc_factor(codec_name: str) -> float:
        """Map source codec efficiency relative to HEVC for quality matching."""
        codec = (codec_name or "").lower()
        if codec in {"hevc", "h265"}:
            return 1.00
        if codec in {"h264", "avc"}:
            return 0.78
        if codec == "av1":
            return 1.28
        if codec == "vp9":
            return 1.18
        if codec in {"mpeg2video", "mpeg4", "msmpeg4v3"}:
            return 0.68
        return 0.90

    def get_adaptive_recording_quality(
        self,
        control: Mapping[str, Any],
        quality_value: int,
        output_width: int,
        output_height: int,
        source_metrics: Optional[Dict[str, Any]] = None,
        output_fps: Optional[float] = None,
    ) -> int:
        """Auto-compute CQ/CRF from source metrics to keep perceived quality close."""
        if not (
            control.get("FFMpegOptionsToggle", False)
            and control.get("FFAutoMatchSourceQualityToggle", False)
        ):
            return quality_value

        if not source_metrics:
            print(
                "[INFO] Source-quality auto match enabled, but probe failed. Using manual Quality unchanged."
            )
            return quality_value

        src_w = max(1.0, source_metrics["width"])
        src_h = max(1.0, source_metrics["height"])
        src_fps = max(0.001, source_metrics["fps"])
        src_bitrate = max(1.0, source_metrics["bit_rate"])
        src_codec = str(source_metrics.get("codec_name", "") or "").lower()
        out_fps = float(output_fps) if output_fps and output_fps > 0 else src_fps

        src_bpppf = src_bitrate / (src_w * src_h * src_fps)
        src_pixels = src_w * src_h
        out_pixels = float(max(1, output_width) * max(1, output_height))
        scale_ratio = out_pixels / src_pixels

        codec_factor = self._source_codec_to_hevc_factor(src_codec)
        target_bpppf = src_bpppf * codec_factor
        temporal_ratio = max(0.5, min(2.0, out_fps / src_fps))
        target_bpppf *= temporal_ratio**0.35

        if scale_ratio > 1.0:
            up_steps = math.log2(scale_ratio)
            target_bpppf *= min(1.35, 1.0 + 0.15 * up_steps)
        elif scale_ratio < 1.0:
            down_steps = math.log2(1.0 / max(scale_ratio, 1e-6))
            target_bpppf *= max(0.70, 1.0 - 0.20 * down_steps)

        if target_bpppf >= 0.25:
            auto_quality = 14
        elif target_bpppf >= 0.16:
            auto_quality = 16
        elif target_bpppf >= 0.11:
            auto_quality = 18
        elif target_bpppf >= 0.08:
            auto_quality = 20
        elif target_bpppf >= 0.055:
            auto_quality = 22
        elif target_bpppf >= 0.038:
            auto_quality = 24
        elif target_bpppf >= 0.028:
            auto_quality = 26
        elif target_bpppf >= 0.020:
            auto_quality = 28
        elif target_bpppf >= 0.014:
            auto_quality = 30
        else:
            auto_quality = 33

        adapted_quality = max(12, min(36, int(auto_quality)))

        print(
            "[INFO] Source-quality auto match: "
            f"source={src_w:.0f}x{src_h:.0f}@{src_fps:.3f} "
            f"codec={src_codec} bitrate={src_bitrate / 1_000_000:.3f}Mbps "
            f"src_bpppf={src_bpppf:.5f} target_bpppf={target_bpppf:.5f} "
            f"out_fps={out_fps:.3f} temporal_ratio={temporal_ratio:.3f}, "
            f"manual_quality={quality_value} auto_quality={adapted_quality}"
        )
        return adapted_quality

    def start_process(
        self,
        output_filename: str,
        frame_width: int,
        frame_height: int,
        fps: float,
        control: Mapping[str, Any],
        is_segment: bool = False,
        media_path: Optional[str] = None,
        start_time_sec: float = 0.0,
        end_time_sec: float = 0.0,
    ) -> bool:
        """
        Builds the FFmpeg command and opens the subprocess.
        """
        if fps <= 0:
            print("[ERROR] Invalid FPS provided to encoder.")
            return False

        # Apply enhancer dimension scaling
        if control.get("FrameEnhancerEnableToggle"):
            enhancer_type = control.get("FrameEnhancerTypeSelection", "")
            if enhancer_type in ("RealEsrgan-x2-Plus", "BSRGan-x2"):
                frame_height *= 2
                frame_width *= 2
            elif enhancer_type in (
                "RealEsrgan-x4-Plus",
                "BSRGan-x4",
                "UltraSharp-x4",
                "UltraMix-x4",
                "RealEsr-General-x4v3",
            ):
                frame_height *= 4
                frame_width *= 4

        frame_height_down = frame_height
        frame_width_down = frame_width
        if control.get("FrameEnhancerDownToggle"):
            if frame_width != 1920 or frame_height != 1080:
                frame_width_down_mult = frame_width / 1920
                frame_height_down = math.ceil(frame_height / frame_width_down_mult) & ~1
                frame_width_down = 1920

        # Quality Adaptation
        source_metrics = (
            self.probe_source_video_metrics(media_path) if media_path else None
        )
        ffquality = self.get_adaptive_recording_quality(
            control=control,
            quality_value=int(control.get("FFQualitySlider", 20)),
            output_width=frame_width_down
            if control.get("FrameEnhancerDownToggle")
            else frame_width,
            output_height=frame_height_down
            if control.get("FrameEnhancerDownToggle")
            else frame_height,
            source_metrics=source_metrics,
            output_fps=fps,
        )

        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{frame_width}x{frame_height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
        ]

        if is_segment and media_path:
            args.extend(
                [
                    "-ss",
                    str(start_time_sec),
                    "-to",
                    str(end_time_sec),
                    "-i",
                    media_path,
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0?",
                    "-c:a",
                    "aac",
                    "-shortest",
                ]
            )

        # Video codec args
        if control.get("HDREncodeToggle"):
            args.extend(
                [
                    "-c:v",
                    "libx265",
                    "-profile:v",
                    "main10",
                    "-preset",
                    str(control.get("FFPresetsHDRSelection", "medium")),
                    "-pix_fmt",
                    "yuv420p10le",
                    "-x265-params",
                    f"crf={ffquality}:vbv-bufsize=10000:vbv-maxrate=10000:selective-sao=0:no-sao=1:strong-intra-smoothing=0:rect=0:aq-mode={int(control.get('FFSpatialAQToggle', 0))}:t-aq={int(control.get('FFTemporalAQToggle', 0))}:hdr-opt=1:repeat-headers=1:colorprim=bt2020:range=limited:transfer=smpte2084:colormatrix=bt2020nc:master-display='G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)':max-cll=1000,400",
                ]
            )
        else:
            args.extend(
                [
                    "-c:v",
                    "hevc_nvenc",
                    "-preset",
                    str(control.get("FFPresetsSDRSelection", "p4")),
                    "-profile:v",
                    "main10",
                    "-cq",
                    str(ffquality),
                    "-pix_fmt",
                    "yuv420p10le",
                    "-colorspace",
                    "rgb",
                    "-color_primaries",
                    "bt709",
                    "-color_trc",
                    "bt709",
                    "-spatial-aq",
                    str(int(control.get("FFSpatialAQToggle", 0))),
                    "-temporal-aq",
                    str(int(control.get("FFTemporalAQToggle", 0))),
                    "-tier",
                    "high",
                    "-tag:v",
                    "hvc1",
                ]
            )

        target_matrix = "bt2020nc" if control.get("HDREncodeToggle") else "bt709"
        scale_params = f"in_range=pc:out_range=tv:out_color_matrix={target_matrix}"

        if control.get("FrameEnhancerDownToggle"):
            args.extend(
                [
                    "-vf",
                    f"scale={frame_width_down}x{frame_height_down}:{scale_params}:flags=lanczos+accurate_rnd+full_chroma_int",
                ]
            )
        else:
            args.extend(["-vf", f"scale={scale_params}"])

        args.append(output_filename)

        try:
            self.recording_sp = subprocess.Popen(
                args, stdin=subprocess.PIPE, bufsize=-1
            )
            self.frames_written = 0
            return True
        except FileNotFoundError:
            print(
                "[ERROR] FFmpeg command not found. Ensure FFmpeg is installed and in system PATH."
            )
            return False
        except Exception as e:
            print(f"[ERROR] Failed to start FFmpeg subprocess: {e}")
            return False

    def write_frame(self, frame: numpy.ndarray) -> bool:
        """Writes a BGR numpy array to the FFmpeg stdin pipe."""
        if (
            self.recording_sp
            and self.recording_sp.stdin
            and not self.recording_sp.stdin.closed
        ):
            try:
                self.recording_sp.stdin.write(frame.tobytes())
                self.frames_written += 1
                return True
            except OSError as e:
                print(f"[WARN] Error writing frame to FFmpeg stdin: {e}")
                return False
        return False

    def close_process(self, timeout: int = 120) -> None:
        """Safely closes the stdin pipe and waits for the FFmpeg process to finalize."""
        if not self.recording_sp:
            return

        # 1. Graceful Shutdown Request (Send EOF via stdin)
        if self.recording_sp.stdin and not self.recording_sp.stdin.closed:
            try:
                self.recording_sp.stdin.close()
            except OSError as e:
                print(f"[WARN] Error closing FFmpeg stdin: {e}")

        # 2. Wait for natural finalization
        try:
            # Wait up to 'timeout' seconds for FFmpeg to safely flush buffers and write the MOOV atom.
            # Crucial for 4K/8K/VR180 where I/O flushing takes time.
            self.recording_sp.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(
                f"[WARN] FFmpeg subprocess timed out after {timeout}s. Attempting graceful terminate..."
            )

            # 3. Escalation Step 1: SIGTERM (Polite request to stop)
            self.recording_sp.terminate()
            try:
                # Give FFmpeg 5 seconds to respond to the terminate signal and write headers
                self.recording_sp.wait(timeout=5)
                print("[INFO] FFmpeg closed cleanly after terminate signal.")
            except subprocess.TimeoutExpired:
                # 4. Escalation Step 2: SIGKILL (Forceful destruction)
                print(
                    "[ERROR] FFmpeg ignored terminate signal and is hanging. Forcing kill (SIGKILL)."
                )
                self.recording_sp.kill()
                self.recording_sp.wait()
        except Exception as e:
            print(f"[ERROR] Error waiting for FFmpeg subprocess: {e}")

        self.recording_sp = None

    def is_running(self) -> bool:
        """Check if the subprocess is currently active."""
        return self.recording_sp is not None and self.recording_sp.poll() is None


class FFmpegPostProcessor:
    """
    Handles stateless post-processing operations via FFmpeg:
    Audio extraction, audio concatenation, and fallback video-only muxing.
    """

    @staticmethod
    def validate_audio_file(audio_file_path: str) -> bool:
        """Validate that an audio file can be properly decoded by FFmpeg."""
        if not os.path.exists(audio_file_path):
            print(f"[ERROR] Audio file does not exist: {audio_file_path}")
            return False

        try:
            args = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                audio_file_path,
            ]
            result = subprocess.run(args, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                print(f"[WARN] ffprobe failed for {audio_file_path}: {result.stderr}")
                return False

            probe_data = json.loads(result.stdout)
            audio_streams = [
                s
                for s in probe_data.get("streams", [])
                if s.get("codec_type") == "audio"
            ]

            if not audio_streams:
                print(f"[WARN] No audio stream found in {audio_file_path}")
                return False

            format_info = probe_data.get("format", {})
            duration = format_info.get("duration")
            if duration is None or float(duration) <= 0:
                print(f"[WARN] Invalid or zero duration in {audio_file_path}")
                return False

            print(f"[INFO] Audio validation passed: {duration}s duration")
            return True

        except subprocess.TimeoutExpired:
            print(f"[WARN] Audio validation timed out for {audio_file_path}")
            return False
        except json.JSONDecodeError:
            print(f"[WARN] Invalid ffprobe output for {audio_file_path}")
            return False
        except Exception as e:
            print(f"[WARN] Audio validation failed for {audio_file_path}: {e}")
            return False

    @staticmethod
    def extract_audio_segments(
        media_path: str,
        fps: float,
        segments: List[Tuple[int, int]],
        temp_audio_dir: str,
        frame_origin: int = 0,
        time_offset_sec: float = 0.0,
    ) -> Tuple[bool, List[str]]:
        """Extract audio from the original media for each frame segment."""
        audio_files = []
        for idx, (start_frame, end_frame) in enumerate(segments):
            if fps > 0:
                start_time = time_offset_sec + max(
                    0.0, (start_frame - frame_origin) / fps
                )
                end_time = time_offset_sec + max(
                    0.0, ((end_frame + 1) - frame_origin) / fps
                )
            else:
                start_time = time_offset_sec
                end_time = time_offset_sec

            if start_time >= end_time:
                continue

            audio_file = os.path.join(temp_audio_dir, f"audio_segment_{idx:04d}.m4a")
            audio_files.append(audio_file)

            args = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-err_detect",
                "ignore_err",
                "-i",
                media_path,
                "-ss",
                str(start_time),
                "-to",
                str(end_time),
                "-vn",
                "-map",
                "0:a:0?",
                "-af",
                "aresample=async=1:first_pts=0",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-y",
                audio_file,
            ]

            try:
                print(
                    f"[INFO] Extracting audio segment {idx + 1}/{len(segments)}: {start_time:.3f}s → {end_time:.3f}s"
                )
                subprocess.run(args, check=True, capture_output=True, text=True)

                if not FFmpegPostProcessor.validate_audio_file(audio_file):
                    print(
                        f"[WARN] Validation failed for segment {idx + 1}, retrying extraction once"
                    )
                    subprocess.run(args, check=True, capture_output=True, text=True)
                    if not FFmpegPostProcessor.validate_audio_file(audio_file):
                        print(
                            f"[ERROR] Retried segment {idx + 1} is still invalid after validation"
                        )
                        for audio in audio_files:
                            try:
                                os.remove(audio)
                            except OSError:
                                pass
                        return False, []

                print(f"[INFO] Segment {idx + 1} extracted successfully")
            except Exception as e:
                print(f"[ERROR] Failed to extract audio segment {idx + 1}: {e}")
                for audio in audio_files:
                    try:
                        os.remove(audio)
                    except OSError:
                        pass
                return False, []

        print(f"[INFO] All {len(segments)} audio segment(s) extracted successfully")
        return True, audio_files

    @staticmethod
    def concatenate_audio_segments(
        audio_files: List[str], temp_audio_dir: str
    ) -> Optional[str]:
        """Concatenate multiple audio files into a single audio file."""
        if not audio_files:
            return None
        if len(audio_files) == 1:
            return audio_files[0]

        concat_file = os.path.join(temp_audio_dir, "concat_manifest.txt")
        try:
            with open(concat_file, "w") as f:
                for audio_file in audio_files:
                    abs_path = os.path.abspath(audio_file)
                    formatted_path = abs_path.replace("\\", "/")
                    f.write(f"file '{formatted_path}'\n")
        except OSError as e:
            print(f"[ERROR] Failed to create concat manifest: {e}")
            return None

        output_audio = os.path.join(temp_audio_dir, "audio_concatenated.m4a")
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-vn",
            "-af",
            "aresample=async=1:first_pts=0",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-y",
            output_audio,
        ]

        try:
            print(f"[INFO] Concatenating {len(audio_files)} audio segment(s)...")
            subprocess.run(args, check=True)
            print("[INFO] ✓ Successfully concatenated audio segments")
            return output_audio
        except Exception as e:
            print(f"[ERROR] Failed to concatenate audio segments: {e}")
            return None

    @staticmethod
    def write_video_only_output(source_video: str, output_video: str) -> bool:
        """Fallback writer: produce a playable video-only output when audio handling fails."""
        if not source_video or not os.path.exists(source_video):
            print(f"[ERROR] Video-only fallback source missing: {source_video}")
            return False

        if output_video and os.path.exists(output_video):
            try:
                os.remove(output_video)
            except OSError:
                pass

        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            source_video,
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-y",
            output_video,
        ]

        try:
            subprocess.run(args, check=True)
            print(
                f"[WARN] Audio processing failed; emitted video-only output: {output_video}"
            )
            return True
        except Exception as e:
            print(f"[ERROR] Video-only remux fallback failed: {e}")
            return False

    @staticmethod
    def concatenate_segments_video_only(
        list_file_path: str, final_file_path: str
    ) -> bool:
        """Fallback concatenation for segment mode when audio concat fails."""
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file_path,
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-y",
            final_file_path,
        ]

        try:
            subprocess.run(args, check=True)
            print(
                f"[WARN] Segment audio concat failed; emitted video-only output: {final_file_path}"
            )
            return True
        except Exception as e:
            print(f"[ERROR] Segment video-only fallback concat failed: {e}")
            return False
