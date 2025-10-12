import os
import sys
import logging
import platform
import ctypes
from pathlib import Path
from typing import Optional, Union

try:
    import tensorrt as trt

    # Initialize the TensorRT logger for global use
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    # Initialize TensorRT plugins
    trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")
except ModuleNotFoundError:
    # Placeholder if tensorrt is not installed
    TRT_LOGGER = None
    print("Warning: The tensorrt module was not found. The code will not run.")

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    Uses the recommended API (separate Builder/Engine/Serialization).
    """

    def __init__(
        self,
        verbose: bool = False,
        custom_plugin_path: Optional[str] = None,
        builder_optimization_level: int = 3,
    ):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param custom_plugin_path: Path to the custom plugin library (DLL or SO).
        """
        if TRT_LOGGER is None:
            raise RuntimeError("TensorRT is not installed or could not be imported.")

        if verbose:
            TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE

        # Load custom plugins if specified
        if custom_plugin_path is not None:
            log.info(f"Loading custom plugin from: {custom_plugin_path}")
            if platform.system().lower() == "linux":
                ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL)
            else:
                # Adding winmode=0 for compatibility on Windows
                ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL, winmode=0)

        # Build the TensorRT builder and configuration using the same logger
        self.builder = trt.Builder(TRT_LOGGER)
        self.config = self.builder.create_builder_config()

        # Set the optimization level (default 3)
        self.config.builder_optimization_level = builder_optimization_level

        # Set the workspace memory pool limit to 3 GB
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 * (2**30))

        # Attributes will be set in create_network
        self.network: Optional[trt.INetworkDefinition] = None
        self.parser: Optional[trt.OnnxParser] = None
        self.batch_size: Optional[int] = None

    def create_network(
        self, onnx_path: str, opt_shapes: Optional[dict[str, tuple]] = None
    ):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        :param opt_shapes: Dictionary of optimization shapes (for dynamic inputs).
                           Ex: {'input_name': (min_shape, opt_shape, max_shape)}
        """
        if self.network is not None:
            log.warning("A network already exists. Resetting.")

        # Use the EXPLICIT_BATCH flag for compatibility
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, TRT_LOGGER)

        onnx_path = os.path.realpath(onnx_path)
        log.info(f"Loading ONNX file: {onnx_path}")

        try:
            with open(onnx_path, "rb") as f:
                if not self.parser.parse(f.read()):
                    log.error("Failed to load ONNX file: %s", onnx_path)
                    for error_idx in range(self.parser.num_errors):
                        log.error(self.parser.get_error(error_idx))
                    sys.exit(1)
        except Exception as e:
            log.error(f"Error opening or parsing the ONNX file: {e}")
            sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("TensorRT Network Description")

        # Handle optimization profile for dynamic shapes
        profile = self.builder.create_optimization_profile()

        for net_input in inputs:
            log.info(
                f"Input '{net_input.name}' | Shape: {net_input.shape} | Dtype: {net_input.dtype}"
            )

            # In EXPLICIT_BATCH API, the first dimension is the batch size
            if self.batch_size is None:
                self.batch_size = net_input.shape[0]
            elif self.batch_size != net_input.shape[0]:
                log.warning(
                    f"Inconsistent batch size detected ({self.batch_size} vs {net_input.shape[0]})."
                )

            if opt_shapes and net_input.name in opt_shapes:
                min_shape, opt_shape, max_shape = opt_shapes[net_input.name]
                log.info(
                    f"Adding optimization profile for '{net_input.name}': Min={min_shape}, Opt={opt_shape}, Max={max_shape}"
                )
                profile.set_shape(net_input.name, min_shape, opt_shape, max_shape)
            elif net_input.shape[0] == -1:
                # If batch is dynamic (-1) but no shapes are provided, it's an error.
                log.error(
                    f"Input '{net_input.name}' has a dynamic batch size (-1). Optimization shapes must be provided via `opt_shapes`."
                )
                sys.exit(1)

        for net_output in outputs:
            log.info(
                f"Output '{net_output.name}' | Shape: {net_output.shape} | Dtype: {net_output.dtype}"
            )

        # Add the profile to the configuration ONLY if it's used.
        if opt_shapes:  # If opt_shapes was provided and populated, we add the profile.
            log.info("Adding optimization profile to configuration.")
            self.config.add_optimization_profile(profile)
        else:
            # If the network is fully static (which is the case for shape (1, 3, 256, 256)),
            # no optimization profile is strictly necessary.
            pass

    def create_engine(self, engine_path: str, precision: str = "fp16") -> bool:
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, 'fp32', 'fp16' or 'int8'.
        :return: True if building and serialization succeed, False otherwise.
        """
        if self.network is None:
            log.error(
                "TensorRT network has not been created. Call `create_network` first."
            )
            return False

        engine_path = os.path.realpath(engine_path)
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        log.info(f"Building {precision.upper()} Engine in {engine_path}")

        if precision == "fp16":
            if self.builder.platform_has_fast_fp16:
                self.config.set_flag(trt.BuilderFlag.FP16)
            else:
                log.warning(
                    "FP16 is not supported natively on this platform/device. Using FP32."
                )
        elif precision == "int8":
            # INT8 requires an IInt8Calibrator, which complicates the example.
            if self.builder.platform_has_fast_int8:
                self.config.set_flag(trt.BuilderFlag.INT8)
            else:
                log.warning(
                    "INT8 is not supported natively on this platform/device. Using FP32."
                )
        elif precision == "fp32":
            pass  # Default

        # 1. Build the Serialized Engine object.
        # CORRECTION: Utiliser build_serialized_network() à la place de build_engine() pour la compatibilité API.
        log.info("Building engine (this may take a few minutes)...")
        serialized_engine = self.builder.build_serialized_network(
            self.network, self.config
        )

        # 2. Check for build error
        if serialized_engine is None:
            log.error("Error building the TensorRT engine!")
            return False

        # 3. Write the serialized engine to disk
        try:
            with open(engine_path, "wb") as f:
                log.info(f"Serializing engine to file: {engine_path}")
                f.write(serialized_engine)
        except Exception as e:
            log.error(f"Error writing the serialized engine to disk: {e}")
            return False

        log.info("TensorRT engine build successful.")
        return True


def change_extension(
    file_path: Union[str, Path], new_extension: str, version: Optional[str] = None
) -> str:
    """
    Change the extension of the file path and optionally prepend a version.
    """
    new_extension = new_extension.lstrip(".")
    p = Path(file_path)

    if version:
        new_file_path = p.with_suffix(f".{version}.{new_extension}")
    else:
        new_file_path = p.with_suffix(f".{new_extension}")

    return str(new_file_path)


def onnx_to_trt(
    onnx_model_path: str,
    trt_model_path: Optional[str] = None,
    precision: str = "fp16",
    custom_plugin_path: Optional[str] = None,
    verbose: bool = False,
    opt_shapes: Optional[dict[str, tuple]] = None,
) -> None:
    """
    Main function to convert an ONNX model to a TensorRT engine.
    :param onnx_model_path: Path to the input ONNX model.
    :param trt_model_path: Path to the output TensorRT engine (optional).
    :param precision: Precision ('fp32', 'fp16', 'int8').
    :param custom_plugin_path: Path to the custom plugin library.
    :param verbose: Enables verbose TensorRT mode.
    :param opt_shapes: Dictionary of optimization shapes for dynamic inputs.
    """
    if trt_model_path is None:
        if TRT_LOGGER is None:
            # Fallback if tensorrt is not imported
            trt_version = "unknown"
        else:
            trt_version = trt.__version__
        trt_model_path = change_extension(onnx_model_path, "trt", version=trt_version)

    builder = EngineBuilder(verbose=verbose, custom_plugin_path=custom_plugin_path)
    print(onnx_model_path)
    print(trt_model_path)
    builder.create_network(onnx_model_path, opt_shapes=opt_shapes)
    builder.create_engine(trt_model_path, precision)
