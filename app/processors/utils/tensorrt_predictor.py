import numpy as np
import torch
import platform
from queue import Queue
from threading import Lock
from typing import Dict, Optional

try:
    from torch.cuda import nvtx
    import tensorrt as trt
    import ctypes
except ModuleNotFoundError:
    # These modules are not critical for basic import,
    # but the class will fail to initialize if they are not found.
    pass

# Dictionary for converting numpy data types to torch data types.
# This ensures that tensors created in PyTorch have the correct precision
# corresponding to the data types defined in the ONNX/TensorRT model.
numpy_to_torch_dtype_dict = {
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}
# Handle boolean type compatibility across numpy versions.
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Initialize the TensorRT logger. Setting the severity to ERROR reduces verbose logging.
# A placeholder is used if the 'trt' module is not available.
if "trt" in globals():
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
else:
    TRT_LOGGER = None


class TensorRTPredictor:
    """
    Manages a pool of TensorRT execution contexts for efficient, thread-safe,
    and asynchronous inference.

    This class is designed to handle inference requests from multiple threads by maintaining
    a queue of execution contexts. This avoids the overhead of creating a new context
    for each inference call. It uses TensorRT's zero-copy mechanism by directly binding
    PyTorch tensor memory addresses to the engine's inputs and outputs, which is highly
    efficient for GPU operations.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the TensorRTPredictor.

        Args:
            **kwargs: Keyword arguments for configuration.
                - device (str): The computation device ('cuda' or 'cpu'). Defaults to 'cuda'.
                - debug (bool): Enables debug mode. Defaults to False.
                - pool_size (int): The number of execution contexts to create in the pool. Defaults to 10.
                - custom_plugin_path (str, optional): Path to a custom TensorRT plugin library (.so or .dll).
                - model_path (str): The path to the serialized TensorRT engine file (.trt). This is mandatory.
        """
        self.device = kwargs.get("device", "cuda")
        self.debug = kwargs.get("debug", False)
        self.pool_size = kwargs.get("pool_size", 10)

        # Load custom plugin library if provided. This is necessary for engines
        # that use custom layers not natively supported by TensorRT.
        custom_plugin_path = kwargs.get("custom_plugin_path", None)
        if custom_plugin_path is not None:
            try:
                if platform.system().lower() == "linux":
                    ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL)
                else:
                    # On Windows, winmode=0 is needed for compatibility with certain libraries.
                    ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL, winmode=0)
            except Exception as e:
                raise RuntimeError(f"Error loading the custom plugin: {e}")

        engine_path = kwargs.get("model_path", None)
        if not engine_path:
            raise ValueError("The 'model_path' parameter is mandatory.")

        # Deserialize the TensorRT engine from the file.
        try:
            with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                engine_data = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_data)
        except Exception as e:
            raise RuntimeError(f"Error during engine deserialization: {e}")

        if self.engine is None:
            raise RuntimeError("Engine deserialization failed.")

        # Create a pool of execution contexts. This is a key optimization for multi-threaded
        # applications, as creating contexts is an expensive operation.
        self.context_pool: "Optional[Queue[trt.IExecutionContext]]" = Queue(
            maxsize=self.pool_size
        )
        self.lock = Lock()
        for _ in range(self.pool_size):
            # WE NO LONGER PRE-ALLOCATE BUFFERS HERE.
            # Contexts are created, but input/output buffers are bound dynamically in `predict_async`
            # based on the provided PyTorch tensors.
            context = self.engine.create_execution_context()
            self.context_pool.put(context)

    def predict_async(
        self, bindings: Dict[str, torch.Tensor], stream: torch.cuda.Stream
    ) -> None:
        """
        Executes asynchronous inference by writing directly to the provided tensors.

        This method leverages zero-copy by binding the memory pointers of the input
        and output PyTorch tensors to the TensorRT engine. The inference is queued
        on the specified CUDA stream and returns immediately (non-blocking).

        Args:
            bindings (Dict[str, torch.Tensor]): A dictionary mapping the names of ALL tensors
                                                (both inputs AND outputs) to their torch.Tensor objects.
                                                The output tensors must be pre-allocated with the correct shape and dtype.
            stream (torch.cuda.Stream): The CUDA stream on which to execute the inference.
        """
        if self.context_pool is None:
            raise RuntimeError(
                "The context pool has been cleaned up and is no longer available."
            )

        # Get a free execution context from the pool. This call will block if the pool is empty
        # until a context is returned by another thread.
        context = self.context_pool.get()

        try:
            # Push a range to the NVTX profiler for performance analysis.
            nvtx.range_push("set_bindings_and_execute_async")

            # Bind the memory addresses of all provided tensors to the engine.
            for name, tensor in bindings.items():
                # For inputs with dynamic shapes, the context must be informed of the tensor's shape.
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    if -1 in self.engine.get_tensor_shape(name):
                        context.set_input_shape(name, tensor.shape)
                # Set the memory address for the binding.
                context.set_tensor_address(name, tensor.data_ptr())

            # Launch the asynchronous execution using v3, which relies on the addresses
            # set by `set_tensor_address`. This is a non-blocking call.
            noerror = context.execute_async_v3(stream.cuda_stream)
            if not noerror:
                raise RuntimeError("ERROR: Asynchronous inference failed.")

            # Pop the NVTX range.
            nvtx.range_pop()

        finally:
            # CRITICAL: Return the context to the pool so other threads can use it.
            # This is done in a `finally` block to ensure it happens even if errors occur.
            if self.context_pool is not None:
                self.context_pool.put(context)

    def cleanup(self) -> None:
        """
        Safely cleans up and releases all TensorRT resources (engine and contexts).
        This is crucial to prevent CUDA memory leaks.
        """
        if hasattr(self, "engine") and self.engine is not None:
            del self.engine
            self.engine = None

        if hasattr(self, "context_pool") and self.context_pool is not None:
            while not self.context_pool.empty():
                context = self.context_pool.get()
                if context is not None:
                    del context
            self.context_pool = None

    def __del__(self) -> None:
        """
        Ensures cleanup is called when the object is garbage collected.
        """
        self.cleanup()
