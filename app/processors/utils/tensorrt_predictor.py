import numpy as np
import torch
from collections import OrderedDict
import platform
from queue import Queue
from threading import Lock
from typing import Dict, Any, OrderedDict as OrderedDictType

try:
    from torch.cuda import nvtx
    import tensorrt as trt
    import ctypes
except ModuleNotFoundError:
    pass

# Dizionario per la conversione dei tipi di dati numpy a torch
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
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

if 'trt' in globals():
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
else:
    TRT_LOGGER = None


class TensorRTPredictor:
    def __init__(self, **kwargs) -> None:
        self.device = kwargs.get("device", 'cuda')
        self.debug = kwargs.get("debug", False)
        self.pool_size = kwargs.get("pool_size", 10)

        custom_plugin_path = kwargs.get("custom_plugin_path", None)
        if custom_plugin_path is not None:
            try:
                if platform.system().lower() == 'linux':
                    ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL)
                else:
                    ctypes.CDLL(custom_plugin_path, mode=ctypes.RTLD_GLOBAL, winmode=0)
            except Exception as e:
                raise RuntimeError(f"Errore nel caricamento del plugin personalizzato: {e}")

        engine_path = kwargs.get("model_path", None)
        if not engine_path:
            raise ValueError("Il parametro 'model_path' è obbligatorio.")

        try:
            with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                engine_data = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_data)
        except Exception as e:
            raise RuntimeError(f"Errore nella deserializzazione dell'engine: {e}")

        if self.engine is None:
            raise RuntimeError("La deserializzazione dell'engine è fallita.")
            
        self.context_pool = Queue(maxsize=self.pool_size)
        self.lock = Lock()
        for _ in range(self.pool_size):
            # NOUS NE PRÉ-ALLOUONS PLUS DE BUFFERS ICI
            context = self.engine.create_execution_context()
            self.context_pool.put(context)

    def predict_async(self, bindings: Dict[str, torch.Tensor], stream: torch.cuda.Stream) -> None:
        """
        Exécute l'inférence asynchrone en écrivant directement dans les tenseurs fournis.
        :param bindings: Dictionnaire mappant les noms de tous les tenseurs (entrées ET sorties) à leurs objets torch.Tensor.
        :param stream: Le stream CUDA pour l'exécution.
        """
        context = self.context_pool.get()

        try:
            nvtx.range_push("set_bindings_and_execute_async")
            
            # Lier les adresses de tous les tenseurs fournis
            for name, tensor in bindings.items():
                # Pour les entrées avec des shapes dynamiques, il faut informer le contexte
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    if -1 in self.engine.get_tensor_shape(name):
                         context.set_input_shape(name, tensor.shape)
                context.set_tensor_address(name, tensor.data_ptr())
            
            # Lancer l'exécution asynchrone avec v3 (qui utilise les adresses déjà définies)
            noerror = context.execute_async_v3(stream.cuda_stream)
            if not noerror:
                raise RuntimeError("ERROR: asynchronous inference failed.")

            nvtx.range_pop()

        finally:
            self.context_pool.put(context)

    def cleanup(self) -> None:
        if hasattr(self, 'engine') and self.engine is not None:
            del self.engine
            self.engine = None

        if hasattr(self, 'context_pool') and self.context_pool is not None:
            while not self.context_pool.empty():
                context = self.context_pool.get()
                if context is not None:
                    del context
            self.context_pool = None

    def __del__(self) -> None:
        self.cleanup()
