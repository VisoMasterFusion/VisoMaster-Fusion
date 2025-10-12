import math
import torch
from skimage import transform as trans
from torchvision.transforms import v2
from app.processors.utils import faceutil
import numpy as np
from numpy.linalg import norm as l2norm
from typing import TYPE_CHECKING
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from app.processors.models_processor import ModelsProcessor


def _debug_print_tensor(tensor: torch.Tensor, name: str):
    """Helper function to print tensor statistics for debugging."""
    if not isinstance(tensor, torch.Tensor):
        print(f"[CanonSwap Debug] {name}: Not a tensor, but type {type(tensor)}")
        return
    print(
        f"[CanonSwap Debug] {name}: "
        f"shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, "
        f"device={tensor.device}, "
        f"min={tensor.min().item():.6f}, "
        f"max={tensor.max().item():.6f}, "
        f"mean={tensor.mean().item():.6f}"
    )


class FaceSwappers:
    def __init__(self, models_processor: "ModelsProcessor"):
        self.models_processor = models_processor
        self.resize_112 = v2.Resize(
            (112, 112), interpolation=v2.InterpolationMode.BILINEAR, antialias=False
        )

    def run_recognize_direct(
        self, img, kps, similarity_type="Opal", arcface_model="Inswapper128ArcFace"
    ):
        if not self.models_processor.models[arcface_model]:
            self.models_processor.models[arcface_model] = (
                self.models_processor.load_model(arcface_model)
            )

        if arcface_model == "CSCSArcFace":
            embedding, cropped_image = self.recognize_cscs(img, kps)
        else:
            embedding, cropped_image = self.recognize(
                arcface_model, img, kps, similarity_type=similarity_type
            )

        return embedding, cropped_image

    def run_recognize(
        self, img, kps, similarity_type="Opal", face_swapper_model="Inswapper128"
    ):
        arcface_model = self.models_processor.get_arcface_model(face_swapper_model)
        return self.run_recognize_direct(img, kps, similarity_type, arcface_model)

    def recognize(self, arcface_model, img, face_kps, similarity_type):
        if similarity_type == "Optimal":
            # Find transform & Transform
            img, _ = faceutil.warp_face_by_face_landmark_5(
                img,
                face_kps,
                mode="arcfacemap",
                interpolation=v2.InterpolationMode.BILINEAR,
            )
        elif similarity_type == "Pearl":
            # Find transform
            dst = self.models_processor.arcface_dst.copy()
            dst[:, 0] += 8.0

            tform = trans.SimilarityTransform()
            tform.estimate(face_kps, dst)

            # Transform
            img = v2.functional.affine(
                img,
                tform.rotation * 57.2958,
                (tform.translation[0], tform.translation[1]),
                tform.scale,
                0,
                center=(0, 0),
            )
            img = v2.functional.crop(img, 0, 0, 128, 128)
            img = v2.Resize(
                (112, 112), interpolation=v2.InterpolationMode.BILINEAR, antialias=False
            )(img)
        else:
            # Find transform
            tform = trans.SimilarityTransform()
            tform.estimate(face_kps, self.models_processor.arcface_dst)

            # Transform
            img = v2.functional.affine(
                img,
                tform.rotation * 57.2958,
                (tform.translation[0], tform.translation[1]),
                tform.scale,
                0,
                center=(0, 0),
            )
            img = v2.functional.crop(img, 0, 0, 112, 112)

        if arcface_model == "Inswapper128ArcFace":
            cropped_image = img.permute(1, 2, 0).clone()
            if img.dtype == torch.uint8:
                img = img.to(torch.float32)  # Convert to float32 if uint8
            img = torch.sub(img, 127.5)
            img = torch.div(img, 127.5)
        elif arcface_model == "SimSwapArcFace" or arcface_model == "CanonSwapArcFace":
            cropped_image = img.permute(1, 2, 0).clone()
            if img.dtype == torch.uint8:
                img = torch.div(img.to(torch.float32), 255.0)
            img = v2.functional.normalize(
                img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=False
            )
        else:
            cropped_image = img.permute(1, 2, 0).clone()  # 112,112,3
            if img.dtype == torch.uint8:
                img = img.to(torch.float32)  # Convert to float32 if uint8
            # Normalize
            img = torch.div(img, 127.5)
            img = torch.sub(img, 1)

        # Prepare data and find model parameters
        img = torch.unsqueeze(img, 0).contiguous()
        input_name = self.models_processor.models[arcface_model].get_inputs()[0].name

        outputs = self.models_processor.models[arcface_model].get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)

        io_binding = self.models_processor.models[arcface_model].io_binding()
        io_binding.bind_input(
            name=input_name,
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=img.size(),
            buffer_ptr=img.data_ptr(),
        )

        for i in range(len(output_names)):
            io_binding.bind_output(output_names[i], self.models_processor.device)

        # Sync and run model
        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models[arcface_model].run_with_iobinding(io_binding)

        # Return embedding
        return np.array(io_binding.copy_outputs_to_cpu()).flatten(), cropped_image

    def preprocess_image_cscs(self, img, face_kps):
        tform = trans.SimilarityTransform()
        tform.estimate(face_kps, self.models_processor.FFHQ_kps)

        temp = v2.functional.affine(
            img,
            tform.rotation * 57.2958,
            (tform.translation[0], tform.translation[1]),
            tform.scale,
            0,
            center=(0, 0),
        )
        temp = v2.functional.crop(temp, 0, 0, 512, 512)

        image = self.resize_112(temp)

        cropped_image = image.permute(1, 2, 0).clone()
        if image.dtype == torch.uint8:
            image = torch.div(image.to(torch.float32), 255.0)

        image = v2.functional.normalize(
            image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False
        )

        # Ritorna l'immagine e l'immagine ritagliata
        return torch.unsqueeze(
            image, 0
        ).contiguous(), cropped_image  # (C, H, W) e (H, W, C)

    def recognize_cscs(self, img, face_kps):
        # Usa la funzione di preprocessamento
        img, cropped_image = self.preprocess_image_cscs(img, face_kps)

        io_binding = self.models_processor.models["CSCSArcFace"].io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=img.size(),
            buffer_ptr=img.data_ptr(),
        )
        io_binding.bind_output(name="output", device_type=self.models_processor.device)

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()

        self.models_processor.models["CSCSArcFace"].run_with_iobinding(io_binding)

        output = io_binding.copy_outputs_to_cpu()[0]
        embedding = torch.from_numpy(output).to("cpu")
        embedding = torch.nn.functional.normalize(embedding, dim=-1, p=2)
        embedding = embedding.numpy().flatten()

        embedding_id = self.recognize_cscs_id_adapter(img, None)
        embedding = embedding + embedding_id

        return embedding, cropped_image

    def recognize_cscs_id_adapter(self, img, face_kps):
        if not self.models_processor.models["CSCSIDArcFace"]:
            self.models_processor.models["CSCSIDArcFace"] = (
                self.models_processor.load_model("CSCSIDArcFace")
            )

        # Use preprocess_image_cscs when face_kps is not None. When it is None img is already preprocessed.
        if face_kps is not None:
            img, _ = self.preprocess_image_cscs(img, face_kps)

        io_binding = self.models_processor.models["CSCSIDArcFace"].io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=img.size(),
            buffer_ptr=img.data_ptr(),
        )
        io_binding.bind_output(name="output", device_type=self.models_processor.device)

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()

        self.models_processor.models["CSCSIDArcFace"].run_with_iobinding(io_binding)

        output = io_binding.copy_outputs_to_cpu()[0]
        embedding_id = torch.from_numpy(output).to("cpu")
        embedding_id = torch.nn.functional.normalize(embedding_id, dim=-1, p=2)

        return embedding_id.numpy().flatten()

    def calc_swapper_latent_cscs(self, source_embedding):
        latent = source_embedding.reshape((1, -1))
        return latent

    def run_swapper_cscs(self, image, embedding, output):
        if not self.models_processor.models["CSCS"]:
            self.models_processor.models["CSCS"] = self.models_processor.load_model(
                "CSCS"
            )

        io_binding = self.models_processor.models["CSCS"].io_binding()
        io_binding.bind_input(
            name="input_1",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="input_2",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models["CSCS"].run_with_iobinding(io_binding)

    def calc_inswapper_latent(self, source_embedding):
        n_e = source_embedding / l2norm(source_embedding)
        latent = n_e.reshape((1, -1))
        latent = np.dot(latent, self.models_processor.emap)
        latent /= np.linalg.norm(latent)
        return latent

    def run_inswapper(self, image, embedding, output):
        if not self.models_processor.models["Inswapper128"]:
            self.models_processor.models["Inswapper128"] = (
                self.models_processor.load_model("Inswapper128")
            )

        io_binding = self.models_processor.models["Inswapper128"].io_binding()
        io_binding.bind_input(
            name="target",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 128, 128),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="source",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 128, 128),
            buffer_ptr=output.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models["Inswapper128"].run_with_iobinding(io_binding)

    def calc_swapper_latent_ghost(self, source_embedding):
        latent = source_embedding.reshape((1, -1))

        return latent

    def calc_swapper_latent_iss(self, source_embedding, version="A"):
        n_e = source_embedding / l2norm(source_embedding)
        latent = n_e.reshape((1, -1))
        latent = np.dot(latent, self.models_processor.emap)
        latent /= np.linalg.norm(latent)
        return latent

    def run_iss_swapper(self, image, embedding, output, version="A"):
        ISS_MODEL_NAME = f"InStyleSwapper256 Version {version}"
        if not self.models_processor.models[ISS_MODEL_NAME]:
            self.models_processor.models[ISS_MODEL_NAME] = (
                self.models_processor.load_model(ISS_MODEL_NAME)
            )

        io_binding = self.models_processor.models[ISS_MODEL_NAME].io_binding()
        io_binding.bind_input(
            name="target",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="source",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models[ISS_MODEL_NAME].run_with_iobinding(io_binding)

    def calc_swapper_latent_simswap512(self, source_embedding):
        latent = source_embedding.reshape(1, -1)
        # latent /= np.linalg.norm(latent)
        latent = latent / np.linalg.norm(latent, axis=1, keepdims=True)
        return latent

    def run_swapper_simswap512(self, image, embedding, output):
        if not self.models_processor.models["SimSwap512"]:
            self.models_processor.models["SimSwap512"] = (
                self.models_processor.load_model("SimSwap512")
            )

        io_binding = self.models_processor.models["SimSwap512"].io_binding()
        io_binding.bind_input(
            name="input",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="onnx::Gemm_1",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name="output",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 512, 512),
            buffer_ptr=output.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        self.models_processor.models["SimSwap512"].run_with_iobinding(io_binding)

    def run_swapper_ghostface(
        self, image, embedding, output, swapper_model="GhostFace-v2"
    ):
        ghostfaceswap_model, output_name = None, None
        if swapper_model == "GhostFace-v1":
            if not self.models_processor.models["GhostFacev1"]:
                self.models_processor.models["GhostFacev1"] = (
                    self.models_processor.load_model("GhostFacev1")
                )

            ghostfaceswap_model = self.models_processor.models["GhostFacev1"]
            output_name = "781"

        elif swapper_model == "GhostFace-v2":
            if not self.models_processor.models["GhostFacev2"]:
                self.models_processor.models["GhostFacev2"] = (
                    self.models_processor.load_model("GhostFacev2")
                )

            ghostfaceswap_model = self.models_processor.models["GhostFacev2"]
            output_name = "1165"

        elif swapper_model == "GhostFace-v3":
            if not self.models_processor.models["GhostFacev3"]:
                self.models_processor.models["GhostFacev3"] = (
                    self.models_processor.load_model("GhostFacev3")
                )

            ghostfaceswap_model = self.models_processor.models["GhostFacev3"]
            output_name = "1549"

        io_binding = ghostfaceswap_model.io_binding()
        io_binding.bind_input(
            name="target",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=image.data_ptr(),
        )
        io_binding.bind_input(
            name="source",
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 512),
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding.bind_output(
            name=output_name,
            device_type=self.models_processor.device,
            device_id=0,
            element_type=np.float32,
            shape=(1, 3, 256, 256),
            buffer_ptr=output.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        ghostfaceswap_model.run_with_iobinding(io_binding)

    def calc_swapper_latent_canonswap(self, source_embedding):
        # The ID embedding must be L2 normalized to be a unit vector.
        n_e = source_embedding / np.linalg.norm(source_embedding)
        latent = n_e.reshape((1, -1))
        return latent

    def _canonswap_create_deformed_feature(
        self, feature: Tensor, sparse_motions: Tensor
    ) -> Tensor:
        """Replicates the internal grid_sample from DenseMotionNetwork using PyTorch."""
        bs, _, d, h, w = feature.shape
        num_kp = 21  # Constant for CanonSwap model

        feature_repeat = (
            feature.unsqueeze(1).unsqueeze(1).repeat(1, num_kp + 1, 1, 1, 1, 1, 1)
        )
        feature_repeat = feature_repeat.view(bs * (num_kp + 1), -1, d, h, w)
        sparse_motions = sparse_motions.view((bs * (num_kp + 1), d, h, w, -1))

        sparse_deformed = F.grid_sample(
            feature_repeat, sparse_motions, align_corners=False
        )
        sparse_deformed = sparse_deformed.view((bs, num_kp + 1, -1, d, h, w))

        return sparse_deformed

    def _canonswap_calculate_deformation(
        self, sparse_motion: Tensor, mask: Tensor
    ) -> Tensor:
        """Calculates the final deformation field manually using PyTorch."""
        # sparse_motion: (bs, 1+num_kp, d, h, w, 3)
        # mask: (bs, 1+num_kp, d, h, w)

        mask_unsqueezed = mask.unsqueeze(2)  # -> (bs, 1+num_kp, 1, d, h, w)
        sparse_motion_permuted = sparse_motion.permute(
            0, 1, 5, 2, 3, 4
        )  # -> (bs, 1+num_kp, 3, d, h, w)

        deformation_permuted = (sparse_motion_permuted * mask_unsqueezed).sum(
            dim=1
        )  # -> (bs, 3, d, h, w)
        deformation = deformation_permuted.permute(0, 2, 3, 4, 1)  # -> (bs, d, h, w, 3)

        return deformation

    def _canonswap_headpose_pred_to_degree(self, pred: Tensor) -> Tensor:
        """Converts headpose prediction to degrees. Tensor version."""
        device = pred.device
        idx_tensor = torch.arange(66, dtype=torch.float32, device=device)
        pred_softmax = F.softmax(pred, dim=1)
        degree = torch.sum(pred_softmax * idx_tensor, axis=1) * 3 - 97.5
        return degree

    def _canonswap_get_rotation_matrix(
        self, pitch_: Tensor, yaw_: Tensor, roll_: Tensor
    ) -> Tensor:
        """Calculates rotation matrix from pitch, yaw, roll in degrees."""
        # transform to radian
        pitch = pitch_ / 180 * math.pi
        yaw = yaw_ / 180 * math.pi
        roll = roll_ / 180 * math.pi

        device = pitch.device

        if pitch.ndim == 1:
            pitch = pitch.unsqueeze(1)
        if yaw.ndim == 1:
            yaw = yaw.unsqueeze(1)
        if roll.ndim == 1:
            roll = roll.unsqueeze(1)

        bs = pitch.shape[0]
        ones = torch.ones([bs, 1]).to(device)
        zeros = torch.zeros([bs, 1]).to(device)
        x, y, z = pitch, yaw, roll

        rot_x = torch.cat(
            [
                ones,
                zeros,
                zeros,
                zeros,
                torch.cos(x),
                -torch.sin(x),
                zeros,
                torch.sin(x),
                torch.cos(x),
            ],
            dim=1,
        ).reshape([bs, 3, 3])

        rot_y = torch.cat(
            [
                torch.cos(y),
                zeros,
                torch.sin(y),
                zeros,
                ones,
                zeros,
                -torch.sin(y),
                zeros,
                torch.cos(y),
            ],
            dim=1,
        ).reshape([bs, 3, 3])

        rot_z = torch.cat(
            [
                torch.cos(z),
                -torch.sin(z),
                zeros,
                torch.sin(z),
                torch.cos(z),
                zeros,
                zeros,
                zeros,
                ones,
            ],
            dim=1,
        ).reshape([bs, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def _canonswap_transform_keypoint(self, kp_info: dict) -> Tensor:
        """Transforms keypoints using pose, translation, expression, and scale."""
        kp = kp_info["kp"]
        pitch = self._canonswap_headpose_pred_to_degree(kp_info["pitch"])
        yaw = self._canonswap_headpose_pred_to_degree(kp_info["yaw"])
        roll = self._canonswap_headpose_pred_to_degree(kp_info["roll"])

        t, exp, scale = kp_info["t"], kp_info["exp"], kp_info["scale"]
        rot_mat = self._canonswap_get_rotation_matrix(pitch, yaw, roll)
        bs = kp.shape[0]

        kp_transformed = kp @ rot_mat + exp
        kp_transformed = kp_transformed * scale.reshape(bs, 1, 1)
        kp_transformed[..., 0:2] += t[:, :2].reshape(bs, 1, 2)
        return kp_transformed

    def run_canonswap(self, image, embedding, output) -> Tensor:
        """
        Executes the CanonSwap pipeline for a single frame using ONNX io_binding.
        """
        device = self.models_processor.device

        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).to(device)

        embedding = embedding.contiguous()

        image_norm = image.to(torch.float32)
        if image_norm.dim() == 3:
            image_norm = image_norm.unsqueeze(0)
        image_norm = image_norm.contiguous()

        # Load models if not already loaded
        model_names = [
            "CanonSwapMotionExtractor",
            "CanonSwapAppearanceFeatureExtractor",
            "CanonSwapDenseMotionPart1",
            "CanonSwapDenseMotionPart2",
            "CanonSwapSwapModule",
            "CanonSwapRefineModule",
            "CanonSwapWarpingDecoder",
            "CanonSwapSpadeGenerator",
        ]
        models = {}
        for name in model_names:
            if not self.models_processor.models.get(name):
                self.models_processor.models[name] = self.models_processor.load_model(
                    name
                )
            models[name] = self.models_processor.models[name]

        # --- 1. Motion Extraction ---
        kp_info_tensors = {
            "pitch": torch.empty(
                (1, 66), dtype=torch.float32, device=device
            ).contiguous(),
            "yaw": torch.empty(
                (1, 66), dtype=torch.float32, device=device
            ).contiguous(),
            "roll": torch.empty(
                (1, 66), dtype=torch.float32, device=device
            ).contiguous(),
            "t": torch.empty((1, 3), dtype=torch.float32, device=device).contiguous(),
            "exp": torch.empty(
                (1, 63), dtype=torch.float32, device=device
            ).contiguous(),
            "scale": torch.empty(
                (1, 1), dtype=torch.float32, device=device
            ).contiguous(),
            "kp": torch.empty((1, 63), dtype=torch.float32, device=device).contiguous(),
        }
        io_binding_motion = models["CanonSwapMotionExtractor"].io_binding()
        io_binding_motion.bind_input(
            name="input_image",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=image_norm.shape,
            buffer_ptr=image_norm.data_ptr(),
        )
        for name, tensor in kp_info_tensors.items():
            io_binding_motion.bind_output(
                name=name,
                device_type=device,
                device_id=0,
                element_type=np.float32,
                shape=tensor.shape,
                buffer_ptr=tensor.data_ptr(),
            )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        with self.models_processor.model_lock:
            models["CanonSwapMotionExtractor"].run_with_iobinding(io_binding_motion)

        bs = kp_info_tensors["kp"].shape[0]
        kp_info_tensors["kp"] = kp_info_tensors["kp"].reshape(bs, 21, 3)
        kp_info_tensors["exp"] = kp_info_tensors["exp"].reshape(bs, 21, 3)

        # --- 2. Transform Keypoints ---
        x_t = self._canonswap_transform_keypoint(kp_info_tensors).contiguous()
        x_can = (
            kp_info_tensors["scale"].reshape(bs, 1, 1) * kp_info_tensors["kp"]
        ).contiguous()

        # --- 3. Appearance Feature Extraction ---
        feature_volume = torch.empty(
            (1, 32, 16, 64, 64), dtype=torch.float32, device=device
        ).contiguous()
        io_binding_app = models["CanonSwapAppearanceFeatureExtractor"].io_binding()
        io_binding_app.bind_input(
            name="input_image",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=image_norm.shape,
            buffer_ptr=image_norm.data_ptr(),
        )
        io_binding_app.bind_output(
            name="feature_volume",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=feature_volume.shape,
            buffer_ptr=feature_volume.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        with self.models_processor.model_lock:
            models["CanonSwapAppearanceFeatureExtractor"].run_with_iobinding(
                io_binding_app
            )

        # --- 4. First Warp (Driving -> Canonical) ---
        compressed_feature = torch.empty(
            (1, 4, 16, 64, 64), dtype=torch.float32, device=device
        ).contiguous()
        heatmap = torch.empty(
            (1, 22, 1, 16, 64, 64), dtype=torch.float32, device=device
        ).contiguous()
        sparse_motion = torch.empty(
            (1, 22, 16, 64, 64, 3), dtype=torch.float32, device=device
        ).contiguous()
        io_binding_dense1 = models["CanonSwapDenseMotionPart1"].io_binding()
        io_binding_dense1.bind_input(
            name="feature_3d",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=feature_volume.shape,
            buffer_ptr=feature_volume.data_ptr(),
        )
        io_binding_dense1.bind_input(
            name="kp_driving",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=x_can.shape,
            buffer_ptr=x_can.data_ptr(),
        )
        io_binding_dense1.bind_input(
            name="kp_source",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=x_t.shape,
            buffer_ptr=x_t.data_ptr(),
        )
        io_binding_dense1.bind_output(
            name="compressed_feature",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=compressed_feature.shape,
            buffer_ptr=compressed_feature.data_ptr(),
        )
        io_binding_dense1.bind_output(
            name="heatmap",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=heatmap.shape,
            buffer_ptr=heatmap.data_ptr(),
        )
        io_binding_dense1.bind_output(
            name="sparse_motion",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=sparse_motion.shape,
            buffer_ptr=sparse_motion.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        with self.models_processor.model_lock:
            models["CanonSwapDenseMotionPart1"].run_with_iobinding(io_binding_dense1)

        deformed_feature = self._canonswap_create_deformed_feature(
            compressed_feature, sparse_motion
        ).contiguous()

        mask = torch.empty(
            (1, 22, 16, 64, 64), dtype=torch.float32, device=device
        ).contiguous()
        occlusion_map_initial = torch.empty(
            (1, 1, 64, 64), dtype=torch.float32, device=device
        ).contiguous()
        io_binding_dense2 = models["CanonSwapDenseMotionPart2"].io_binding()
        io_binding_dense2.bind_input(
            name="heatmap",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=heatmap.shape,
            buffer_ptr=heatmap.data_ptr(),
        )
        io_binding_dense2.bind_input(
            name="deformed_feature",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=deformed_feature.shape,
            buffer_ptr=deformed_feature.data_ptr(),
        )
        io_binding_dense2.bind_output(
            name="mask",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=mask.shape,
            buffer_ptr=mask.data_ptr(),
        )
        io_binding_dense2.bind_output(
            name="occlusion_map",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=occlusion_map_initial.shape,
            buffer_ptr=occlusion_map_initial.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        with self.models_processor.model_lock:
            models["CanonSwapDenseMotionPart2"].run_with_iobinding(io_binding_dense2)

        deformation_initial = self._canonswap_calculate_deformation(sparse_motion, mask)
        f_can = F.grid_sample(
            feature_volume, deformation_initial, align_corners=False
        ).contiguous()

        # --- 5. Swap and Refine in Canonical Space ---
        f_can_swapped = torch.empty_like(f_can).contiguous()
        io_binding_swap = models["CanonSwapSwapModule"].io_binding()
        io_binding_swap.bind_input(
            name="feature_volume",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=f_can.shape,
            buffer_ptr=f_can.data_ptr(),
        )
        io_binding_swap.bind_input(
            name="id_embedding",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=embedding.shape,
            buffer_ptr=embedding.data_ptr(),
        )
        io_binding_swap.bind_output(
            name="swapped_feature_volume",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=f_can_swapped.shape,
            buffer_ptr=f_can_swapped.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        with self.models_processor.model_lock:
            models["CanonSwapSwapModule"].run_with_iobinding(io_binding_swap)

        f_can_swapped_refined = torch.empty_like(f_can_swapped).contiguous()
        io_binding_refine = models["CanonSwapRefineModule"].io_binding()
        io_binding_refine.bind_input(
            name="feature_volume",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=f_can_swapped.shape,
            buffer_ptr=f_can_swapped.data_ptr(),
        )
        io_binding_refine.bind_output(
            name="refined_feature_volume",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=f_can_swapped_refined.shape,
            buffer_ptr=f_can_swapped_refined.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        with self.models_processor.model_lock:
            models["CanonSwapRefineModule"].run_with_iobinding(io_binding_refine)

        # --- 6. Second Warp (Canonical -> Driving) ---
        compressed_feature_final = torch.empty_like(compressed_feature).contiguous()
        heatmap_final = torch.empty_like(heatmap).contiguous()
        sparse_motion_final = torch.empty_like(sparse_motion).contiguous()
        io_binding_dense3 = models["CanonSwapDenseMotionPart1"].io_binding()
        io_binding_dense3.bind_input(
            name="feature_3d",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=f_can_swapped_refined.shape,
            buffer_ptr=f_can_swapped_refined.data_ptr(),
        )
        io_binding_dense3.bind_input(
            name="kp_driving",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=x_t.shape,
            buffer_ptr=x_t.data_ptr(),
        )
        io_binding_dense3.bind_input(
            name="kp_source",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=x_can.shape,
            buffer_ptr=x_can.data_ptr(),
        )
        io_binding_dense3.bind_output(
            name="compressed_feature",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=compressed_feature_final.shape,
            buffer_ptr=compressed_feature_final.data_ptr(),
        )
        io_binding_dense3.bind_output(
            name="heatmap",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=heatmap_final.shape,
            buffer_ptr=heatmap_final.data_ptr(),
        )
        io_binding_dense3.bind_output(
            name="sparse_motion",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=sparse_motion_final.shape,
            buffer_ptr=sparse_motion_final.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        with self.models_processor.model_lock:
            models["CanonSwapDenseMotionPart1"].run_with_iobinding(io_binding_dense3)

        deformed_feature_final = self._canonswap_create_deformed_feature(
            compressed_feature_final, sparse_motion_final
        ).contiguous()

        mask_final = torch.empty_like(mask).contiguous()
        occlusion_map_final = torch.empty_like(occlusion_map_initial).contiguous()
        io_binding_dense4 = models["CanonSwapDenseMotionPart2"].io_binding()
        io_binding_dense4.bind_input(
            name="heatmap",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=heatmap_final.shape,
            buffer_ptr=heatmap_final.data_ptr(),
        )
        io_binding_dense4.bind_input(
            name="deformed_feature",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=deformed_feature_final.shape,
            buffer_ptr=deformed_feature_final.data_ptr(),
        )
        io_binding_dense4.bind_output(
            name="mask",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=mask_final.shape,
            buffer_ptr=mask_final.data_ptr(),
        )
        io_binding_dense4.bind_output(
            name="occlusion_map",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=occlusion_map_final.shape,
            buffer_ptr=occlusion_map_final.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        with self.models_processor.model_lock:
            models["CanonSwapDenseMotionPart2"].run_with_iobinding(io_binding_dense4)

        deformation_final = self._canonswap_calculate_deformation(
            sparse_motion_final, mask_final
        )
        warped_3d_volume = F.grid_sample(
            f_can_swapped_refined, deformation_final, align_corners=False
        ).contiguous()

        # --- 7. Decode to 2D Feature and then to Image ---
        warped_feature_2d = torch.empty(
            (1, 256, 64, 64), dtype=torch.float32, device=device
        ).contiguous()
        io_binding_warpdec = models["CanonSwapWarpingDecoder"].io_binding()
        io_binding_warpdec.bind_input(
            name="warped_3d_volume",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=warped_3d_volume.shape,
            buffer_ptr=warped_3d_volume.data_ptr(),
        )
        io_binding_warpdec.bind_input(
            name="occlusion_map",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=occlusion_map_final.shape,
            buffer_ptr=occlusion_map_final.data_ptr(),
        )
        io_binding_warpdec.bind_output(
            name="warped_feature_2d",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=warped_feature_2d.shape,
            buffer_ptr=warped_feature_2d.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        with self.models_processor.model_lock:
            models["CanonSwapWarpingDecoder"].run_with_iobinding(io_binding_warpdec)

        output_image_norm = torch.empty(
            (1, 3, 512, 512), dtype=torch.float32, device=device
        ).contiguous()
        io_binding_spade = models["CanonSwapSpadeGenerator"].io_binding()
        io_binding_spade.bind_input(
            name="warped_feature_2d",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=warped_feature_2d.shape,
            buffer_ptr=warped_feature_2d.data_ptr(),
        )
        io_binding_spade.bind_output(
            name="output_image",
            device_type=device,
            device_id=0,
            element_type=np.float32,
            shape=output_image_norm.shape,
            buffer_ptr=output_image_norm.data_ptr(),
        )

        if self.models_processor.device == "cuda":
            torch.cuda.synchronize()
        elif self.models_processor.device != "cpu":
            self.models_processor.syncvec.cpu()
        with self.models_processor.model_lock:
            models["CanonSwapSpadeGenerator"].run_with_iobinding(io_binding_spade)

        # --- 8. Format Output ---
        output_frame = torch.mul(output_image_norm.squeeze(0), 255)

        return output_frame
