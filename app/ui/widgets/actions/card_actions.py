from typing import TYPE_CHECKING, Dict
import uuid

import numpy
import cv2
import torch
from torchvision.transforms import v2

import app.ui.widgets.actions.common_actions as common_widget_actions
from app.ui.widgets.actions import list_view_actions
import app.helpers.miscellaneous as misc_helpers

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow


def clear_target_faces(main_window: "MainWindow", refresh_frame=True):
    if main_window.video_processor.processing:
        main_window.video_processor.stop_processing()
    main_window.targetFacesList.clear()
    for _, target_face in main_window.target_faces.items():
        target_face.deleteLater()
    main_window.target_faces = {}
    main_window.parameters.clear()

    main_window.selected_target_face_id = None
    # Set Parameter widget values to default
    common_widget_actions.set_widgets_values_using_face_id_parameters(
        main_window=main_window, face_id=False
    )
    if refresh_frame:
        common_widget_actions.refresh_frame(main_window=main_window)


def clear_input_faces(main_window: "MainWindow"):
    main_window.inputFacesList.clear()
    for _, input_face in main_window.input_faces.items():
        input_face.deleteLater()
    main_window.input_faces = {}

    for _, target_face in main_window.target_faces.items():
        target_face.assigned_input_faces = {}
        target_face.calculate_assigned_input_embedding()
    common_widget_actions.refresh_frame(main_window=main_window)


def clear_merged_embeddings(main_window: "MainWindow"):
    main_window.inputEmbeddingsList.clear()
    for _, embed_button in main_window.merged_embeddings.items():
        embed_button.deleteLater()
    main_window.merged_embeddings = {}

    for _, target_face in main_window.target_faces.items():
        target_face.assigned_merged_embeddings = {}
        target_face.calculate_assigned_input_embedding()
    common_widget_actions.refresh_frame(main_window=main_window)


def uncheck_all_input_faces(main_window: "MainWindow"):
    # Uncheck All other input faces
    for _, input_face_button in main_window.input_faces.items():
        input_face_button.setChecked(False)


def uncheck_all_merged_embeddings(main_window: "MainWindow"):
    for _, embed_button in main_window.merged_embeddings.items():
        embed_button.setChecked(False)


def find_target_faces(main_window: "MainWindow"):
    control = main_window.control.copy()
    video_processor = main_window.video_processor
    if video_processor.media_path:
        frame = None
        media_capture = video_processor.media_capture

        if video_processor.file_type == "image":
            frame = misc_helpers.read_image_file(video_processor.media_path)
        elif video_processor.file_type == "video" and media_capture:
            # MODIFICATION: Pass rotation
            ret, frame = misc_helpers.read_frame(
                media_capture, video_processor.media_rotation
            )
            # ---
            media_capture.set(
                cv2.CAP_PROP_POS_FRAMES, video_processor.current_frame_number
            )
        elif video_processor.file_type == "webcam" and media_capture:
            # MODIFICATION: Pass 0 for webcam rotation
            ret, frame = misc_helpers.read_frame(media_capture, 0)
            # ---
            media_capture.set(
                cv2.CAP_PROP_POS_FRAMES, video_processor.current_frame_number
            )

        if frame is not None:
            # Frame must be in RGB format
            frame = frame[..., ::-1]  # Swap the channels from BGR to RGB

            # print(frame)
            img = torch.from_numpy(frame.astype("uint8")).to(
                main_window.models_processor.device
            )
            img = img.permute(2, 0, 1)
            if control["ManualRotationEnableToggle"]:
                img = v2.functional.rotate(
                    img,
                    angle=control["ManualRotationAngleSlider"],
                    interpolation=v2.InterpolationMode.BILINEAR,
                    expand=True,
                )

            _, kpss_5, _ = main_window.models_processor.run_detect(
                img,
                control["DetectorModelSelection"],
                max_num=control["MaxFacesToDetectSlider"],
                score=float(control["DetectorScoreSlider"]) / 100.0,
                input_size=(512, 512),
                use_landmark_detection=control["LandmarkDetectToggle"],
                landmark_detect_mode=control["LandmarkDetectModelSelection"],
                landmark_score=float(control["LandmarkDetectScoreSlider"]) / 100.0,
                from_points=control["DetectFromPointsToggle"],
                rotation_angles=[0]
                if not control["AutoRotationToggle"]
                else [0, 90, 180, 270],
            )

            ret = []
            for face_kps in kpss_5:
                face_emb, cropped_img = (
                    main_window.models_processor.run_recognize_direct(
                        img,
                        face_kps,
                        control["SimilarityTypeSelection"],
                        control["RecognitionModelSelection"],
                    )
                )
                ret.append([face_kps, face_emb, cropped_img, img])

            if ret:
                # Loop through all faces in video frame
                for face in ret:
                    found = False
                    # Check if this face has already been found
                    for face_id, target_face in main_window.target_faces.items():
                        parameters = main_window.parameters[target_face.face_id]
                        threshhold = parameters["SimilarityThresholdSlider"]
                        if main_window.models_processor.findCosineDistance(
                            target_face.get_embedding(
                                str(control["RecognitionModelSelection"])
                            ),
                            face[1],
                        ) >= float(threshhold):
                            found = True
                            break
                    if not found:
                        face_img = face[2].cpu().numpy()
                        face_img = face_img[
                            ..., ::-1
                        ]  # Swap the channels from RGB to BGR
                        face_img = numpy.ascontiguousarray(face_img)
                        # crop = cv2.resize(face[2].cpu().numpy(), (82, 82))
                        pixmap = common_widget_actions.get_pixmap_from_frame(
                            main_window, face_img
                        )

                        # Only store the embedding for the currently selected recognition model
                        embedding_store: Dict[str, numpy.ndarray] = {}
                        selected_recognition_model = control[
                            "RecognitionModelSelection"
                        ]

                        # The embedding for the selected model was already calculated
                        # face[1] holds the embedding calculated using control["RecognitionModelSelection"]
                        embedding_store[str(selected_recognition_model)] = face[1]
                        # --- MODIFICATION END ---

                        face_id = str(uuid.uuid1())

                        # Pass the embedding_store containing only the selected model's embedding
                        list_view_actions.add_media_thumbnail_to_target_faces_list(
                            main_window, face_img, embedding_store, pixmap, face_id
                        )

                        if control.get("KeepInputToggle", False) or control.get(
                            "AutoSwapToggle", False
                        ):
                            new_target_face = main_window.target_faces.get(face_id)
                            if new_target_face:
                                # Assign checked Input Faces
                                for (
                                    input_face_id,
                                    input_face_button,
                                ) in main_window.input_faces.items():
                                    if input_face_button.isChecked():
                                        new_target_face.assigned_input_faces[
                                            input_face_id
                                        ] = input_face_button.embedding_store

                                # Assign checked Embeddings
                                for (
                                    embed_id,
                                    embed_button,
                                ) in main_window.merged_embeddings.items():
                                    if embed_button.isChecked():
                                        new_target_face.assigned_merged_embeddings[
                                            embed_id
                                        ] = embed_button.embedding_store

                                # Recalculate assigned embeddings
                                new_target_face.calculate_assigned_input_embedding()

            # Select the first target face if no target face is already selected
        if main_window.target_faces and not main_window.selected_target_face_id:
            list(main_window.target_faces.values())[0].click()

    if main_window.video_processor.processing:
        main_window.video_processor.stop_processing()
    common_widget_actions.refresh_frame(main_window)

    common_widget_actions.update_gpu_memory_progressbar(main_window)
