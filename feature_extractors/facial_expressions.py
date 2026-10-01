import cv2
import mediapipe as mp
import numpy as np


def distance(point1, point2):
    """Calculate Euclidean distance between two (x, y) points."""
    return np.sqrt(
        (point2[0] - point1[0]) ** 2
        + (point2[1] - point1[1]) ** 2
    )


def extract_face_features(
    video_path,
    model_path="models/face_landmarker.task",
    sample_fps=5
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    frame_step = max(
        1,
        round(video_fps / sample_fps)
    )

    frame_number = 0

    # Store measurements from throughout the video
    head_movements = []
    mouth_open_values = []
    mouth_movements = []
    brow_movements = []
    mouth_width_movements = []
    brow_rising_intervals = 0
    mouth_widening_intervals = 0
    movement_intervals = 0
    previous_measurements = None

    # =========================================
    # CREATE MEDIAPIPE FACE LANDMARKER
    # =========================================

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=model_path
        ),

        running_mode=VisionRunningMode.VIDEO,

        # Only analyze one face
        num_faces=1,

        min_face_detection_confidence=0.5,

        min_face_presence_confidence=0.5,

        min_tracking_confidence=0.5
    )

    with FaceLandmarker.create_from_options(options) as landmarker:

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            # Only process about 5 frames per second
            if frame_number % frame_step != 0:
                frame_number += 1
                continue

            # =========================================
            # PREPARE FRAME FOR MEDIAPIPE
            # =========================================

            # OpenCV uses BGR.
            # MediaPipe expects RGB.
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Convert OpenCV image into a MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # VIDEO mode requires a timestamp
            timestamp_ms = int(
                (frame_number / video_fps) * 1000
            )

            # Run MediaPipe Face Landmarker
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # =========================================
            # CHECK WHETHER A FACE WAS FOUND
            # =========================================

            if result.face_landmarks:

                # We requested max_num_faces=1,
                # so use the first detected face.
                landmarks = result.face_landmarks[0]

                # =====================================
                # FACE WIDTH
                # =====================================

                # Approximate left and right sides
                # of the face.
                left_face = landmarks[234]
                right_face = landmarks[454]

                face_width = distance(
                    (left_face.x, left_face.y),
                    (right_face.x, right_face.y)
                )

                # =====================================
                # 2. MOUTH OPENNESS
                # =====================================

                # Inner upper and lower lips
                upper_lip = landmarks[13]
                lower_lip = landmarks[14]

                mouth_open = distance(
                    (upper_lip.x, upper_lip.y),
                    (lower_lip.x, lower_lip.y)
                )

                # Normalize by face width
                if face_width > 0:
                    mouth_open /= face_width

                mouth_open_values.append(mouth_open)

                # =====================================
                # 3. EYEBROW HEIGHT
                # =====================================

                # Left eyebrow / eye
                left_brow = landmarks[105]
                left_eye = landmarks[159]

                # Right eyebrow / eye
                right_brow = landmarks[334]
                right_eye = landmarks[386]

                brow_elevation = (
                    (left_eye.y - left_brow.y)
                    + (right_eye.y - right_brow.y)
                ) / 2

                # =====================================
                # 4. SMILE MEASUREMENT
                # =====================================

                # Left and right mouth corners
                left_corner = landmarks[61]
                right_corner = landmarks[291]

                mouth_width = distance(
                    (left_corner.x, left_corner.y),
                    (right_corner.x, right_corner.y)
                )

                if face_width > 0:
                    brow_elevation /= face_width
                    mouth_width /= face_width

                nose = landmarks[1]
                timestamp_seconds = timestamp_ms / 1000
                current_measurements = (
                    (nose.x, nose.y),
                    mouth_open,
                    brow_elevation,
                    mouth_width,
                    timestamp_seconds,
                )

                if previous_measurements is not None and face_width > 0:
                    previous_nose, previous_mouth_open, previous_brow, previous_mouth_width, previous_time = previous_measurements
                    elapsed_seconds = timestamp_seconds - previous_time

                    if elapsed_seconds > 0:
                        head_speed = distance(
                            previous_nose,
                            current_measurements[0]
                        ) / face_width / elapsed_seconds
                        mouth_speed = abs(
                            mouth_open - previous_mouth_open
                        ) / elapsed_seconds
                        brow_change = brow_elevation - previous_brow
                        mouth_width_change = mouth_width - previous_mouth_width

                        head_movements.append(head_speed)
                        mouth_movements.append(mouth_speed)
                        brow_movements.append(abs(brow_change) / elapsed_seconds)
                        mouth_width_movements.append(
                            abs(mouth_width_change) / elapsed_seconds
                        )
                        brow_rising_intervals += brow_change > 0
                        mouth_widening_intervals += mouth_width_change > 0
                        movement_intervals += 1

                previous_measurements = current_measurements

            else:

                # Face disappeared.
                # Don't compare its future position
                # against an old position.
                previous_measurements = None

            frame_number += 1

    cap.release()

    # =========================================
    # FINAL FEATURE CALCULATIONS
    # =========================================

    # -----------------------------------------
    # HEAD MOVEMENT MEAN
    # -----------------------------------------

    head_movement_mean = np.mean(head_movements) if head_movements else 0


    # -----------------------------------------
    # MOUTH OPEN MEAN
    # -----------------------------------------

    mouth_open_mean = np.mean(mouth_open_values) if mouth_open_values else 0


    # -----------------------------------------
    # Percent of tracked intervals with upward eyebrow movement.
    # -----------------------------------------

    brow_raise_pct = (
        brow_rising_intervals / movement_intervals * 100
        if movement_intervals else 0
    )


    # -----------------------------------------
    # Percent of tracked intervals with outward mouth-corner movement.
    # -----------------------------------------

    smile_pct = (
        mouth_widening_intervals / movement_intervals * 100
        if movement_intervals else 0
    )


    # -----------------------------------------
    # EXPRESSIVENESS SCORE
    # -----------------------------------------

    movement_magnitudes = (
        head_movements
        + mouth_movements
        + brow_movements
        + mouth_width_movements
    )
    expressiveness_score = (
        float(np.mean(movement_magnitudes))
        if movement_magnitudes else 0
    )


    # =========================================
    # RETURN FEATURES
    # =========================================

    return {

        "head_movement_mean":
            float(head_movement_mean),

        "mouth_open_mean":
            float(mouth_open_mean),

        "brow_raise_pct":
            float(brow_raise_pct),

        "smile_pct":
            float(smile_pct),

        "expressiveness_score":
            float(expressiveness_score)
    }