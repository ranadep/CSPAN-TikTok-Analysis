import cv2
import mediapipe as mp
import numpy as np


def distance(point1, point2):
    return np.sqrt(
        (point2[0] - point1[0]) ** 2
        + (point2[1] - point1[1]) ** 2
    )


def extract_body_features(
    video_path,
    model_path="models/pose_landmarker.task",
    sample_fps=5
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    duration_seconds = total_frames / video_fps

    frame_step = max(
        1,
        round(video_fps / sample_fps)
    ) #frames to skip for specified analysis

    sampled_frames = 0
    visible_frames = 0

    previous_left = None
    previous_right = None

    wrist_speeds = []
    frame_wrist_speeds = []

    left_positions = []
    right_positions = []

    frame_number = 0

    # -------------------------
    # Create MediaPipe model
    # -------------------------

    #creating shortcuts
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    #configuring detector
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path
        ),

        running_mode=VisionRunningMode.VIDEO, #confirms sequential frames (video) input

        num_poses=1, #only analyze one person

        min_pose_detection_confidence=0.5,

        min_pose_presence_confidence=0.5,

        min_tracking_confidence=0.5
    )


    #creates PoseLandmarker object as landmarker, and close resources after
    with PoseLandmarker.create_from_options(options) as landmarker:

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                break

            # Skip frames so we process about 5 FPS
            if frame_number % frame_step != 0:
                frame_number += 1
                continue

            sampled_frames += 1
            sample_frame_speeds = []

            # OpenCV BGR (using NumPy) -> RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Convert NumPy array to MediaPipe Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, #creates image in standard sRGB format
                data=rgb_frame #sets the data to the RGB array
            )

            # MediaPipe VIDEO mode needs a timestamp
            timestamp_ms = int(
                (frame_number / video_fps) * 1000 #converts frame number to milliseconds
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # -------------------------
            # Did MediaPipe find a body?
            # -------------------------

            if result.pose_landmarks:

                landmarks = result.pose_landmarks[0]

                # MediaPipe Pose landmark numbers:
                # 15 = left wrist
                # 16 = right wrist

                left_wrist = landmarks[15]
                right_wrist = landmarks[16]

                # Tasks landmarks contain visibility scores
                left_visible = left_wrist.visibility >= 0.5
                right_visible = right_wrist.visibility >= 0.5

                if left_visible or right_visible:
                    visible_frames += 1

                # -------------------------
                # LEFT WRIST
                # -------------------------

                if left_visible:

                    current_left = (
                        left_wrist.x,
                        left_wrist.y
                    )

                    left_positions.append(current_left)

                    if previous_left is not None:
                        previous_position, previous_timestamp = previous_left
                        elapsed_seconds = (timestamp_ms - previous_timestamp) / 1000

                        if elapsed_seconds > 0:
                            speed = distance(
                                previous_position,
                                current_left
                            ) / elapsed_seconds
                            wrist_speeds.append(speed)
                            sample_frame_speeds.append(speed)

                    previous_left = (current_left, timestamp_ms)

                else:
                    previous_left = None

                # -------------------------
                # RIGHT WRIST
                # -------------------------

                if right_visible:

                    current_right = (
                        right_wrist.x,
                        right_wrist.y
                    )

                    right_positions.append(current_right)

                    if previous_right is not None:
                        previous_position, previous_timestamp = previous_right #assigning tuple values to variables
                        elapsed_seconds = (timestamp_ms - previous_timestamp) / 1000

                        if elapsed_seconds > 0:
                            speed = distance(
                                previous_position,
                                current_right
                            ) / elapsed_seconds
                            wrist_speeds.append(speed)
                            sample_frame_speeds.append(speed)

                    previous_right = (current_right, timestamp_ms)

                else:
                    previous_right = None

            else:

                # No body found
                previous_left = None
                previous_right = None

            frame_wrist_speeds.append(max(sample_frame_speeds, default=0))

            frame_number += 1

    cap.release()

    # =================================
    # FINAL FEATURE CALCULATIONS
    # =================================

    if sampled_frames > 0:

        hands_visible_pct = (
            visible_frames / sampled_frames
        ) * 100

    else:

        hands_visible_pct = 0


    if wrist_speeds:

        wrist_speed_mean = np.mean(
            wrist_speeds
        )

        wrist_speed_max = np.max(
            wrist_speeds
        )

    else:

        wrist_speed_mean = 0
        wrist_speed_max = 0


    # -------------------------
    # Gesture bursts
    # -------------------------

    if wrist_speeds:

        threshold = np.mean(frame_wrist_speeds) + np.std(frame_wrist_speeds)

        burst_count = 0
        currently_in_burst = False

        for speed in frame_wrist_speeds:

            if (
                speed > threshold
                and not currently_in_burst
            ):

                burst_count += 1
                currently_in_burst = True

            elif speed <= threshold:

                currently_in_burst = False

    else:

        burst_count = 0


    duration_minutes = duration_seconds / 60

    if duration_minutes > 0:

        gesture_bursts_per_min = (
            burst_count / duration_minutes
        )

    else:

        gesture_bursts_per_min = 0


    # -------------------------
    # Gesture range
    # -------------------------

    all_positions = (
        left_positions
        + right_positions
    )

    if all_positions:

        xs = [
            point[0]
            for point in all_positions
        ]

        ys = [
            point[1]
            for point in all_positions
        ]

        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)

        gesture_range = np.sqrt(
            x_range ** 2
            + y_range ** 2
        )

    else:

        gesture_range = 0


    return {

        "hands_visible_pct":
            float(hands_visible_pct),

        "wrist_speed_mean":
            float(wrist_speed_mean),

        "wrist_speed_max":
            float(wrist_speed_max),

        "gesture_bursts_per_min":
            float(gesture_bursts_per_min),

        "gesture_range":
            float(gesture_range)
    }