import cv2
import mediapipe as mp
import numpy as np
from moviepy import VideoFileClip
import librosa
import os
import tempfile
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

# Indices for Eyes (Landmarks)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Security constants
MAX_PROCESSING_TIME = 300  # 5 minutes max
MAX_FRAMES = 10000  # Limit frames to prevent infinite loops
MODEL_PATH = 'face_landmarker.task'

def get_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio safely"""
    try:
        # Vertical coordinates
        p2_p6 = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
        p3_p5 = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])
        # Horizontal coordinate
        p1_p4 = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])
        return (p2_p6 + p3_p5) / (2.0 * p1_p4)
    except (IndexError, TypeError):
        return 1.0  # Default open eye ratio

def analyze_audio(video_path):
    """Analyze audio for synthetic silence with error handling"""
    try:
        # Validate input file
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return 0  # Low risk for invalid files

        # Extract Audio from Video with timeout
        start_time = time.time()
        try:
            video = VideoFileClip(video_path)
            if video.duration > 600:  # 10 minutes max
                return 0  # Skip very long videos
        except Exception:
            return 0  # Audio extraction failed

        # Create secure temp audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_path = temp_audio.name
            try:
                video.audio.write_audiofile(audio_path, codec='pcm_s16le', verbose=False, logger=None)
            except TypeError:
                # Fallback for older moviepy versions
                video.audio.write_audiofile(audio_path, codec='pcm_s16le')
            finally:
                video.close()

        try:
            # Load Audio for Analysis with size check
            if os.path.getsize(audio_path) > 50 * 1024 * 1024:  # 50MB max
                return 0

            y, sr = librosa.load(audio_path, sr=None)

            # Detect "Synthetic Silence"
            stft = np.abs(librosa.stft(y))
            db = librosa.amplitude_to_db(stft, ref=np.max)

            silence_threshold = -60  # dB
            silent_parts = db[db < silence_threshold]

            if len(silent_parts) > 0:
                avg_silence_db = np.mean(silent_parts)
                print(f"Noise Floor in Silence: {avg_silence_db:.2f} dB")

                if avg_silence_db < -80:
                    return 1  # High risk

            return 0  # Low risk
        finally:
            # Clean up temp audio file
            try:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
            except Exception:
                pass

    except Exception as e:
        print(f"Audio analysis error: {str(e)}")
        return 0

def count_blinks(video_path):
    """Count blinks with safety limits and error handling"""
    try:
        # Validate model file
        if not os.path.exists(MODEL_PATH):
            print("Warning: Face landmarker model not found")
            return 0

        # Validate video file
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            return 0

        # Model path
        options = FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )

        face_landmarker = FaceLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0

        blink_count = 0
        is_blinking = False
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30fps
        start_time = time.time()

        while cap.isOpened() and frame_count < MAX_FRAMES:
            # Check timeout
            if time.time() - start_time > MAX_PROCESSING_TIME:
                print("Processing timeout reached")
                break

            success, frame = cap.read()
            if not success or frame is None:
                break

            frame_count += 1
            height, width = frame.shape[:2]

            # Skip invalid frames
            if height < 10 or width < 10:
                continue

            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(frame_count * 1000 / fps)

                results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

                if results and results.face_landmarks:
                    for face_landmarks in results.face_landmarks:
                        if len(face_landmarks) < max(LEFT_EYE + RIGHT_EYE) + 1:
                            continue

                        coords = np.array([[lm.x * width, lm.y * height] for lm in face_landmarks])

                        left_ear = get_ear(coords, LEFT_EYE)
                        right_ear = get_ear(coords, RIGHT_EYE)
                        avg_ear = (left_ear + right_ear) / 2.0

                        if avg_ear < 0.25:
                            if not is_blinking:
                                blink_count += 1
                                is_blinking = True
                        else:
                            is_blinking = False
            except Exception as e:
                # Skip problematic frames
                continue

        cap.release()
        face_landmarker.close()
        return blink_count

    except Exception as e:
        print(f"Blink counting error: {str(e)}")
        return 0

    options = FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )

    face_landmarker = FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)

    blink_count = 0
    is_blinking = False
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(frame_count * 1000 / fps)
        
        results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                coords = np.array([[lm.x * width, lm.y * height] for lm in face_landmarks])
                
                left_ear = get_ear(coords, LEFT_EYE)
                right_ear = get_ear(coords, RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < 0.25:
                    if not is_blinking:
                        blink_count += 1
                        is_blinking = True
                else:
                    is_blinking = False

    cap.release()
    face_landmarker.close()
    return blink_count

def deepfake_final_verdict(video_path):
    """Main detection function with comprehensive error handling"""
    try:
        # Validate input
        if not video_path or not isinstance(video_path, str):
            return 0.1

        if not os.path.exists(video_path):
            return 0.1

        # Analyze audio with timeout
        audio_risk = analyze_audio(video_path)

        # Count blinks with timeout
        blink_count = count_blinks(video_path)

        # Simple logic: if audio high risk, high probability
        # If blink count low (<10), increase risk
        probability = 0.0
        if audio_risk == 1:
            probability += 0.5

        if blink_count < 10:
            probability += 0.3

        # Cap at 1.0
        probability = min(probability, 1.0)

        # If no risks, low probability
        if probability == 0.0:
            probability = 0.1  # Some base

        return probability

    except Exception as e:
        print(f"Detection error: {str(e)}")
        return 0.1  # Default to low risk on error