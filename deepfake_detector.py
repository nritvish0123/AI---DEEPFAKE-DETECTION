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
MAX_FRAMES = 5000  # Limit frames
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
            return "Error: Invalid video file"

        # Extract Audio from Video
        video = VideoFileClip(video_path)
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
            # Load Audio for Analysis
            y, sr = librosa.load(audio_path)

            # Detect "Synthetic Silence"
            stft = np.abs(librosa.stft(y))
            db = librosa.amplitude_to_db(stft, ref=np.max)

            silence_threshold = -60  # dB
            silent_parts = db[db < silence_threshold]

            if len(silent_parts) > 0:
                avg_silence_db = np.mean(silent_parts)
                print(f"Noise Floor in Silence: {avg_silence_db:.2f} dB")

                if avg_silence_db < -80:
                    return "High Risk: Synthetic Silence Detected"

            return "Low Risk: Natural Room Tone Detected"
        finally:
            # Clean up temp audio file
            try:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
            except Exception:
                pass

    except Exception as e:
        return f"Error: {str(e)}"

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found!")
    print("Please download the face landmarker model from:")
    print("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
    exit(1)

# Model path
model_path = MODEL_PATH

options = FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

face_landmarker = FaceLandmarker.create_from_options(options)

# Get video path from command line or use default
import sys
video_path = sys.argv[1] if len(sys.argv) > 1 else "my_face.mp4"

if not os.path.exists(video_path):
    print(f"Error: Video file '{video_path}' not found!")
    exit(1)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file!")
    exit(1)

blink_count = 0
is_blinking = False
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS) or 30
start_time = time.time()

# Analyze audio
audio_verdict = analyze_audio(video_path)
print(f"Audio Analysis: {audio_verdict}")

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

        cv2.putText(frame, f"Blinks: {blink_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, audio_verdict, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Combined Deepfake Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Frame processing error: {str(e)}")
        continue

cap.release()
cv2.destroyAllWindows()
face_landmarker.close()

print(f"Final Blink Count: {blink_count}")
print(f"Overall Verdict: {audio_verdict}")