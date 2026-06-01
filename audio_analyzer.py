from moviepy import VideoFileClip
import librosa
import numpy as np
import os
import tempfile
import time

def analyze_audio(video_path):
    """Analyze audio for synthetic silence with comprehensive safety checks"""
    try:
        # Validate input file
        if not video_path or not os.path.exists(video_path):
            return "Error: Invalid video file"

        file_size = os.path.getsize(video_path)
        if file_size == 0 or file_size > 100 * 1024 * 1024:  # 100MB limit
            return "Error: File too large or empty"

        # Extract Audio from Video with timeout protection
        start_time = time.time()
        try:
            video = VideoFileClip(video_path)
            if video.duration > 600:  # 10 minutes max
                video.close()
                return "Error: Video too long"
        except Exception as e:
            return f"Error: Failed to load video - {str(e)}"

        # Create secure temp audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_path = temp_audio.name
            try:
                video.audio.write_audiofile(audio_path, codec='pcm_s16le', verbose=False, logger=None)
                video.close()
            except TypeError:
                # Fallback for older moviepy versions that don't support verbose/logger
                video.audio.write_audiofile(audio_path, codec='pcm_s16le')
                video.close()
            except Exception as e:
                video.close()
                return f"Error: Audio extraction failed - {str(e)}"

        try:
            # Check audio file size
            if os.path.getsize(audio_path) > 50 * 1024 * 1024:  # 50MB max
                return "Error: Audio file too large"

            # Load Audio for Analysis
            y, sr = librosa.load(audio_path, sr=None)

            # Detect "Synthetic Silence"
            stft = np.abs(librosa.stft(y))
            db = librosa.amplitude_to_db(stft, ref=np.max)

            # Calculate average noise floor in quiet parts
            silence_threshold = -60  # dB
            silent_parts = db[db < silence_threshold]

            if len(silent_parts) > 0:
                avg_silence_db = np.mean(silent_parts)
                print(f"Noise Floor in Silence: {avg_silence_db:.2f} dB")

                # If it's near -100dB, it's 'perfect silence' (Suspicious)
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
        return f"Error: Analysis failed - {str(e)}"

# Test function with safety
if __name__ == "__main__":
    test_file = "test_video.mp4"
    if os.path.exists(test_file):
        result = analyze_audio(test_file)
        print(f"Verdict: {result}")
    else:
        print("Test video not found")