import sounddevice as sd
import numpy as np
import whisper

model = whisper.load_model("base")

def record_audio(duration=5, fs=16000):
    print(f"Recording audio for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is done
    audio = np.squeeze(audio)
    print("Audio recording complete")
    return audio, fs

def transcribe(audio, fs):
    print("Transcribing audio...")
    
    # Whisper expects float32 numpy array at 16000 Hz sample rate
    # Resample if needed, here assuming fs=16000 already
    # Convert numpy float32 to the format whisper expects (1D float32 array)

    result = model.transcribe(audio, fp16=False)
    text = result["text"].strip()
    print(f" result: {text}")
    return text 