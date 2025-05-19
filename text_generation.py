import sounddevice as sd
import numpy as np
import whisper
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyttsx3

# Load Whisper model for speech-to-text
whisper_model = whisper.load_model("base")

# Load local Qwen model
# Load Qwen model from Hugging Face
print("🚀 Loading Qwen model from Hugging Face...")
model_id = "Qwen/Qwen1.5-0.5B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cpu")
device = torch.device("cpu")

# Load inventory CSV
df = pd.read_csv("inventory.csv")

# Initialize TTS engine (pyttsx3)
engine = pyttsx3.init()

def speak(text):
    print(f"🤖 AI: {text}")
    engine.say(text)
    engine.runAndWait()

# Record audio
def record_audio(duration=5, fs=16000):
    print(f"🎤 Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("✅ Audio recorded")
    return np.squeeze(audio)

# Transcribe using Whisper
def transcribe_audio(audio):
    print("📝 Transcribing...")
    result = whisper_model.transcribe(audio, fp16=False)
    text = result["text"].strip()
    print(f"🗣️ You said: {text}")
    return text

# Check inventory
def retrieve_inventory_info(question):
    items = df['item'].str.lower().tolist()
    matched = [item for item in items if item in question.lower()]
    if not matched:
        return ""
    rows = df[df['item'].str.lower().isin(matched)]
    info = ""
    for _, row in rows.iterrows():
        info += f"{row['item']} is in {row['location']} with quantity {row['quantity']}.\n"
    return info.strip()

# Generate response using Qwen
def generate_answer(question):
    context = retrieve_inventory_info(question)
    if context:
        prompt = f"<|im_start|>user\nAnswer using the inventory info below:\n\n{context}\n\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt = f"<|im_start|>user\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\n"

    print(f"\n📨 Prompt to Qwen:\n{prompt}\n")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    return response.strip()

# === Main Loop ===
if __name__ == "__main__":
    audio = record_audio()
    question = transcribe_audio(audio)
    if question:
        response = generate_answer(question)
        speak(response)
    else:
        print("❗ No voice input detected.")
