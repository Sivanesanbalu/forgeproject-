import speech_recognition as sr

def detect_wake_word(timeout=5):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for wake word...")
        audio = r.listen(source, phrase_time_limit=timeout)

    try:
        text = r.recognize_google(audio).lower()
        print(f"Heard: {text}")
        if "hello assistant" in text or "hey assistant" in text:
            return True
    except Exception as e:
        pass
    return False
