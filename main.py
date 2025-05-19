import wake_word
import speech_to_text
import text_generation
import text_to_speech

def main():
    print("Smart Voice Assistant Started")
    while True:
        print("Listening for wake word...")
        if wake_word.detect_wake_word():
            text_to_speech.speak("I am listening. Please ask your question.")
            
            audio_np, fs = speech_to_text.record_audio(duration=5)
            question = speech_to_text.transcribe(audio_np, fs)
            print(f"User asked: {question}")

            if question == "":
                text_to_speech.speak("Sorry, I didn't catch that. Please try again.")
                continue

            answer = text_generation.generate_answer(question)
            print(f"Assistant answer: {answer}")

            text_to_speech.speak(answer)

if __name__ == "__main__":
    main()