import whisper

model = whisper.load_model("turbo")
result = model.transcribe("song2.mp3")
print(result["text"])