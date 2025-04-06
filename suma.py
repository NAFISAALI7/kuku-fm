# Install dependencies (run in terminal if needed)
# pip install moviepy openai-whisper transformers torch

import whisper
from moviepy.editor import VideoFileClip
from transformers import pipeline

# Step 1: Extract audio from video
def extract_audio(video_path, audio_output="audio.wav"):
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_output)
    return audio_output

# Step 2: Transcribe audio using Whisper
def transcribe_audio(audio_path):
    model = whisper.load_model("base")  # Or use "medium"/"large" for higher accuracy
    result = model.transcribe(audio_path)
    return result["text"]

# Step 3: Summarize transcript
def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summary = ""
    for chunk in chunks:
        summary += summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
    return summary

# Example usage
video_file = "your_video.mp4"
audio = extract_audio(video_file)
transcript = transcribe_audio(audio)
summary = summarize_text(transcript)

print("SUMMARY:\n", summary)
