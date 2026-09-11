#!/usr/bin/env python3
"""
Mail.ru Course Transcriber
Transcribes video lectures using OpenAI's Whisper API
"""

import os
import sys
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
COURSE_LINKS = {
    1: 'links/course1.txt',
    2: 'links/course2.txt'
}

def get_video_links(course_num):
    """Load video links from course file"""
    filepath = COURSE_LINKS.get(course_num)
    if not filepath or not Path(filepath).exists():
        print(f"Course {course_num} not found")
        return []
    
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def download_video(url, output_path):
    """Download video from URL"""
    print(f"Downloading: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def extract_audio(video_path, audio_path):
    """Extract audio from video file"""
    print(f"Extracting audio from: {video_path}")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)
    video.close()

def transcribe_audio(audio_path):
    """Transcribe audio using OpenAI Whisper API"""
    print(f"Transcribing: {audio_path}")
    
    with open(audio_path, 'rb') as audio_file:
        response = requests.post(
            'https://api.openai.com/v1/audio/transcriptions',
            headers={'Authorization': f'Bearer {OPENAI_API_KEY}'},
            files={'file': audio_file},
            data={'model': 'whisper-1'}
        )
    
    response.raise_for_status()
    return response.json()['text']

def process_course(course_num):
    """Process entire course"""
    links = get_video_links(course_num)
    output_dir = Path(f'transcripts_course{course_num}')
    output_dir.mkdir(exist_ok=True)
    
    for idx, link in enumerate(links, 1):
        try:
            video_path = output_dir / f'video_{idx}.mp4'
            audio_path = output_dir / f'audio_{idx}.mp3'
            
            # Download video
            download_video(link, video_path)
            
            # Extract audio
            extract_audio(str(video_path), str(audio_path))
            
            # Transcribe
            transcript = transcribe_audio(str(audio_path))
            
            # Save transcript
            transcript_path = output_dir / f'transcript_{idx}.txt'
            with open(transcript_path, 'w') as f:
                f.write(transcript)
            
            print(f"✓ Completed video {idx}")
            
            # Cleanup
            video_path.unlink()
            audio_path.unlink()
            
        except Exception as e:
            print(f"✗ Error processing video {idx}: {e}")
            continue

if __name__ == '__main__':
    course_num = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"Processing course {course_num}...")
    process_course(course_num)
    print("Done!")
