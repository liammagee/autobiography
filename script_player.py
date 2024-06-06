import os
import re
import openai
from openai import OpenAI, AsyncOpenAI
import requests
from dotenv import load_dotenv
from pydub import AudioSegment
import io


load_dotenv()

# Configure OpenAI API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# An API key is defined here. You'd normally get this from the service you're accessing. It's a form of authentication.

# Define constants for the script
CHUNK_SIZE = 1024  # Size of chunks to read/write at a time
XI_API_KEY = os.environ.get("XI_API_KEY")  # Your API key for authentication

# This is the URL for the API endpoint we'll be making a GET request to.
url = "https://api.elevenlabs.io/v1/voices"


previous_request_ids = []

# Here, headers for the HTTP request are being set up. 
# Headers provide metadata about the request. In this case, we're specifying the content type and including our API key for authentication.
headers = {
  "Accept": "application/json",
  "xi-api-key": XI_API_KEY,
  "Content-Type": "application/json"
}

# Read the Markdown file
def read_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def remove_markdown(text):
    # Remove bold and italics
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic*
    text = re.sub(r'__([^_]+)__', r'\1', text)      # __bold__
    text = re.sub(r'_([^_]+)_', r'\1', text)        # _italic_
    return text

# Parse dialogue lines
def parse_dialogue(content):
    dialogue_lines = re.findall(r'(\w+): (.*?)(?=\n\w+:|\Z)', content, re.DOTALL)
    # Remove Markdown syntax from dialogue text
    cleaned_dialogue_lines = [(speaker, remove_markdown(text)) for speaker, text in dialogue_lines]
    return cleaned_dialogue_lines

# Convert text to speech using OpenAI's TTS
def text_to_speech_openai(text, speaker, speed, filename):

    # Define a dictionary for each speaker's voice
    speakers_voices = {
        'system': 'onyx',
        'Sasha': 'nova',
        'Zhang': 'echo',
        'Ashley': 'fable'
    }

    voice = speakers_voices.get(speaker, 'default_voice')

    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text, 
        speed=speed,
        response_format="mp3"
    )
    response.stream_to_file(filename)


def text_to_speech_elevenlabs(text, index, all_texts, speaker, speed, filename):
    global previous_request_ids
    
    is_last_paragraph = index == len(all_texts) - 1
    is_first_paragraph = index == 0
    
    # Define a dictionary for each speaker's voice
    speakers_voices = {
        'system': 'Hxit40sC3cJk9gObCO0u',
        'Sasha': 'LkletHyEKUtF1g5Ggh77',
        'Zhang': 'NMbn4FNN0acONjKLsueJ',
        'Ashley': 'YvR5CCFzZ0XzyFkKhccl'
    }

    voice_id = speakers_voices.get(speaker, 'default_voice')
    print(f"{speaker}: {voice_id}")

    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream",
        json={
            "text": text,
            "model_id": "eleven_turbo_v2",
            "output_format":"mp3_22050_32",
            "previous_request_ids": previous_request_ids[-3:],
            "previous_text": None if is_first_paragraph else " ".join(all_texts[:index]),
            "next_text": None if is_last_paragraph else " ".join(all_texts[index + 1:])
        },
        headers={"xi-api-key": XI_API_KEY},
    )
    
    if response.status_code != 200:
        print(f"Error encountered, status: {response.status_code}, "
               f"content: {response.text}")
        quit()

    print(f"Successfully converted paragraph {index + 1}/{len(all_texts)}")
    req_id = response.headers["request-id"]
    previous_request_ids.append(req_id)
    audio_segment = AudioSegment.from_mp3(io.BytesIO(response.content))
    audio_segment.export(f"./audio/{req_id}_{index}.mp3", format="wav")

    return audio_segment


# Main script
def main():
    file_path = 'scripts/dialogue_20240606105552.md'
    output_dir = './audio'
    speed = 1.25
    os.makedirs(output_dir, exist_ok=True)

    dialogue_content = read_markdown_file(file_path)
    dialogue_lines = parse_dialogue(dialogue_content)
    
    # Extract only the values (dialogue text)
    dialogue_texts = [line[1] for line in dialogue_lines]

    for i, (speaker, line) in enumerate(dialogue_lines):
        print(speaker)
    # quit()
    audio_segments = []

    running_text = ''
    for i, (speaker, line) in enumerate(dialogue_lines):
        filename = os.path.join(output_dir, f'{i+1}_{speaker}_{speed}.mp3')
        # Check if the individual audio file already exists
        if os.path.exists(filename):
            print(f"Audio file already exists: {filename}")
            audio_segment = AudioSegment.from_mp3(filename)
            audio_segments.append(audio_segment)
        else:
            # text_to_speech_openai(line, speaker, speed, filename)
            # audio_segment = AudioSegment.from_mp3(filename)
            audio_segment = text_to_speech_elevenlabs(line, i, dialogue_texts, speaker, speed, filename)
            audio_segments.append(audio_segment)
            print(f'Saved: {filename}')
        
        running_text += line + ' '

        # Load the saved audio file into pydub AudioSegment
        # audio_segment = AudioSegment.from_mp3(filename)
        #audio_segments.append(audio_segment)
    

    # segment = audio_segments[0]
    # for new_segment in audio_segments[1:]:
    #     segment = segment + new_segment
            
    # Combine all audio segments
    combined_audio = sum(audio_segments)

    combined_output_file = os.path.join(output_dir, "combined_output_xi_{speed}.mp3")
            
    # Export the combined audio to a single MP3 file
    combined_audio.export(combined_output_file, format='mp3')
    print(f'Combined audio saved to: {combined_output_file}')


if __name__ == '__main__':
    main()