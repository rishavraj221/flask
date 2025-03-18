from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
from pydub import AudioSegment
import io
import base64

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load the Whisper model
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


@app.route("/")
def index():
    return "WebSocket server is running."


def webm_to_pcm(data):
    """
    Convert webm audio data to raw PCM audio (float32).
    """
    # Convert webm (or other formats) to wav using pydub
    audio = AudioSegment.from_file(io.BytesIO(data), format="webm")
    audio = audio.set_frame_rate(16000).set_channels(1)

    # Convert to numpy array (16-bit PCM format)
    samples = np.array(audio.get_array_of_samples(),
                       dtype=np.float32) / 32768.0  # Normalize to -1.0 to 1.0
    return samples


@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    """
    Handles incoming audio chunks from the client.
    """
    try:
        # Convert the incoming webm data to raw PCM

        # audio_data = data.split(',')[1]
        audio_bytes = base64.b64decode(data)
        with open("test.webm", 'ab') as f:
            f.write(audio_bytes)

        # pcm_audio = webm_to_pcm(data)
        # print(f"pcm audio {pcm_audio}")

        # # Process with Whisper
        # input_features = processor(
        #     pcm_audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

        # # Generate transcription
        # predicted_ids = model.generate(input_features)
        # transcription = processor.batch_decode(
        #     predicted_ids, skip_special_tokens=True)[0]

        # # Send transcription back to the client
        # emit("transcription", transcription)
        emit("transcription", "inspecting the issue ...")
    except Exception as e:
        emit("error", {"message": str(e)})


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5466)
