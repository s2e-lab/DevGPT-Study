from fastapi import FastAPI, StreamingResponse
import os

app = FastAPI()

WAV_DIR = "path_to_directory_containing_wav_files"  # Update this to your path

def stream_wav_files():
    files = [f for f in os.listdir(WAV_DIR) if f.endswith(".wav")]
    for file in files:
        file_path = os.path.join(WAV_DIR, file)
        with open(file_path, "rb") as f:
            while chunk := f.read(4096):  # reading 4KB at a time
                yield chunk
        yield b"\n\n\n"  # Separator between files, modify as needed

@app.get("/stream_wav")
def stream():
    return StreamingResponse(stream_wav_files(), media_type="audio/wav")
