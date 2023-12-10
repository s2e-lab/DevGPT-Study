from pydub import AudioSegment

# Load the first WAV file
combined = AudioSegment.from_wav("file1.wav")

# Concatenate other WAV files
other_files = ["file2.wav", "file3.wav"]  # Add as many as needed
for file in other_files:
    sound = AudioSegment.from_wav(file)
    combined += sound

# Export the combined audio
combined.export("combined.wav", format="wav")
