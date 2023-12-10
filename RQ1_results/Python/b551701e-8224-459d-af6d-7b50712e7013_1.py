async def text_to_speech(text_file: str, audio_file: str, voice: str = "zh-CN-XiaoxiaoNeural"):
    """Main function"""
    text = read_subtitles_from_vtt_file(text_file)
    if text:
        ssml = convert_to_ssml(text)
        communicate = edge_tts.Communicate(ssml, voice, rate='+75%')
        await communicate.save(audio_file)
