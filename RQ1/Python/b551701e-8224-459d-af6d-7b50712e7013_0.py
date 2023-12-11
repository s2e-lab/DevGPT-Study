import webvtt
import edge_tts
import os
import asyncio
import nest_asyncio

nest_asyncio.apply()


def read_subtitles_from_vtt_file(file_path: str) -> str:
    """从.vtt文件读取字幕内容"""
    if not os.path.exists(file_path):
        print(f"The file {file_path} does not exist.")
        return ""

    # 从.vtt文件中读取字幕
    captions = webvtt.read(file_path)

    # 将所有字幕组合成一个 SSML 字符串，每个字幕之间用空格分隔，并在每个字幕后添加一个小的停顿
    return ' '.join(f"{caption.text}<break strength='weak'/>" for caption in captions)


def convert_to_ssml(text: str) -> str:
    """将文本转换为 SSML"""
    return f"<speak>{text}</speak>"


async def text_to_speech(text_file: str, audio_file: str, voice: str = "zh-CN-XiaoxiaoNeural"):
    """Main function"""
    text = read_subtitles_from_vtt_file(text_file)
    if text:
        ssml = convert_to_ssml(text)
        communicate = edge_tts.Communicate(ssml, voice, rate='+75%')
        await communicate.save(audio_file)


def process_all_files(subtitle_dir: str, output_dir: str):
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all .vtt files in the subtitle directory
    for file_name in os.listdir(subtitle_dir):
        if file_name.endswith('.vtt'):
            # Remove the .vtt extension
            base_name = os.path.splitext(file_name)[0]
            text_file = os.path.join(subtitle_dir, file_name)
            audio_file = os.path.join(output_dir, base_name + '.mp3')
            asyncio.run(text_to_speech(text_file, audio_file))


# Specify the directories for the subtitles and the output audio files
SUBTITLE_DIR = '../youtube/subtitle'
OUTPUT_DIR = '../youtube/mp3'

# Process all files
process_all_files(SUBTITLE_DIR, OUTPUT_DIR)
