def parse_large_json(large_json):
    smaller_json = {
        "order": "v",
        "md5sum": "",
        "name": "",
        "filename": "",
        "filesize": "",
        "requires": "",
        "ramrequired": "",
        "parameters": "",
        "quant": "",
        "type": "",
        "description": "",
        "url": "",
        "promptTemplate": "",
        "systemPrompt": ""
    }

    smaller_json["order"] = "v"
    smaller_json["md5sum"] = large_json["files"]["all"][0]["sha256checksum"]
    smaller_json["name"] = large_json["name"]
    smaller_json["filename"] = large_json["files"]["highlighted"]["economical"]["name"]
    smaller_json["filesize"] = str(large_json["files"]["all"][0]["sizeBytes"])
    smaller_json["requires"] = "2.4.14"
    smaller_json["ramrequired"] = "8"
    smaller_json["parameters"] = large_json["numParameters"]
    smaller_json["quant"] = large_json["files"]["all"][0]["quantization"]
    smaller_json["type"] = "LLaMA2"
    smaller_json["description"] = large_json["description"]
    smaller_json["url"] = large_json["files"]["all"][0]["url"]
    smaller_json["promptTemplate"] = "[INST] %1 [/INST]"
    smaller_json["systemPrompt"] = "[INST]<<SYS>>You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.<</SYS>>[/INST]"

    return smaller_json

# Example usage:
large_json = {
    # ... (the original large JSON content)
}

smaller_json = parse_large_json(large_json)
print(smaller_json)
