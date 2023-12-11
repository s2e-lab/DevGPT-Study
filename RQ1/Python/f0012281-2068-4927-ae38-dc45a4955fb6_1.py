from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# モデルとトークナイザーの準備
model_name = "satellite-instrument-roberta-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# テスト用のテキスト
text = "衛星の軌道上での動作テストが成功しました。観測データの収集も順調です。"

# テキストのトークン化
tokens = tokenizer.encode(text, add_special_tokens=True)

# モデルにトークンを入力して結果を取得
inputs = {
    "input_ids": torch.tensor(tokens).unsqueeze(0),
    "attention_mask": torch.tensor([1] * len(tokens)).unsqueeze(0),
}
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1)[0]

# 結果の表示
print("テキスト:", text)
print("予測結果:", predictions)
