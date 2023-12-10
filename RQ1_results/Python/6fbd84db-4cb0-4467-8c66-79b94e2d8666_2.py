import csv

# Load humans' and GPT's answers without the code part
def load_text_without_code(csv_filename):
    with open(csv_filename, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        text = ""
        for row in reader:
            _, _, entry_text, code = row
            text += entry_text.replace(code, "") + " "
        return text.strip()

# Example usage
csv_filename = "chat_data.csv"
text_without_code = load_text_without_code(csv_filename)

text = f"""
{text_without_code}
"""
