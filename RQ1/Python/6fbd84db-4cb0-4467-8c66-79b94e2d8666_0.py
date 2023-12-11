class prompt_work():
    def __init__(self, output_filename) -> None:
        self.filename = output_filename
        self.max_tokens = 16000
        self.split_size = 1
        self.prompt = ""
        self.resp_text = ""

    def run_all(self):
        # ... your existing code ...
        self.resp_text = "your result"  # Set the value of resp_text here

    def get_resp_text(self):
        return self.resp_text
