class LMLangClassifier:
    def __init__(self, path=None):
        """
        Initialize the language classifier.

        Parameters:
            path (str): Optional. Path to a file containing pre-trained language models.
        """
        ...

    def fit(self, save=True):
        """
        Train the language models.

        Parameters:
            save (bool): Optional. Whether to save the trained models to a file.

        Returns:
            str: Path to the file where the models are saved.
        """
        ...

    @staticmethod
    def calculate_cosine(a: typing.Dict[str, float], b: typing.Dict[str, float]) -> float:
        """
        Calculate the cosine similarity between two numeric vectors.

        Parameters:
            a (Dict[str, float]): First numeric vector represented as a dictionary.
            b (Dict[str, float]): Second numeric vector represented as a dictionary.

        Returns:
            float: Cosine similarity between the two vectors.
        """
        ...

    @staticmethod
    def extract_xgrams(text: str, n_vals: typing.List[int]) -> typing.List[str]:
        """
        Extract a list of n-grams from a text.

        Parameters:
            text (str): The text from which to extract n-grams.
            n_vals (List[int]): List of n-gram sizes to extract.

        Returns:
            List[str]: List of extracted n-grams.
        """
        ...

    @classmethod
    def build_model(cls, text: str, n_vals=range(1, 4)) -> typing.Dict[str, int]:
        """
        Build a language model from a text.

        Parameters:
            text (str): The text from which to build the language model.
            n_vals (range): Optional. Range of n-gram sizes to include in the model.

        Returns:
            Dict[str, int]: Language model containing n-grams and their probabilities.
        """
        ...

    def identify_language(self, text: str, n_vals=range(1, 4)) -> str:
        """
        Identify the language of a given text.

        Parameters:
            text (str): The text whose language to identify.
            n_vals (range): Optional. Range of n-gram sizes to use for language identification.

        Returns:
            str: Identified language.
        """
        ...

    def predict(self, text: str, n_vals=range(1, 4)) -> str:
        """
        Predict the language of a given text.

        Parameters:
            text (str): The text whose language to predict.
            n_vals (range): Optional. Range of n-gram sizes to use for prediction.

        Returns:
            Dict[str, float]: Dictionary of languages and their similarity scores to the input text.
        """
        ...
