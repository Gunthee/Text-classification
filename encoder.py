from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class SentenceTransformerEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        # รองรับทั้ง list / Series / ndarray
        if not isinstance(X, list):
            X = list(X)

        return self.model.encode(
            X,
            batch_size=32,
            show_progress_bar=False
        )