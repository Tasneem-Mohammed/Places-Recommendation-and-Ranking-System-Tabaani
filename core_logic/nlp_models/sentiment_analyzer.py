from transformers import pipeline
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class LanguageDetector:
    """
    A helper class to detect the dominant language of a text.
    """
    def detect(self, text: str) -> str:
        """
        Detects if a text is predominantly Arabic or English by counting characters.

        Args:
            text (str): The input text to analyze.

        Returns:
            str: 'ar' for Arabic, 'en' for English, or 'unknown'.
        """
        if not isinstance(text, str):
            return 'unknown'

        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF')
        english_chars = sum(1 for char in text if 'a' <= char.lower() <= 'z')

        if arabic_chars > english_chars:
            return 'ar'
        elif english_chars > arabic_chars:
            return 'en'
        else:
            return 'unknown'

class ReviewSentimentAnalyzer:
    """
    Manages sentiment analysis models for English and Arabic to score reviews.
    """
    def __init__(
        self,
        en_model_path: str = './models/roberta',
        ar_model_path: str = './models/araberta'
    ):
        """
        Initializes and loads sentiment analysis models from specified paths.

        Args:
            en_model_path (str): The file path to the English sentiment model.
            ar_model_path (str): The file path to the Arabic sentiment model.
        """
        self.pipelines: Dict[str, Any] = {}
        self.language_detector = LanguageDetector()
        try:
            logger.info(f"Loading English sentiment model from: {en_model_path}...")
            self.pipelines['en'] = pipeline("sentiment-analysis", model=en_model_path)

            logger.info(f"Loading Arabic sentiment model from: {ar_model_path}...")
            self.pipelines['ar'] = pipeline("sentiment-analysis", model=ar_model_path)

            logger.info("All sentiment models loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading sentiment models: {e}")
            self.pipelines = {}

    def analyze_reviews(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyzes a list of review dictionaries and adds a sentiment score to each.

        The score is 1.0 for positive, 0.0 for negative, and 0.5 for neutral/unknown.

        Args:
            reviews (List[Dict[str, Any]]): A list of review dictionaries,
                each expected to have a 'text' key.

        Returns:
            List[Dict[str, Any]]: The same list of dictionaries, with a new
                'sentiment_score' key added to each one.
        """
        if not self.pipelines:
            logger.warning("Sentiment models not loaded. Scores will default to 0.5.")
            for review in reviews:
                review['sentiment_score'] = 0.5
            return reviews

        for review in reviews:
            text = review.get('text', '')

            if not text or not text.strip():
                review['sentiment_score'] = 0.5
                continue

            lang = self.language_detector.detect(text)
            if lang in self.pipelines:
                try:
                    result = self.pipelines[lang](text)[0]
                    # Map 'LABEL_1' (positive) to 1.0, 'LABEL_0' (negative) to 0.0
                    review['sentiment_score'] = 1.0 if result['label'] == 'LABEL_1' else 0.0
                except Exception as e:
                    logger.error(f"Failed to analyze text for language '{lang}': {e}")
                    review['sentiment_score'] = 0.5
            else:
                review['sentiment_score'] = 0.5

        return reviews