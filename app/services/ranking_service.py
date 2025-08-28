from typing import List, Dict, Any, Tuple
import logging

from core_logic.nlp_models.sentiment_analyzer import ReviewSentimentAnalyzer
from core_logic.nlp_models.preference_embedder import PreferenceEmbedder
from core_logic.scoring import calculate_place_attribute_profile, calculate_preference_score
from app.utils.scoring_utils import calculate_proximity_score, calculate_budget_score, calculate_aggregated_sentiment

# Set up logging for the ranking service
ranking_service_logger = logging.getLogger("app.services.ranking_service")
ranking_service_logger.setLevel(logging.INFO)

class RankingService:
    """
    A service to handle the core logic of ranking places based on a user's preferences.
    It consolidates various scoring metrics into a final, weighted score.
    """

    def __init__(self, 
                 embedder: PreferenceEmbedder, 
                 sentiment_analyzer: ReviewSentimentAnalyzer,
                 predefined_attributes: List[str]):
        """
        Initializes the RankingService with necessary models and data.
        
        Args:
            embedder: An instance of PreferenceEmbedder for preference matching.
            sentiment_analyzer: An instance of ReviewSentimentAnalyzer for sentiment analysis.
            predefined_attributes: A list of attributes to consider for scoring.
        """
        self.embedder = embedder
        self.sentiment_analyzer = sentiment_analyzer
        self.predefined_attributes = predefined_attributes
        ranking_service_logger.info("RankingService initialized with NLP models.")

    def _calculate_final_score(
        self,
        place_data: Dict[str, Any],
        user_data: Dict[str, Any],
        max_distance_km: float = 10
    ) -> Dict[str, float]:
        """
        Calculates the final weighted score for a single place based on multiple factors.
        This is a private helper method used by rank_places.
        """
        ranking_service_logger.debug(f"Calculating final score for place: {place_data.get('name', 'N/A')}")
        
        # 1. Calculate Sentiment Score (Weight: 0.25)
        # Assuming place_data['reviews'] is a list of review texts.
        reviews_with_sentiment = self.sentiment_analyzer.analyze_reviews(place_data.get('reviews', []))
        sentiment_score = calculate_aggregated_sentiment(reviews_with_sentiment)
    
        # 2. Calculate Preference Matching Score (Weight: 0.30)
        place_profile = calculate_place_attribute_profile(
            place_data.get('reviews', []), self.embedder, self.predefined_attributes
        )
        preference_score = calculate_preference_score(
            user_data.get('preferences', ""), place_profile, self.predefined_attributes
        )
        
        # 3. Calculate Proximity Score (Weight: 0.30)
        proximity_score = calculate_proximity_score(
            user_data.get('coords', {}), place_data.get('coords', {}), max_distance_km
        )
    
        # 4. Calculate Budget Matching Score (Weight: 0.15)
        budget_score = calculate_budget_score(
            user_data.get('budget', ""), place_data.get('budget_range', "")
        )
    
        # final weighted formula
        final_score = (
            (sentiment_score * 0.25) +
            (preference_score * 0.30) +
            (proximity_score * 0.30) +
            (budget_score * 0.15)
        )
        
        # Return detailed breakdown of scores
        return {
            "final_score": round(final_score, 4),
            "sentiment_score": round(sentiment_score, 4),
            "preference_score": round(preference_score, 4),
            "proximity_score": round(proximity_score, 4),
            "budget_score": round(budget_score, 4)
        }

    def rank_restaurants(self, restaurants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        A simple pass-through method for initial testing.
        This can be replaced with the more sophisticated ranking logic later.
        """
        ranking_service_logger.warning("Using simple rating-based ranking. Consider using `rank_places` for advanced logic.")
        return sorted(restaurants, key=lambda x: x.get("rating", 0), reverse=True)

    def rank_places(
        self,
        user_data: Dict[str, Any],
        places_list: List[Dict[str, Any]],
        max_distance_km: float = 10
    ) -> List[Dict[str, Any]]:
        """
        Calculates scores for multiple places and returns them sorted by final score.
        """
        scored_places = []
        ranking_service_logger.info(f"Starting to score {len(places_list)} places.")
    
        for place in places_list:
            try:
                score_details = self._calculate_final_score(
                    place_data=place,
                    user_data=user_data,
                    max_distance_km=max_distance_km
                )
                
                place['scoring_details'] = score_details
                scored_places.append(place)
            except Exception as e:
                ranking_service_logger.error(f"Error scoring place {place.get('name', 'N/A')}: {e}")
                continue # Skip to the next place if an error occurs
    
        # Sort the places by their final score in descending order
        sorted_places = sorted(
            scored_places,
            key=lambda p: p.get('scoring_details', {}).get('final_score', -1),
            reverse=True
        )
        
        ranking_service_logger.info("Ranking complete.")
        return sorted_places
