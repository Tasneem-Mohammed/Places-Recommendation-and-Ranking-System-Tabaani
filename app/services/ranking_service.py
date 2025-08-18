from typing import List, Dict, Any, Tuple

from core_logic.nlp_models.sentiment_analyzer import ReviewSentimentAnalyzer # Assuming this exists
from core_logic.nlp_models.preference_embedder import PreferenceEmbedder
from core_logic.scoring import calculate_place_attribute_profile, calculate_preference_score
from app.utils.scoring_utils import calculate_proximity_score, calculate_budget_score, calculate_aggregated_sentiment

def calculate_final_score(
    place_data: Dict[str, Any],
    user_data: Dict[str, Any],
    embedder: PreferenceEmbedder,
    sentiment_analyzer: ReviewSentimentAnalyzer,
    predefined_attributes: List[str],
    max_distance_km: float = 10
) -> Dict[str, float]:
    """Calculates the final weighted score for a place based on multiple factors."""
    # 1. Calculate Sentiment Score (Weight: 0.25)
    reviews_with_sentiment = sentiment_analyzer.analyze_reviews(place_data['reviews'])
    sentiment_score = calculate_aggregated_sentiment(reviews_with_sentiment)

    # 2. Calculate Preference Matching Score (Weight: 0.30)
    place_profile = calculate_place_attribute_profile(
        place_data['reviews'], embedder, predefined_attributes
    )
    preference_score = calculate_preference_score(
        user_data['preferences'], place_profile, predefined_attributes
    )
    
    # 3. Calculate Proximity Score (Weight: 0.30)
    proximity_score = calculate_proximity_score(
        user_data['coords'], place_data['coords'], max_distance_km
    )

    # 4. Calculate Budget Matching Score (Weight: 0.15)
    budget_score = calculate_budget_score(
        user_data['budget'], place_data['budget']
    )

    # final weighted formula
    final_score = (
        (sentiment_score * 0.25) +
        (preference_score * 0.30) +
        (proximity_score * 0.30) +
        (budget_score * 0.15)
    )

    return {
        "final_score": round(final_score, 4),
        "sentiment_score": round(sentiment_score, 4),
        "preference_score": round(preference_score, 4),
        "proximity_score": round(proximity_score, 4),
        "budget_score": round(budget_score, 4)
    }
    
def rank_places(
    user_data: Dict[str, Any],
    places_list: List[Dict[str, Any]],
    embedder: PreferenceEmbedder,
    sentiment_analyzer: ReviewSentimentAnalyzer,
    predefined_attributes: List[str],
    max_distance_km: float = 10
) -> List[Dict[str, Any]]:
    """Calculates scores for multiple places and returns them sorted by final score."""
    scored_places = []
    print(f"\nScoring {len(places_list)} places...")

    for place in places_list:
        score_details = calculate_final_score(
            place_data=place,
            user_data=user_data,
            embedder=embedder,
            sentiment_analyzer=sentiment_analyzer,
            predefined_attributes=predefined_attributes,
            max_distance_km=max_distance_km
        )
        
        place['scoring_details'] = score_details
        scored_places.append(place)

    sorted_places = sorted(
        scored_places,
        key=lambda p: p['scoring_details']['final_score'],
        reverse=True
    )
    
    print("Ranking complete.")
    return sorted_places