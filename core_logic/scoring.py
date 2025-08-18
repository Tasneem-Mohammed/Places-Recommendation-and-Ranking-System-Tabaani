import json
from collections import defaultdict
from typing import List, Dict, Any

from core_logic.nlp_models.preference_embedder import PreferenceEmbedder

def calculate_place_attribute_profile(
    place_reviews: List[Dict[str, Any]],
    preference_embedder_instance: PreferenceEmbedder,
    predefined_attributes: List[str]
) -> Dict[str, float]:
    """
    Aggregates attribute scores for all reviews of a given place to create
    an overall attribute profile.
    """
    if not place_reviews:
        print("No reviews provided for this place. Returning zero attribute profile.")
        return {attr: 0.0 for attr in predefined_attributes}

    attribute_sums = defaultdict(float)
    review_count_with_content = 0

    for review in place_reviews:
        review_text = review.get('text', '')
        if review_text and review_text.strip():
            scores = preference_embedder_instance.get_review_attribute_scores(review_text)
            for attr, score in scores.items():
                attribute_sums[attr] += score
            review_count_with_content += 1

    place_attribute_profile = {}
    if review_count_with_content > 0:
        for attr in predefined_attributes:
            place_attribute_profile[attr] = attribute_sums[attr] / review_count_with_content
    else:
        place_attribute_profile = {attr: 0.0 for attr in predefined_attributes}

    return place_attribute_profile


def calculate_preference_score(
    user_preferences: Dict[str, float],
    place_attribute_profile: Dict[str, float],
    predefined_attributes: List[str]
) -> float:
    """
    Calculates a single preference score for a place based on user's weighted
    preferences and the place's attribute profile.
    """
    total_score = 0.0
    total_user_weight = sum(user_preferences.values())

    normalized_user_preferences = {}
    if total_user_weight > 0:
        for attr, weight in user_preferences.items():
            normalized_user_preferences[attr] = weight / total_user_weight
    else:
        print("Warning: No user preferences provided for preference score calculation. Returning 0.")
        return 0.0

    for attr in predefined_attributes:
        user_weight = normalized_user_preferences.get(attr, 0.0)
        place_attr_score = place_attribute_profile.get(attr, 0.0)
        total_score += user_weight * place_attr_score

    return total_score