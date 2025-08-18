import math
from typing import List, Dict, Any, Tuple

def calculate_proximity_score(
    user_coords: Tuple[float, float], 
    place_coords: Tuple[float, float],
    max_distance_km: int = 10
) -> float:
    """Calculates a normalized proximity score (0-1) based on geographic distance."""
    R = 6371
    lat1, lon1 = math.radians(user_coords[0]), math.radians(user_coords[1])
    lat2, lon2 = math.radians(place_coords[0]), math.radians(place_coords[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    if distance > max_distance_km:
        return 0.0
    
    return 1.0 - (distance / max_distance_km)


def calculate_budget_score(user_budget: int, place_budget: int) -> float:
    """Calculates a normalized budget match score (0-1)."""
    max_budget_diff = 3.0
    diff = abs(user_budget - place_budget)
    
    return 1.0 - (diff / max_budget_diff)


def calculate_aggregated_sentiment(reviews_with_sentiment: List[Dict[str, Any]]) -> float:
    """Calculates the average sentiment score from a list of analyzed reviews."""
    if not reviews_with_sentiment:
        return 0.5
    
    total_score = sum(review.get('sentiment_score', 0.5) for review in reviews_with_sentiment)
    return total_score / len(reviews_with_sentiment)