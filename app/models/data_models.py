from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

@dataclass
class Place:
    # Basir dataset fields:
    name: str
    country: str 
    city: str 
    cuisine: str
    recommended_dish: str
    avg_price_usd: float
    budget_range: str
    suitability: str
    
    # Google Maps API fields
    address: str
    rating: float
    num_reviews: int
    price_level: str
    latitude: float
    longitude: float
    place_id: str
    link: str

@dataclass
class Review:
    place_name: str
    review_index: int
    author: str
    text: str
    rating: float
    language: str