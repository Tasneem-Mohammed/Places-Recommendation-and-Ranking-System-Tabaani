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

class RestaurantService:
    """Service class that wraps CSVDataService to provide the interface expected by the web server."""
    
    def __init__(self):
        from app.services.csv_data_service import CSVDataService
        self.csv_service = CSVDataService()
    
    def get_unique_countries(self) -> List[str]:
        """Get list of available countries."""
        return self.csv_service.get_unique_countries()
    
    def get_unique_cities(self, country: str = None) -> List[str]:
        """Get list of cities for a specific country."""
        return self.csv_service.get_unique_cities(country)
    
    def get_unique_suitabilities(self) -> List[str]:
        """Get list of available suitability options."""
        return self.csv_service.get_unique_suitabilities()
    
    def find_restaurants(self, country: str = None, city: str = None, suitability: str = None) -> List[Dict]:
        """Find restaurants based on criteria."""
        return self.csv_service.find_restaurants(country, city, suitability)
    
    def get_data_source_info(self) -> Dict:
        """Get information about the data source."""
        stats = self.csv_service.get_stats()
        return {
            "source": "CSV",
            "stats": stats
        }