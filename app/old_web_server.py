# app\web_server.py
import os
import pandas as pd
import json
import sys
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import asyncio
import nest_asyncio
import logging

# Set up logging for the application
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
logger = logging.getLogger(__name__)

# Ensure the project root is in the system path for correct imports
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# Apply nest_asyncio to allow asyncio to run inside Flask's event loop
nest_asyncio.apply()

# --- START OF FIX ---
# Get the absolute path of the current script's directory (app)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current directory (project root)
project_root = os.path.dirname(current_dir)
# Add the project root to the system path
sys.path.insert(0, project_root)

# Import from the new, refactored structure
from config.settings import BASIR_JSON_PATH, SERPAPI_API_KEY, MAPS_API_KEY
from app.services.data_collector_service import GoogleMapsCollectorBasir
from app.repositories.data_repository import DataRepository
from app.services.llm import LLMClient
from core_logic.nlp_models.sentiment_analyzer import ReviewSentimentAnalyzer
from core_logic.nlp_models.preference_embedder import PreferenceEmbedder
from app.services.ranking_service import RankingService
from core_logic.research_agents.restaurent_search_agent import RestaurantSearchAgent
from core_logic.utils.haversine import haversine

app = Flask(__name__)
CORS(app)

# Global instances of services and repositories
data_repo = DataRepository(raw_data_dir=os.path.dirname(BASIR_JSON_PATH))

google_maps_collector = None
if MAPS_API_KEY:
    try:
        google_maps_collector = GoogleMapsCollectorBasir(api_key=MAPS_API_KEY, output_dir=os.path.dirname(BASIR_JSON_PATH))
    except Exception as e:
        logger.warning(f"Failed to initialize GoogleMapsCollectorBasir: {e}")

llm_client = LLMClient()
restaurant_agent = None
if SERPAPI_API_KEY:
    try:
        restaurant_agent = RestaurantSearchAgent(llm_client=llm_client, serpapi_key=SERPAPI_API_KEY)
    except Exception as e:
        logger.warning(f"Failed to initialize RestaurantSearchAgent: {e}")

embedder = None
sentiment_analyzer = None

# Predefined attributes for the PreferenceEmbedder
PREDEFINED_ATTRIBUTES = [
    "Cozy", "Trendy", "Romantic", "Lively", "Quiet", "Elegant", "Casual", "Artistic",
    "Bohemian", "Family-Friendly", "Pet-Friendly", "Outdoor Seating", "Good for Groups",
    "Good for Solo", "Gourmet", "Comfort Food", "Healthy", "Vegan-Friendly", "Dessert",
    "Coffee", "Date", "Scenic View", "Parking Available", "Wheelchair Accessible",
    "Wi-Fi Available", "Workspace"
]
class MockSentimentAnalyzer:
    """Mock sentiment analyzer for demo purposes."""
    def analyze_reviews(self, reviews):
        app.logger.debug("MockSentimentAnalyzer: Analyzing reviews.")
        import random
        for review in reviews:
            review['sentiment_score'] = random.uniform(0.4, 0.9)
        return reviews

class MockPreferenceEmbedder:
    """Mock preference embedder for demo purposes."""
    def _init_(self):
        self.attribute_names = PREDEFINED_ATTRIBUTES
        app.logger.debug("MockPreferenceEmbedder: Initialized.")

    def generate_attribute_embeddings(self, attributes):
        self.attribute_names = attributes
        app.logger.debug("MockPreferenceEmbedder: Generated attribute embeddings.")

    def get_review_attribute_scores(self, review_text):
        app.logger.debug("MockPreferenceEmbedder: Getting attribute scores for a review.")
        import random
        scores = {}
        review_lower = review_text.lower()
        for attr in self.attribute_names:
            attr_lower = attr.lower()
            if attr_lower in review_lower or attr_lower.replace('-', ' ') in review_lower:
                scores[attr] = random.uniform(0.6, 0.9)
            else:
                scores[attr] = random.uniform(0.0, 0.3)
        return scores
def initialize_models():
    """
    Initializes the PreferenceEmbedder and SentimentAnalyzer models.
    """
    global embedder, sentiment_analyzer
    app.logger.info("Initializing models...")

    try:
        embedder = PreferenceEmbedder(model_name='jinaai/jina-embeddings-v3')
        embedder.generate_attribute_embeddings(PREDEFINED_ATTRIBUTES)
        app.logger.info("PreferenceEmbedder initialized successfully.")

        try:
            sentiment_analyzer = ReviewSentimentAnalyzer(
                en_model_path='hayn404/roberta-finetuned',
                ar_model_path='hayn404/araberta_finetuned'
            )
            app.logger.info("SentimentAnalyzer initialized successfully.")
        except Exception as e:
            app.logger.warning(f"Could not load sentiment models: {e}")
            app.logger.warning("Using mock sentiment analyzer for demo.")
            sentiment_analyzer = MockSentimentAnalyzer()

    except Exception as e:
        app.logger.error(f"Error initializing models: {e}", exc_info=True)
        app.logger.warning("Using mock models for demo.")
        embedder = MockPreferenceEmbedder()
        sentiment_analyzer = MockSentimentAnalyzer()

    app.logger.info("All models initialized.")

# Load initial places data
places_data = data_repo.load_from_json(BASIR_JSON_PATH)

@app.route('/')
def index():
    """Serves the main UI page."""
    logger.info("Serving index.html.")
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
async def get_recommendations():
    """
    API endpoint for getting ranked place recommendations.
    This is now an async function to support `await` calls.
    """
    logger.info("Received a recommendation request.")
    try:
        data = request.get_json()
        if not data:
            logger.error("Request body is empty or not valid JSON.")
            return jsonify({"success": False, "error": "Invalid JSON payload."}), 400

        user_city = data.get('city', '').strip().lower()
        user_latitude = data.get('latitude')
        user_longitude = data.get('longitude')
        max_distance_km = float(data.get('max_distance', 50))

        if not user_city and (user_latitude is None or user_longitude is None):
            logger.warning("City or user coordinates are required.")
            return jsonify({"success": False, "error": "City or user coordinates are required."}), 400

        

        # Determine the initial set of places to filter
        if user_city:
            filtered_places = [p for p in places_data if user_city in p.get('address', '').strip().lower()]
        else:
            filtered_places = places_data
        
        # Check if the existing dataset is sufficient for the city, if not, augment it
#        if len(filtered_places) < 10 and restaurant_agent and google_maps_collector:
        if len(filtered_places) < 10 and restaurant_agent:

            logger.info(f"Fewer than 10 places found in {user_city}. Augmenting data with search agent.")
            new_places_json = await restaurant_agent.execute(query=user_city, max_urls_to_process=50)
            new_place_names = json.loads(new_places_json).get("restaurants", [])
            print(f"New place names found: {new_places_json}")
            if new_place_names:
                logger.info(f"Initiating Google Maps collection for {len(new_place_names)} new places.")
                temp_csv_path = "temp_new_restaurants.csv"
                pd.DataFrame({"Restaurant Name": new_place_names, "City": [user_city] * len(new_place_names)}).to_csv(temp_csv_path, index=False)
                new_places_df, new_reviews_df = await google_maps_collector.collect_data(input_csv_path=temp_csv_path, max_places=len(new_place_names))
                os.remove(temp_csv_path)

                for _, row in new_places_df.iterrows():
                    new_place = row.to_dict()
                    new_place['reviews'] = new_reviews_df[new_reviews_df['place_name'] == new_place['name']].to_dict('records')
                    places_data.append(new_place)

                filtered_places = [p for p in places_data if user_city in p.get('address', '').strip().lower()]
                logger.info(f"Dataset now has {len(filtered_places)} places for {user_city}.")
            else:
                logger.info("Search agent found no new places.")
        elif len(filtered_places) < 10:
            logger.info(f"Fewer than 10 places found in {user_city}, but augmentation agents are not available.")

        if not filtered_places:
            logger.warning(f"No places found for '{user_city}'.")
            return jsonify({"success": False, "error": "No places found for the specified city."}), 404
        
        # UPDATED LOGIC: Pre-process the list to ensure each place has both a 'coords' and 'budget' key.
        processed_places = []
        for place in filtered_places:
            lat = place.get('latitude')
            lon = place.get('longitude')
            
            # Check for coordinates and create the 'coords' key
            if lat is not None and lon is not None:
                place['coords'] = (lat, lon)
            else:
                # If no coords, skip the place as it can't be ranked by proximity
                continue 
            
            # Check for 'budget' and provide a default if it's missing
            if 'budget' not in place or place['budget'] is None:
                # Use a default budget value to prevent KeyError
                place['budget'] = 0 
                
            processed_places.append(place)

        if not processed_places:
            logger.warning("No places found with both valid coordinates and budget data.")
            return jsonify({"success": False, "error": "No places found with valid data."}), 404
            
        # Determine user coordinates based on the request
        user_coords = (user_latitude, user_longitude) if user_latitude is not None and user_longitude is not None else None

        # Build the user data dictionary for the ranking service
        user_data = {
            "preferences": data.get('preferences', {}),
            "budget": data.get('budget', 2),
            "coords": user_coords
        }
        print("processed_places:", processed_places)
        # The core ranking logic is now a standalone function
        ranked_places = rank_places(
            user_data=user_data,
            places_list=processed_places,  # Pass the new, pre-processed list
            embedder=embedder,
            sentiment_analyzer=sentiment_analyzer,
            predefined_attributes=PREDEFINED_ATTRIBUTES,
            max_distance_km=max_distance_km
        )
        
        recommendations = [
            {
                "name": p["name"],
                "address": p.get("address", "Unknown"),
                "budget": p.get("budget", p.get("avg_price_usd", 0)),
                "scoring_details": p.get("scoring_details", {}),
                "distance": haversine(user_coords[0], user_coords[1], p['coords'][0], p['coords'][1]) if user_coords and p.get('coords') else None
            } for p in ranked_places if user_coords is None or (p.get('coords') and haversine(user_coords[0], user_coords[1], p['coords'][0], p['coords'][1]) <= max_distance_km)
        ]
        
        if user_coords:
            recommendations.sort(key=lambda r: r['distance'])

        return jsonify({"success": True, "recommendations": recommendations})

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

    
@app.route('/api/attributes')
def get_attributes():
    """API endpoint to get predefined attributes."""
    logger.info("Serving predefined attributes.")
    return jsonify({"attributes": PREDEFINED_ATTRIBUTES})

if __name__ == '__main__':
    initialize_models()
    app.run(debug=True, port=5000)