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
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Apply nest_asyncio to allow asyncio to run inside Flask's event loop
nest_asyncio.apply()

# Import from the new, refactored structure
from config.settings import BASIR_JSON_PATH, SERPAPI_API_KEY, MAPS_API_KEY
from app.services.data_collector_service import GoogleMapsCollectorBasir
from app.repositories.data_repository import DataRepository
from app.services.llm import LLMClient
from core_logic.nlp_models.sentiment_analyzer import ReviewSentimentAnalyzer
from core_logic.nlp_models.preference_embedder import PreferenceEmbedder
from core_logic.scoring import rank_places
from core_logic.research_agents.restaurent_search_agent import RestaurantSearchAgent
from core_logic.utils.haversine import haversine

app = Flask(__name__)
CORS(app)

# Global instances of services and repositories
data_repo = DataRepository(raw_data_dir=os.path.dirname(BASIR_JSON_PATH))
google_maps_collector = GoogleMapsCollectorBasir(api_key=MAPS_API_KEY, output_dir=os.path.dirname(BASIR_JSON_PATH))
llm_client = LLMClient()
restaurant_agent = RestaurantSearchAgent(llm_client=llm_client, serpapi_key=SERPAPI_API_KEY)
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

def initialize_models():
    """Initializes the PreferenceEmbedder and ReviewSentimentAnalyzer."""
    global embedder, sentiment_analyzer
    logger.info("Initializing models...")
    try:
        embedder = PreferenceEmbedder()
        embedder.generate_attribute_embeddings(PREDEFINED_ATTRIBUTES)
        logger.info("PreferenceEmbedder initialized successfully.")

        sentiment_analyzer = ReviewSentimentAnalyzer() # The constructor handles model loading
        logger.info("ReviewSentimentAnalyzer initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing models: {e}", exc_info=True)
        # In a production app, you would handle this more gracefully,
        # perhaps by loading mock models or shutting down gracefully.

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
        if not user_city:
            logger.warning("City parameter is missing.")
            return jsonify({"success": False, "error": "City parameter is required."}), 400

        filtered_places = [p for p in places_data if user_city in p.get('address', '').strip().lower()]

        # Check if the existing dataset is sufficient for the city, if not, augment it
        if len(filtered_places) < 10:
            logger.info(f"Fewer than 10 places found in {user_city}. Augmenting data with search agent.")
            new_places_json = await restaurant_agent.execute(query=user_city, max_urls_to_process=10)
            new_place_names = json.loads(new_places_json).get("restaurants", [])
            
            if new_place_names:
                # Use the collector to get details for the newly found names
                # The collector should have a method to take a list of names directly or a temp file
                logger.info(f"Initiating Google Maps collection for {len(new_place_names)} new places.")
                temp_csv_path = "temp_new_restaurants.csv"
                pd.DataFrame({"Restaurant Name": new_place_names, "City": [user_city] * len(new_place_names)}).to_csv(temp_csv_path, index=False)
                new_places_df, new_reviews_df = await google_maps_collector.collect_data(input_csv_path=temp_csv_path, max_places=len(new_place_names))
                os.remove(temp_csv_path)

                # Add new data to our in-memory list
                for _, row in new_places_df.iterrows():
                    new_place = row.to_dict()
                    # Add reviews to the new place dictionary
                    new_place['reviews'] = new_reviews_df[new_reviews_df['place_name'] == new_place['name']].to_dict('records')
                    places_data.append(new_place)

                # Re-filter the places list to include the newly added ones
                filtered_places = [p for p in places_data if user_city in p.get('address', '').strip().lower()]
                logger.info(f"Dataset now has {len(filtered_places)} places for {user_city}.")
            else:
                logger.info("Search agent found no new places.")

        if not filtered_places:
            logger.warning(f"No places found for '{user_city}'.")
            return jsonify({"success": False, "error": "No places found for the specified city."}), 404
        
        # Determine user coordinates (logic remains similar)
        user_coords = None
        # ... (simplified for brevity, a real app would use a geocoder here)
        if filtered_places:
            user_coords = (filtered_places[0]['latitude'], filtered_places[0]['longitude'])

        user_data = {
            "preferences": data.get('preferences', {}),
            "budget": data.get('budget', 2),
            "coords": user_coords
        }

        # The core ranking logic is now a standalone function
        ranked_places = rank_places(
            user_data=user_data,
            places_list=filtered_places,
            embedder=embedder,
            sentiment_analyzer=sentiment_analyzer,
            predefined_attributes=PREDEFINED_ATTRIBUTES,
            max_distance_km=float(data.get('max_distance', 50))
        )
        
        recommendations = [
            {
                "name": p["name"],
                "address": p.get("address", "Unknown"),
                "budget": p.get("budget", p.get("avg_price_usd", 0)),
                "scoring_details": p.get("scoring_details", {})
            } for p in ranked_places
        ]
        
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