import os
import sys
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# --- IMPORTS ---
from app.models.data_models import RestaurantService
from app.services.llm import LLMClient
from app.services.ranking_service import RankingService
from core_logic.utils.haversine import haversine
from app.utils.scoring_utils import calculate_proximity_score
from core_logic.nlp_models.sentiment_analyzer import ReviewSentimentAnalyzer
from core_logic.nlp_models.preference_embedder import PreferenceEmbedder
from app.services.old_ranking_service import rank_places
from config.settings import SERPAPI_API_KEY, MAPS_API_KEY, BASIR_JSON_PATH
from app.repositories.data_repository import DataRepository
from app.services.data_collector_service import GoogleMapsCollectorBasir
from core_logic.research_agents.restaurent_search_agent import RestaurantSearchAgent
import asyncio
import json

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')

app = Flask(__name__, template_folder='templates')
CORS(app)

# Predefined attributes for the PreferenceEmbedder
# PREDEFINED_ATTRIBUTES = [
#     "Cozy", "Trendy", "Romantic", "Lively", "Quiet", "Elegant", "Casual", "Artistic",
#     "Bohemian", "Family-Friendly", "Pet-Friendly", "Outdoor Seating", "Good for Groups",
#     "Good for Solo", "Gourmet", "Comfort Food", "Healthy", "Vegan-Friendly", "Dessert",
#     "Coffee", "Date", "Scenic View", "Parking Available", "Wheelchair Accessible",
#     "Wi-Fi Available", "Workspace"
# ]
PREDEFINED_ATTRIBUTES = [
    "Cozy", "Trendy", "Romantic", "Quiet", "Elegant", "Casual", 
    "Outdoor", "Healthy", "Dessert",
    "Coffee", "Parking", "Wheelchair",
    "Wi-Fi"
]

# Initialize services
restaurant_service = RestaurantService()
llm_client = LLMClient()

# Initialize NLP models for ranking
try:
    embedder = PreferenceEmbedder(model_name='jinaai/jina-embeddings-v3')
    embedder.generate_attribute_embeddings(PREDEFINED_ATTRIBUTES)
    app.logger.info("PreferenceEmbedder initialized successfully.")
except Exception as e:
    app.logger.warning(f"Could not load PreferenceEmbedder: {e}")
    embedder = None

try:
    sentiment_analyzer = ReviewSentimentAnalyzer(
        en_model_path='hayn404/roberta-finetuned',
        ar_model_path='hayn404/araberta_finetuned'
    )
    app.logger.info("SentimentAnalyzer initialized successfully.")
except Exception as e:
    app.logger.warning(f"Could not load sentiment models: {e}")
    sentiment_analyzer = None

# Initialize ranking service with available models
if embedder and sentiment_analyzer:
    ranking_service = RankingService(embedder, sentiment_analyzer, PREDEFINED_ATTRIBUTES)
    app.logger.info("RankingService initialized with NLP models.")
else:
    ranking_service = None
    app.logger.warning("RankingService not available - using simple ranking.")

# Helper to run coroutine results in sync context
def run_sync(maybe_coro):
    if asyncio.iscoroutine(maybe_coro):
        return asyncio.run(maybe_coro)
    return maybe_coro

# Optional research/augmentation components (guarded by API keys)
data_repo = None
restaurant_agent = None
try:
    data_repo = DataRepository(raw_data_dir=os.path.dirname(BASIR_JSON_PATH))
except Exception as e:
    app.logger.info(f"DataRepository not initialized: {e}")

if SERPAPI_API_KEY:
    try:
        restaurant_agent = RestaurantSearchAgent(llm_client=llm_client, serpapi_key=SERPAPI_API_KEY)
        app.logger.info("RestaurantSearchAgent initialized.")
    except Exception as e:
        app.logger.warning(f"Failed to initialize RestaurantSearchAgent: {e}")

# --- API Routes ---
@app.route('/')
def index():
    """Serves the main UI page."""
    return render_template('index.html')

@app.route('/api/countries')
def get_countries():
    """Get list of available countries."""
    try:
        countries = restaurant_service.get_unique_countries()
        data_source = restaurant_service.get_data_source_info()
        
        return jsonify({
            "success": True, 
            "countries": countries,
            "data_source": data_source.get("source", "CSV"),
            "total_restaurants": data_source.get("stats", {}).get("total_restaurants", 0)
        })
    except Exception as e:
        app.logger.error(f"Error getting countries: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/cities', methods=['GET'])
def get_cities():
    """API endpoint to get cities for a specific country."""
    try:
        country = request.args.get('country')
        if not country:
            return jsonify({"success": False, "error": "Country parameter is required"}), 400
        
        cities = restaurant_service.get_unique_cities(country)
        return jsonify({"success": True, "cities": cities})
    except Exception as e:
        app.logger.error(f"Error getting cities for country {country}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/suitabilities')
def get_suitabilities():
    """Get list of available suitability options."""
    try:
        suitabilities = restaurant_service.get_unique_suitabilities()
        return jsonify({"success": True, "suitabilities": suitabilities})
    except Exception as e:
        app.logger.error(f"Error getting suitabilities: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/attributes')
def get_attributes():
    """Get predefined attribute names for preference selection."""
    try:
        return jsonify({"success": True, "attributes": PREDEFINED_ATTRIBUTES})
    except Exception as e:
        app.logger.error(f"Error getting attributes: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Get restaurant recommendations based on user preferences."""
    try:
        data = request.get_json()
        country = data.get('country')
        city = data.get('city')
        suitability = data.get('suitability')
        # Optional proximity inputs from client
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        max_distance = data.get('max_distance', 10)
        # Optional multi-factor inputs
        user_preferences = data.get('preferences') or []
        if isinstance(user_preferences, str):
            # Allow either array of attributes or comma-separated string
            user_preferences = [p.strip() for p in user_preferences.split(',') if p.strip()]
        try:
            user_budget = int(data.get('budget')) if data.get('budget') is not None else 2
        except Exception:
            user_budget = 2
        
        if not country:
            return jsonify({"success": False, "error": "Country is required"}), 400
        
        app.logger.info(f"Getting recommendations for: Country={country}, City={city}, Suitability={suitability}")
        
        restaurants = restaurant_service.find_restaurants(
            country=country,
            city=city if city != "Other" else None,
            suitability=suitability
        )
        
        extra_names = []

        # Augment when too few places and agent/collector available (does not rely on JSON dataset)
        if restaurants is not None and len(restaurants) < 1 and city and restaurant_agent and MAPS_API_KEY:
            try:
                app.logger.info(f"Fewer than 1 place found in {city}. Augmenting data with search agent.")
                new_places_json = restaurant_agent.execute(query=city, max_urls_to_process=50)
                new_places_json = run_sync(new_places_json)
                new_place_names = []
                try:
                    new_place_names = json.loads(new_places_json).get("restaurants", [])
                except Exception:
                    app.logger.warning("Could not parse search agent output; skipping augmentation parse.")
                if new_place_names:
                    temp_csv_path = "temp_new_restaurants.csv"
                    import pandas as pd
                    # Cap how many new places to fetch to reduce load
                    names_capped = new_place_names[:10]
                    extra_names = names_capped
                    pd.DataFrame({"Restaurant Name": names_capped, "City": [city] * len(names_capped)}).to_csv(temp_csv_path, index=False)
                    # Instantiate collector with the temporary CSV input (class requires it at init)
                    collector = GoogleMapsCollectorBasir(api_key=MAPS_API_KEY, input_csv_path=temp_csv_path, output_dir=os.path.dirname(BASIR_JSON_PATH))
                    new_places_df, new_reviews_df = run_sync(collector.collect_data(max_places=len(names_capped)))
                    try:
                        os.remove(temp_csv_path)
                    except Exception:
                        pass
                    for _, row in new_places_df.iterrows():
                        new_place = row.to_dict()
                        if 'name' not in new_place and 'Restaurant Name' in new_place:
                            new_place['name'] = new_place['Restaurant Name']
                        # Attach reviews if any
                        if 'name' in new_place:
                            new_place['reviews'] = new_reviews_df[new_reviews_df['place_name'] == new_place['name']].to_dict('records')
                        if 'budget' not in new_place or new_place['budget'] is None:
                            new_place['budget'] = 0
                        restaurants.append(new_place)
                    app.logger.info(f"Augmented dataset to {len(restaurants)} places for {city}.")
            except Exception as e:
                app.logger.warning(f"Augmentation failed: {e}")

        if not restaurants:
            return jsonify({"success": True, "restaurants": [], "message": "No restaurants found matching your criteria."})

        # Build processed places for multi-factor ranking (ensure coords and budget present)
        processed_places = []
        for place in restaurants:
            lat = place.get('latitude')
            lon = place.get('longitude')
            if lat is None or lon is None:
                continue
            place['coords'] = (lat, lon)
            if 'budget' not in place or place['budget'] is None:
                place['budget'] = place.get('price_level') if isinstance(place.get('price_level'), int) else 0
            if 'reviews' not in place:
                place['reviews'] = []
            processed_places.append(place)

        if not processed_places:
            app.logger.warning("No places with valid coordinates to rank.")
            return jsonify({"success": True, "restaurants": []})

        # Prepare user data for multi-factor ranking
        user_coords = (float(latitude), float(longitude)) if latitude is not None and longitude is not None else None
        preferences_weights = {attr: 1.0 for attr in user_preferences} if user_preferences else {}
        user_data = {
            "preferences": preferences_weights,
            "budget": user_budget,
            "coords": user_coords
        }

        # Multi-factor ranking using the established ranking pipeline
        ranked_restaurants = rank_places(
            user_data=user_data,
            places_list=processed_places,
            embedder=embedder if embedder else PreferenceEmbedder(model_name='jinaai/jina-embeddings-v3'),
            sentiment_analyzer=sentiment_analyzer if sentiment_analyzer else ReviewSentimentAnalyzer(
                en_model_path='hayn404/roberta-finetuned', ar_model_path='hayn404/araberta_finetuned'
            ),
            predefined_attributes=PREDEFINED_ATTRIBUTES,
            max_distance_km=float(max_distance)
        )
        
        enhanced_restaurants = []
        for i, restaurant in enumerate(ranked_restaurants):
            # Attach distance and proximity if user location is provided and restaurant has coords
            try:
                if latitude is not None and longitude is not None:
                    rest_lat = restaurant.get('latitude')
                    rest_lon = restaurant.get('longitude')
                    if rest_lat is not None and rest_lon is not None:
                        distance_km = haversine(float(latitude), float(longitude), float(rest_lat), float(rest_lon))
                        restaurant['distance'] = distance_km
                        # Normalize proximity score using the existing utility
                        proximity_score = calculate_proximity_score(
                            (float(latitude), float(longitude)), (float(rest_lat), float(rest_lon)), max_distance_km=float(max_distance)
                        )
                        scoring_details = restaurant.get('scoring_details', {}) or {}
                        scoring_details['proximity_score'] = round(proximity_score, 4)
                        restaurant['scoring_details'] = scoring_details
            except Exception as e:
                app.logger.warning(f"Proximity computation failed for {restaurant.get('restaurant_name', restaurant.get('name', 'Unknown'))}: {e}")
            
            # --- THIS IS THE FIX ---
            # Correctly extract review text from list of dictionaries
            reviews = restaurant.get('reviews', [])
            reviews_text = ""
            if isinstance(reviews, list) and reviews:
                # Extract the 'text' from each dict, handling cases where 'text' might be missing
                review_texts = [str(review.get('text', '')) for review in reviews if isinstance(review, dict)]
                reviews_text = " ".join(review_texts)
            # ---------------------

            if reviews_text.strip():
                try:
                    ai_summary = llm_client.summarize_reviews(reviews_text, restaurant.get('restaurant_name', ''))
                    highlights = llm_client.extract_review_highlights(reviews_text)
                except Exception as e:
                    app.logger.warning(f"LLM processing failed for restaurant {restaurant.get('restaurant_name', 'Unknown')}: {e}")
                    ai_summary = "AI summary generation failed."
                    highlights = {"positive_aspects": [], "negative_aspects": [], "recommended_dishes": [], "overall_sentiment": "neutral"}
            else:
                # No reviews: treat as neutral and omit summary entirely for UI to hide the card
                ai_summary = None
                highlights = {"positive_aspects": [], "negative_aspects": [], "recommended_dishes": [], "overall_sentiment": "neutral"}

            if ai_summary:
                restaurant['ai_summary'] = ai_summary
            restaurant['highlights'] = highlights
            enhanced_restaurants.append(restaurant)
        
        # If user location provided, filter by max_distance and sort by distance asc
        if latitude is not None and longitude is not None:
            enhanced_restaurants = [r for r in enhanced_restaurants if r.get('distance') is not None and r['distance'] <= float(max_distance)]
            enhanced_restaurants.sort(key=lambda r: r.get('distance', float('inf')))
        
        # Limit to top # after filtering/sorting and assign rank
        enhanced_restaurants = enhanced_restaurants[:20]
        for idx, r in enumerate(enhanced_restaurants):
            r['rank'] = idx + 1
        
        return jsonify({"success": True, "restaurants": enhanced_restaurants, "more_restaurant_names": extra_names})
        
    except Exception as e:
        app.logger.error(f"Error getting recommendations: {e}")
        # Also log the traceback for detailed debugging
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)