import os
from dotenv import load_dotenv

load_dotenv()

# Environment variables
MAPS_API_KEY = os.getenv("MAPS_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
BASIR_INPUT_CSV_PATH = os.path.join(RAW_DATA_DIR, 'Basir Research dataset - Restaurants .csv')
BASIR_JSON_PATH = os.path.join(RAW_DATA_DIR, 'basir_combined_places_reviews_final_20250710_142531.json')
