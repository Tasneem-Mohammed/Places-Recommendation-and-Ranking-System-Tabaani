# app/services/csv_data_service.py

import pandas as pd
import ast
import logging
from typing import List, Dict
import os

logger = logging.getLogger(__name__)

class CSVDataService:
    """Service for loading and querying restaurant data from a CSV file."""
    
    def __init__(self, csv_path: str = None):
        if csv_path is None:
            from config.settings import RESTAURANT_CSV_PATH
            csv_path = RESTAURANT_CSV_PATH
        
        self.csv_path = csv_path
        self.data = None
        self.initialized = False
        
        try:
            self.load_data()
            self.initialized = True
            logger.info(f"✅ CSV Data Service initialized successfully with {len(self.data)} restaurants.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize CSV Data Service: {e}", exc_info=True)
            self.initialized = False
    
    def load_data(self):
        """
        Load restaurant data from CSV, standardize, and clean it robustly.
        This function is designed to NOT fail if optional columns are missing.
        """
        try:
            self.data = pd.read_csv(self.csv_path)

            # --- STEP 1: HANDLE DUPLICATE COLUMNS AND RENAME ---
            # The CSV has both original and standardized columns, so we need to handle this carefully
            # First, let's use the standardized columns if they exist, otherwise fall back to original ones
            
            # Map the standardized columns we want to use
            column_mapping = {
                'restaurant_name': 'Restaurant Name',  # Use original if standardized doesn't exist
                'country': 'Country',
                'city': 'City', 
                'suitability': 'Suitability',
                'type_of_cuisine': 'Type of Cuisine', 
                'avg_price_usd': 'Avg Price per Person (USD)',  
                'budget_range': 'Budget Range',
                'rating': 'rating',
                'recommended_dish': 'Recommended Dish',  
                'meal_description': 'Meal Description',
                'address': 'address',
                'reviews': 'reviews',
                'latitude': 'latitude',
                'longitude': 'longitude',
                'num_reviews': 'num_reviews'
            }
            
            # Create the final DataFrame with standardized column names
            final_data = {}
            
            for target_col, source_col in column_mapping.items():
                if source_col in self.data.columns:
                    final_data[target_col] = self.data[source_col]
                else:
                    # If the source column doesn't exist, create a default column
                    if target_col == 'reviews':
                        final_data[target_col] = [[] for _ in range(len(self.data))]
                    elif target_col in ['avg_price_usd', 'rating', 'num_reviews', 'latitude', 'longitude']:
                        final_data[target_col] = 0.0
                    else:
                        final_data[target_col] = ''
            
            self.data = pd.DataFrame(final_data)

            # --- STEP 2: DATA CLEANING AND TYPE CONVERSION ---
            self.data = self.data.dropna(subset=['country', 'restaurant_name'])
            self.data = self.data.fillna({'city': 'Unknown', 'suitability': 'Any'})

            # Convert types safely
            if 'rating' in self.data.columns:
                self.data['rating'] = pd.to_numeric(self.data['rating'], errors='coerce').fillna(3.5)
            if 'avg_price_usd' in self.data.columns:
                self.data['avg_price_usd'] = pd.to_numeric(self.data['avg_price_usd'], errors='coerce').fillna(0.0)
            if 'num_reviews' in self.data.columns:
                self.data['num_reviews'] = pd.to_numeric(self.data['num_reviews'], errors='coerce').fillna(0)
            if 'suitability' in self.data.columns:
                self.data['suitability'] = self.data['suitability'].astype(str)

            # Safely parse the 'reviews' column
            def parse_reviews(reviews_val):
                if pd.isna(reviews_val) or reviews_val in ('[]', ''): return []
                if isinstance(reviews_val, list): return reviews_val # Already parsed
                try: return ast.literal_eval(str(reviews_val))
                except (ValueError, SyntaxError): return [{'text': str(reviews_val), 'rating': None}]
            
            if 'reviews' in self.data.columns:
                self.data['reviews'] = self.data['reviews'].apply(parse_reviews)

            logger.info(f"Data loaded and processed. {len(self.data)} restaurants are ready.")

        except Exception as e:
            logger.error(f"CRITICAL ERROR in load_data: {e}")
            raise

    # --- The rest of the functions remain the same ---
    def get_unique_countries(self) -> List[str]:
        if not self.initialized: return []
        return sorted(self.data['country'].dropna().unique().tolist())
    
    def get_unique_cities(self, country: str = None) -> List[str]:
        if not self.initialized: return []
        df = self.data
        if country:
            df = self.data[self.data['country'] == country]
        return sorted(df['city'].dropna().unique().tolist())
    
    def get_unique_suitabilities(self) -> List[str]:
        if not self.initialized: return []
        all_suitabilities = set()
        for item in self.data['suitability'].dropna():
            for s in item.split(','):
                all_suitabilities.add(s.strip())
        return sorted(list(all_suitabilities))
    
    def find_restaurants(self, country: str = None, city: str = None, suitability: str = None) -> List[Dict]:
        if not self.initialized: return []
        df = self.data.copy()
        if country:
            df = df[df['country'] == country]
        if city and city != "Other":
            df = df[df['city'] == city]
        if suitability:
            df = df[df['suitability'].str.contains(suitability, case=False, na=False)]
        return df.to_dict('records')
            
    def get_restaurant_by_name(self, restaurant_name: str) -> Dict:
        if not self.initialized: return None
        result_df = self.data[self.data['restaurant_name'] == restaurant_name]
        if result_df.empty: return None
        return result_df.iloc[0].to_dict()
    
    def get_stats(self) -> Dict:
        if not self.initialized: return {}
        return {
            'total_restaurants': len(self.data),
            'countries': self.data['country'].nunique(),
            'cities': self.data['city'].nunique(),
            'avg_rating': self.data['rating'].mean()
        }
