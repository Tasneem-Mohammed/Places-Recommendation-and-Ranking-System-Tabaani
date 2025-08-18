import pandas as pd
import json
from typing import List, Dict, Any

class DataRepository:
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir

    def load_places_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Loads a CSV file into a Pandas DataFrame.
        
        Args:
            file_path (str): The full path to the CSV file.
            
        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded {len(df)} rows from {file_path}")
            return df
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return pd.DataFrame()
        except Exception as e:
            print(f"An error occurred while loading the CSV file: {e}")
            return pd.DataFrame()

    def save_to_json(self, data: List[Dict[str, Any]], file_path: str):
        """
        Saves a list of dictionaries to a JSON file.
        
        Args:
            data (List[Dict[str, Any]]): The data to save.
            file_path (str): The full path to the output JSON file.
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Data successfully saved to {file_path}")
        except IOError as e:
            print(f"Error saving file to {file_path}: {e}")

    def load_from_json(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads a list of dictionaries from a JSON file.
        
        Args:
            file_path (str): The full path to the JSON file.
            
        Returns:
            List[Dict[str, Any]]: The loaded data. Returns an empty list if an error occurs.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Successfully loaded data from {file_path}")
            return data
        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {file_path}.")
            return []
        except Exception as e:
            print(f"An unexpected error occurred while loading JSON: {e}")
            return []