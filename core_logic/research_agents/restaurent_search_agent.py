import os
from dotenv import load_dotenv
import re
import json
import asyncio
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
import serpapi
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from playwright.async_api import async_playwright
from src.tools.research.base import ResearchAgent # Assuming this is your base class
from src.llm import LLMClient # Assuming you have this LLMClient

load_dotenv() # Load environment variables from .env

class RestaurantSearchAgent(ResearchAgent):
    """
    A research agent that searches for restaurants in a given region and extracts their names.
    It includes its own web searching and content extraction functionalities.
    """

    def __init__(
        self,
        serpapi_key: str = os.getenv("SERPAPI_API_KEY"),
        timeout_seconds: int = 10
    ):
        """
        Initialize the RestaurantSearchAgent.

        Args:
            serpapi_key (str, optional): SerpApi API key. Defaults to env variable SERPAPI_API_KEY.
            timeout_seconds (int, optional): Timeout for web requests in seconds. Defaults to 10.

        Raises:
            ValueError: If serpapi_key is not provided or empty.
        """
        super().__init__(
            name="RestaurantSearchAgent",
            description="Searches for restaurants in a specified region and extracts their names."
        )
        if not serpapi_key:
            raise ValueError("SERPAPI_API_KEY must be provided or set as an environment variable")
        self.serpapi_key = serpapi_key
        self.timeout_seconds = timeout_seconds
        # This LLM is for the summarization/extraction part - it's currently not used in _extract_restaurant_names directly
        # If you intend to use Langchain's LLMChain, you would use this.
        # For direct message interaction, LLMClient is used.
        self.llm = ChatGroq(
            model_name=os.getenv("MODEL_NAME"),
            api_key=os.getenv("LLM_API_KEY_2"), # Note: Your LLMClient uses LLM_API_KEY, not LLM_API_KEY_2
            temperature=float(os.getenv("TEMPERATURE")),
            max_tokens=int(os.getenv("MAX_TOKENS"))
        )
        self.llm_client = LLMClient() # For _extract_restaurant_names which uses a custom prompt

    def preprocess_query(self, query: str) -> str:
        """
        Preprocess the query to optimize for restaurant searches.
        Adds relevant terms to the query.
        """
        self.log(f"Preprocessing query: {query}")
        # Make the search query specific to finding restaurants
        preprocessed = f"best restaurants in {query.strip()}"
        self.log(f"Preprocessed to: {preprocessed}")
        return preprocessed

    async def search_websites_async(self, query: str, number_of_links: int) -> List[str]:
        """
        Search for websites containing restaurant information using SerpApi.

        Args:
            query (str): The search query (e.g., "restaurants in Paris list").
            number_of_links (int): Number of links to retrieve.

        Returns:
            List[str]: List of URLs from search results.
        """
        self.log(f"Searching for websites with query: {query}")
        params = {
            "engine": "google",
            "q": query, # The preprocessed query already includes "restaurants list"
            "api_key": self.serpapi_key,
            "num": number_of_links
        }
        try:
            results = serpapi.search(params)
            website_urls = [result["link"] for result in results.get("organic_results", [])]
            self.log(f"Found {len(website_urls)} URLs")
            return website_urls
        except Exception as e:
            self.log(f"Error during SerpApi search: {e}")
            return []

    def clean_result_str(self, result_str: str) -> str:
        """
        Clean a string by normalizing spaces while preserving newlines.

        Args:
            result_str (str): The input string to clean.

        Returns:
            str: The cleaned string.
        """
        # This method is good for cleaning text before LLM processing, if needed
        cleaned = re.sub(r'(?<=\n)[ ]+', ' ', result_str)
        cleaned = re.sub(r'(?<!\n)[ ]{2,}', ' ', cleaned)
        return cleaned

    # Re-adding extract_json_section for consistency, though _extract_restaurant_names uses LLMClient directly
    def extract_json_section(self, result_str: str) -> Optional[Dict[str, Any]]:
        """
        Extract a JSON object from a string.
        Useful if the LLM sometimes outputs extra text around the JSON.

        Args:
            result_str (str): The input string containing a JSON object.

        Returns:
            Optional[Dict[str, Any]]: The parsed JSON object, or None if extraction fails.
        """
        result_str = self.clean_result_str(result_str)
        json_start = result_str.find('{')
        if json_start == -1:
            self.log("No JSON object found in the response")
            return None
        brace_count = 1
        pos = json_start + 1
        while pos < len(result_str) and brace_count > 0:
            if result_str[pos] == '{':
                brace_count += 1
            elif result_str[pos] == '}':
                brace_count -= 1
            pos += 1
        json_str = result_str[json_start:pos]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.log(f"Error decoding JSON: {e}")
            return None

    async def extract_content_with_playwright_async(self, url: str) -> tuple[str, str]:
        """
        Extract title and content from a webpage using Playwright.

        Args:
            url (str): The URL to scrape.

        Returns:
            tuple[str, str]: The page title and content.

        Raises:
            Exception: If content extraction fails.
        """
        self.log(f"Extracting content from {url}")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = await browser.new_page()
            page.set_default_timeout(self.timeout_seconds * 1000)
            try:
                await page.goto(url)
                title = await page.title()
                # Focus on common elements that hold main content, e.g., paragraphs, list items
                # You might need to refine these selectors based on common website structures for restaurants
                content_elements = await page.query_selector_all("p, li, h2, h3, span")
                content = "\n".join([await el.inner_text() for el in content_elements if await el.inner_text()])
                await browser.close()
                self.log(f"Extracted {len(content)} characters from {url}")
                return title, content
            except Exception as e:
                await browser.close()
                raise Exception(f"Playwright error on {url}: {str(e)}")

    async def _extract_restaurant_names(self, text: str, query: str) -> List[str]:
        """
        Uses the LLM to extract restaurant names from the given text.

        Args:
            text (str): The content of the webpage.
            query (str): The original query (e.g., "Paris").

        Returns:
            List[str]: A list of extracted restaurant names.
        """
        self.log(f"Using LLM to extract restaurant names for query: '{query}'")
        prompt = f"""
        From the following text, identify and extract all unique restaurant names.
        Focus only on actual restaurant names. Do not include general categories (e.g., "Italian restaurant", "cafe"),
        cities, neighborhoods, or other types of businesses.
        If no clear restaurant names are found, return an JSON array like this:
        []

        Format the output as a JSON array of strings, like this:
        ["Restaurant A Name", "Restaurant B Name", "Restaurant C Name"]

        Text:
        {text[:8000]} # Limit text to fit within typical LLM context windows

        Region related to the search: {query}
        """

        messages = [
            {"role": "user", "content": prompt},
        ]

        response = await self.llm_client.get_response(messages) # <<< AWAIT IS CRUCIAL HERE
        self.log(f"LLM raw response for restaurant extraction: {response}")

        try:
            # Attempt to find the JSON array in the response
            json_match = re.search(r'\[\s*".*?"(?:,\s*".*?")*\s*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                extracted_names = json.loads(json_str)
                if not isinstance(extracted_names, list):
                    self.log(f"LLM did not return a list within JSON: {extracted_names}")
                    return []
                # Filter out empty strings or non-strings and strip whitespace
                return sorted(list(set([name.strip() for name in extracted_names if isinstance(name, str) and name.strip()])))
            else:
                self.log("No valid JSON array found in LLM response for restaurant names.")
                return []
        except json.JSONDecodeError as e:
            self.log(f"Error decoding JSON from LLM response: {e}. Raw response: {response}")
            return []
        except Exception as e:
            self.log(f"An unexpected error occurred during restaurant name extraction: {e}")
            return []

    def postprocess_results(self, results: List[List[str]]) -> List[str]:
        """
        Postprocess the extracted restaurant names, removing duplicates and ensuring unique entries.

        Args:
            results (List[List[str]]): A list of lists, where each inner list contains restaurant names
                                        extracted from a single URL.

        Returns:
            List[str]: A unique, flattened, and sorted list of restaurant names.
        """
        self.log(f"Postprocessing {len(results)} sets of restaurant names.")
        all_restaurants = []
        for i, res_list in enumerate(results): # Added index for debugging clarity
            self.log(f"Postprocessing iteration {i}: Type of res_list: {type(res_list).__name__}")
            if isinstance(res_list, list): # Ensure it's a list before extending
                all_restaurants.extend(res_list)
            else:
                self.log(f"WARNING: Expected a list of strings for res_list, but got {type(res_list).__name__}. Skipping this entry.")

        # Remove duplicates and sort
        unique_restaurants = sorted(list(set(all_restaurants)))
        self.log(f"Found {len(unique_restaurants)} unique restaurant names after postprocessing.")
        return unique_restaurants

    async def _process_single_url(self, url: str, query: str) -> Optional[List[str]]:
        """
        Helper function to process a single URL, extract content, and then restaurant names.
        Handles its own exceptions.
        """
        self.log(f"Processing URL: {url}")
        try:
            title, content = await self.extract_content_with_playwright_async(url)
            if len(content) > 100: # Ensure enough content to process for the LLM
                restaurant_names = await self._extract_restaurant_names(content, query) # <<< AWAIT IS CRUCIAL HERE
                if restaurant_names:
                    self.log(f"Extracted {len(restaurant_names)} names from {url}")
                    return restaurant_names
            else:
                self.log(f"Skipping {url} due to insufficient content.")
        except Exception as e:
            self.log(f"Error processing URL {url}: {e}")
        return None # Return None if processing failed or no names extracted

    async def execute(self, query: str, max_urls_to_process: int = 5) -> str:
        """
        Executes the restaurant search by:
        1. Preprocessing the query.
        2. Searching for relevant websites.
        3. Extracting content from each relevant website concurrently.
        4. Using the LLM to extract restaurant names from the content.
        5. Postprocessing and returning the unique list of restaurant names.

        Args:
            query (str): The region name (e.g., "Paris").
            max_urls_to_process (int): Maximum number of URLs to scrape for content.

        Returns:
            str: A JSON string containing the unique list of restaurant names.
        """
        self.log(f"Executing restaurant search for region: {query}")
        preprocessed_query = self.preprocess_query(query)

        self.log(f"Initiating web search for '{preprocessed_query}'")
        urls = await self.search_websites_async(preprocessed_query, max_urls_to_process)
        self.log(f"Found {len(urls)} URLs for processing.")

        tasks = []
        for url in urls:
            tasks.append(self._process_single_url(url, query)) # These are now coroutine objects to be awaited

        # Run all URL processing tasks concurrently
        # return_exceptions=True means that if a task fails, its exception is returned as a result
        # instead of stopping all other tasks.
        processed_results = await asyncio.gather(*tasks, return_exceptions=True)
        self.log(f"asyncio.gather completed. Processed {len(processed_results)} URLs.")

        extracted_restaurant_lists = []
        for i, res in enumerate(processed_results):
            if isinstance(res, Exception):
                self.log(f"Error in processing URL at index {i}: {res}")
            elif res is not None: # res will be List[str] if successful, None if skipped or no names extracted
                extracted_restaurant_lists.append(res)
            # No need for an else here, as None values are intentionally skipped

        self.log(f"Collected {len(extracted_restaurant_lists)} lists of restaurants before final postprocessing.")
        final_restaurant_names = self.postprocess_results(extracted_restaurant_lists)
        final_answer = json.dumps({"restaurants": final_restaurant_names}, indent=2)
        self.log("Restaurant search completed.")
        return final_answer