import httpx
import logging
import os
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Set up logging for both clients
llm_client_logger = logging.getLogger("app.services.llm")
llm_client_logger.setLevel(logging.DEBUG)
llm_client_logger.info("LLMClient module initialized.")

class LLMClient:
    """Manages communication with LLM providers for both summarization and research tasks."""

    def __init__(self) -> None:
        """Initializes the local LLM for summarization and the remote LLM client for research."""
        #  Initializing summarization model
        self.summarizer = self._initialize_summarizer()
        
        #  Initializing remote research client attributes
        self.conversation_history: list[dict[str, str]] = []
        llm_client_logger.info("LLMClient: Initialized. Conversation history reset.")

    def _initialize_summarizer(self):
        """Helper method to initialize the summarization pipeline."""
        try:
            model_name = "facebook/bart-large-cnn"
            summarizer_pipeline = pipeline(
                "summarization", 
                model=model_name,
                tokenizer=model_name,
                max_length=150,
                min_length=30,
                do_sample=False
            )
            llm_client_logger.info("LLMClient: Initialized with Hugging Face BART summarization model.")
            return summarizer_pipeline
        except Exception as e:
            llm_client_logger.error(f"Failed to initialize Hugging Face pipeline: {e}")
            return None

    async def _send_request_to_llm(self, messages: list[dict[str, str]]) -> str:
        """
        Sends an asynchronous request to the remote LLM provider.
        This is a private method for the research agent functionality.
        """
        url = os.getenv("BASE_URL")
        llm_api_key = os.getenv('LLM_API_KEY')
        model_name = os.getenv("MODEL_NAME")
        temperature = os.getenv("TEMPERATURE", "0.3")
        max_tokens = os.getenv("MAX_TOKENS", "4096")

        if not url:
            llm_client_logger.error("LLMClient: BASE_URL environment variable is not set.")
            return "Error: LLM base URL is not configured."
        if not llm_api_key:
            llm_client_logger.error("LLMClient: LLM_API_KEY environment variable is not set.")
            return "Error: LLM API key is not configured."
        if not model_name:
            llm_client_logger.warning("LLMClient: MODEL_NAME environment variable is not set. Using default or provider's default.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {llm_api_key}",
        }

        try:
            parsed_temperature = float(temperature)
        except ValueError:
            llm_client_logger.error(f"LLMClient: Invalid TEMPERATURE value '{temperature}'. Defaulting to 0.3.")
            parsed_temperature = 0.3
        
        try:
            parsed_max_tokens = int(max_tokens)
        except ValueError:
            llm_client_logger.error(f"LLMClient: Invalid MAX_TOKENS value '{max_tokens}'. Defaulting to 4096.")
            parsed_max_tokens = 4096

        payload = {
            "messages": messages,
            "model": model_name,
            "temperature": parsed_temperature,
            "max_tokens": parsed_max_tokens,
            "stream": False,
        }

        llm_client_logger.debug(f"LLMClient: Sending request to URL: {url}")
        
        llm_client_logger.debug(f"LLMClient: Request Payload (messages count: {len(messages)}, model: {payload['model']},"
                                 f" temp: {payload['temperature']}, max_tokens: {payload['max_tokens']}): {payload['messages'][-1]['content'][:100]}...")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                
                choices = data.get("choices", [])
                if not choices or "message" not in choices[0] or "content" not in choices[0]["message"]:
                    llm_client_logger.error("LLM response missing required keys. Raw data: %s", data)
                    return "Error: LLM did not return a valid response format."
                
                content = choices[0]["message"]["content"]
                llm_client_logger.info(f"LLMClient: Successfully retrieved LLM content (snippet): {content[:100]}...")
                return content

        except httpx.TimeoutException as e:
            llm_client_logger.error(f"LLMClient: Request timed out: {str(e)}")
            return "Request timed out."
        except httpx.RequestError as e:
            error_message = f"LLMClient: Error getting LLM response: {str(e)}"
            llm_client_logger.error(error_message, exc_info=True)
            return error_message
        except Exception as e:
            llm_client_logger.critical(f"LLMClient: An unexpected error occurred: {type(e).__name__} - {str(e)}", exc_info=True)
            return "An unexpected error occurred."
            
    async def get_response(self, messages: list[dict[str, str]]) -> str:
        """
        Get a response from the remote LLM based on provided messages.
        Does NOT manage conversation history internally.
        """
        llm_client_logger.warning("LLMClient: Using 'get_response'. Consider 'get_response_with_history' for conversational context.")
        return await self._send_request_to_llm(messages)

    def summarize_reviews(self, reviews_text: str, restaurant_name: str = "") -> str:
        """
        Summarize restaurant reviews using the local LLM.
        
        Args:
            reviews_text: Combined text of restaurant reviews
            restaurant_name: Name of the restaurant (optional)
            
        Returns:
            Summarized review text
        """
        if not self.summarizer or not reviews_text or reviews_text.strip() == "":
            return "No reviews available for analysis."

        try:
            cleaned_reviews = self._clean_reviews_text(reviews_text)
            
            if len(cleaned_reviews.split()) < 10:
                return "Limited review data available. Customers have shared positive feedback about this restaurant."
            
            max_input_length = 1000
            if len(cleaned_reviews) > max_input_length:
                cleaned_reviews = cleaned_reviews[:max_input_length] + "..."
            
            prompt_text = f"Customer reviews for {restaurant_name}: {cleaned_reviews}"
            
            summary_result = self.summarizer(
                prompt_text,
                max_length=120,
                min_length=25,
                do_sample=False
            )
            
            summary = summary_result[0]['summary_text']
            summary = self._enhance_restaurant_summary(summary, restaurant_name)
            
            llm_client_logger.info(f"LLMClient: Successfully summarized reviews for {restaurant_name}")
            return summary

        except Exception as e:
            error_message = f"Error summarizing reviews: {str(e)}"
            llm_client_logger.error(error_message)
            return "Unable to generate AI summary at this time. Please check individual reviews for detailed feedback."

    def _clean_reviews_text(self, reviews_text: str) -> str:
        """Clean and prepare reviews text for summarization."""
        cleaned = re.sub(r'\s+', ' ', reviews_text.strip())
        sentences = cleaned.split('.')
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        return '. '.join(meaningful_sentences)

    def _enhance_restaurant_summary(self, summary: str, restaurant_name: str) -> str:
        """Enhance the generated summary to be more restaurant-focused."""
        if restaurant_name and restaurant_name.lower() not in summary.lower():
            summary = f"At {restaurant_name}, {summary.lower()}"
        
        if not summary.endswith('.'):
            summary += '.'
        
        summary = summary[0].upper() + summary[1:] if len(summary) > 1 else summary.upper()
        return summary

    def generate_restaurant_insights(self, restaurant_data: dict) -> str:
        """
        Generate insights about a restaurant based on its data.
        This creates a template-based description since we're using a summarization model.
        """
        try:
            restaurant_name = restaurant_data.get('restaurant_name', 'This restaurant')
            cuisine = restaurant_data.get('type_of_cuisine', restaurant_data.get('cuisine', 'cuisine'))
            price_range = restaurant_data.get('budget_range', 'moderate pricing')
            rating = restaurant_data.get('rating', 0)
            suitability = restaurant_data.get('suitability', 'various occasions')
            recommended_dish = restaurant_data.get('recommended_dish', '')

            insights_parts = []
            insights_parts.append(f"{restaurant_name} specializes in {cuisine}")
            
            if price_range:
                insights_parts.append(f"with {price_range}")
            
            if rating and rating > 0:
                rating_text = "excellent" if rating >= 4.5 else "very good" if rating >= 4.0 else "good" if rating >= 3.5 else "decent"
                insights_parts.append(f"and maintains {rating_text} customer satisfaction")
            
            if suitability:
                insights_parts.append(f"Perfect for {suitability}")
            
            if recommended_dish:
                insights_parts.append(f"Don't miss their {recommended_dish}")

            insights = '. '.join(insights_parts) + '.'
            
            llm_client_logger.info(f"LLMClient: Generated insights for {restaurant_name}")
            return insights

        except Exception as e:
            llm_client_logger.error(f"Error generating insights: {str(e)}")
            return "A quality dining establishment offering great food and service."

    def extract_review_highlights(self, reviews_text: str) -> dict:
        """
        Extract key highlights from restaurant reviews using pattern matching.
        Since we're using a summarization model, we'll use rule-based extraction.
        """
        if not reviews_text or reviews_text.strip() == "":
            return {
                "positive_aspects": [], "negative_aspects": [],
                "recommended_dishes": [], "overall_sentiment": "neutral"
            }

        try:
            reviews_lower = reviews_text.lower()
            
            positive_keywords = [
                "excellent", "amazing", "delicious", "fantastic", "wonderful", "great",
                "perfect", "outstanding", "superb", "incredible", "awesome", "love",
                "best", "fresh", "tasty", "flavorful", "good service", "friendly staff"
            ]
            
            negative_keywords = [
                "terrible", "awful", "bad", "poor", "disappointing", "worst", "horrible",
                "slow service", "rude", "cold food", "overpriced", "dirty", "long wait"
            ]
            
            positive_aspects = [keyword.title() for keyword in positive_keywords if keyword in reviews_lower]
            negative_aspects = [keyword.title() for keyword in negative_keywords if keyword in reviews_lower]
            
            dish_patterns = [
                r'(pizza|burger|pasta|salad|steak|chicken|fish|soup|sandwich|dessert)',
                r'(appetizer|main course|entree|side dish)'
            ]
            
            recommended_dishes = []
            for pattern in dish_patterns:
                matches = re.findall(pattern, reviews_lower)
                recommended_dishes.extend([match.title() for match in matches])
            
            positive_count = len(positive_aspects)
            negative_count = len(negative_aspects)
            
            if positive_count > negative_count:
                overall_sentiment = "positive"
            elif negative_count > positive_count:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
            
            highlights = {
                "positive_aspects": list(set(positive_aspects))[:5],
                "negative_aspects": list(set(negative_aspects))[:3],
                "recommended_dishes": list(set(recommended_dishes))[:3],
                "overall_sentiment": overall_sentiment
            }
            
            llm_client_logger.info("LLMClient: Successfully extracted review highlights")
            return highlights
            
        except Exception as e:
            llm_client_logger.error(f"Error extracting highlights: {str(e)}")
            return {
                "positive_aspects": ["Good food quality"], "negative_aspects": [],
                "recommended_dishes": [], "overall_sentiment": "positive"
            }