import httpx
import logging
import os
llm_client_logger = logging.getLogger("dlai_agent.LLMClient")
llm_client_logger.setLevel(logging.DEBUG) # Ensure this logger instance also logs debug
llm_client_logger.info("LLMClient module initialized.")


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self) -> None:
        self.conversation_history: list[dict[str, str]] = []
        llm_client_logger.info("LLMClient: Initialized. Conversation history reset.")

    async def _send_request_to_llm(self, messages: list[dict[str, str]]) -> str:
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
                                f" temp: {payload['temperature']}, max_tokens: {payload['max_tokens']}): {payload['messages'][-1]['content'][:100]}...") # Log last message content snippet

        try:
            async with httpx.AsyncClient(timeout=30.0) as client: # Added a timeout for robustness
                response = await client.post(url, headers=headers, json=payload)

                llm_client_logger.debug(f"LLMClient: Raw response status code: {response.status_code}")
                llm_client_logger.debug(f"LLMClient: Raw response content: {response.text[:500]}...") # Log content snippet

                response.raise_for_status() # This will raise the HTTPStatusError on 4xx/5xx
                data = response.json()
                llm_client_logger.debug(f"LLMClient: Received successful response data keys: {data.keys()}")

                # Defensive check to avoid IndexError
                choices = data.get("choices", [])
                if not choices:
                    llm_client_logger.error("LLM response missing 'choices' array.")
                    return "Error: LLM did not return any choices."
                
                first_choice = choices[0]
                if "message" not in first_choice or "content" not in first_choice["message"]:
                    llm_client_logger.error("LLM response missing 'choices[0].message.content'. Raw data: %s", data)
                    return "Error: LLM did not return a valid response format."
                
                llm_client_logger.debug(f"LLMClient: data.choices[0]: {first_choice}")

                content = first_choice["message"]["content"]
                llm_client_logger.info(f"LLMClient: Successfully retrieved LLM content (snippet): {content[:100]}...")
                return content

        except httpx.TimeoutException as e:
            error_message = f"LLMClient: Request timed out: {str(e)}"
            llm_client_logger.error(error_message)
            return error_message
        except httpx.RequestError as e:
            error_message = f"LLMClient: Error getting LLM response due to network or request issue: {str(e)}"
            llm_client_logger.error(error_message)

            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
                llm_client_logger.error(f"LLMClient: HTTP Status code: {status_code}")
                llm_client_logger.error(f"LLMClient: Response details (from HTTPStatusError): {e.response.text}")
            
            return error_message
        except Exception as e:
            # Catch any other unexpected errors during JSON parsing or data access
            error_message = f"LLMClient: An unexpected error occurred: {type(e).__name__} - {str(e)}"
            llm_client_logger.critical(error_message, exc_info=True) # Log full traceback
            return error_message

    async def get_response(self, messages: list[dict[str, str]]) -> str:
        """
        (Original function) Get a response from the LLM based on provided messages.
        Does NOT manage conversation history internally.

        Args:
            messages: A list of message dictionaries for the current turn.

        Returns:
            The LLM's response as a string.
        """
        llm_client_logger.warning("LLMClient: Using 'get_response'. Consider 'get_response_with_history' for conversational context.")
        return await self._send_request_to_llm(messages)

