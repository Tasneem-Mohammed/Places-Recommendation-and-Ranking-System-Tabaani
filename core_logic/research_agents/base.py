from abc import ABC, abstractmethod
from typing import Any
import logging

logger = logging.getLogger(__name__)

class BaseResearchAgent(ABC):
    """
    Abstract base class for all research agents, defining the core lifecycle methods.
    """
    def __init__(self, name: str, description: str):
        """
        Initializes a research agent.

        Args:
            name (str): The name of the agent.
            description (str): A brief description of the agent's purpose.
        """
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, query: str, **kwargs) -> Any:
        """
        Executes the agent's primary research function.

        This method should be implemented to perform the main task,
        orchestrating preprocessing, execution steps, and postprocessing.

        Args:
            query (str): The primary research query.
            **kwargs: Additional parameters for customization.

        Returns:
            Any: The final, processed result of the research operation.
        """
        pass

    @abstractmethod
    def preprocess_query(self, query: str) -> str:
        """
        Preprocesses the query before the main execution.
        """
        pass

    @abstractmethod
    def postprocess_results(self, results: Any) -> Any:
        """
        Postprocesses the results after the main execution to format them.
        """
        pass

    def log_info(self, message: str) -> None:
        """Logs an informational message with the agent's name."""
        logger.info(f"[{self.name}] {message}")

    def log_error(self, message: str) -> None:
        """Logs an error message with the agent's name."""
        logger.error(f"[{self.name}] {message}")