import json
import re
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError


class PromptClassification:
    """Builder class for classifying user prompts."""

    def __init__(self):
        """Initialize the prompt builder."""
        self.prompts_dir = Path(__file__)/ "prompts"
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load the system prompt from the text file.

        Returns:
            System prompt string
        """
        prompt_file = self.prompts_dir / "classifier.txt"
        if not prompt_file.exists():
            # Fallback to default prompt if file doesn't exist
            return (
                "You are a prompt difficulty classifier. Classify the given user prompt into exactly one of three labels—"
                "simple, medium, or complex—based on the level of reasoning, technical depth, and domain knowledge required. "
                "Output only one label with no explanations, no additional text, and no formatting. "
                "Do not repeat the user prompt or add punctuation."
            )
        return prompt_file.read_text().strip()


    def create_classifier_prompt(self, user_prompt: str) -> List[Dict[str, str]]:
        """Create a classsifier prompt with user_prompt and system prompt.

        Args:
            user_prompt: User question

        Returns:
            Formatted Chat message
        """

        prompt = [
                {
                    "role": "system",
                    "content": f"{self.system_prompt}"
                },
                {
                    "role": "user",
                    "content": f"{user_prompt}",
                },
            ]

        return prompt

    