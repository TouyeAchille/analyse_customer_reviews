"""Define the state structures for the agent."""

import logging
from pydantic import BaseModel, Field, root_validator
from typing import Optional, Literal

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


class State(BaseModel):
    """state"""

    # user input
    customer_query: Optional[str] = Field(
        default=None, description="Text input provided by the user."
    )
    customer_audio_file: Optional[str] = Field(
        default=None, description="file path Raw audio input provided by the user."
    )
    input_mode: Optional[Literal["text", "voice"]] = None

    # gpt parameters
    gpt_model_name: str = Field(
        default="gpt-4o-mini", description=" type of gpt model name to use."
    )
    temperature: float = Field(
        default=0.0, description=" zero mean the model will become deterministic."
    )
    max_tokens: int = Field(default=4096, description="max numbers of output tokens")
    voice_language: str = Field(
        default="en", description=" The language of the input audio"
    )

    # whisper model for transcription
    speech2text_model_name: str = Field(
        default="whisper-1", description="whisper model name for speech to text"
    )

    # output for each step (node)
    gpt_answer: dict = Field(default=None, description="gpt output")
    prompt: list = Field(default=None, description="prompt template")
    audio_transcribe: str = Field(
        default=None, description="output of audio transcription with wishper model"
    )

    #@root_validator(pre=True)
    def detect_input_mode(cls, values: dict):
        query = values.get("customer_query")
        audio = values.get("customer_audio_file")

        logger.debug(
            f"Detecting input mode. Received query: {bool(query)}, audio: {bool(audio)}"
        )

        if query and not audio:
            values["input_mode"] = "text"
            logger.info("Input mode set to 'text'")
        elif audio and not query:
            values["input_mode"] = "voice"
            logger.info("Input mode set to 'voice'")
        elif not query and not audio:
            logger.error("No input provided by the user.")
            raise ValueError("No input detected. Please provide either text or audio.")
        else:
            logger.error(
                "Both text and audio inputs were provided. Input is ambiguous."
            )
            raise ValueError(
                "Ambiguous input. Please provide either text or audio, not both."
            )

        return values
