"""
Transcribe audio customer reviews using OpenAI Whisper via LangChain.

This module loads an audio file provided by the user (if any),
uses the Whisper model to transcribe it into text,
and saves the transcription into the appropriate output directory.

Main features:
- Automatically detects if an audio file was provided.
- Uses LangChain's OpenAIWhisperParser to transcribe.
- Logs all key steps: loading, transcribing, saving.
- Ensures API key is set via environment variable.
"""


import os
import dotenv
import logging
import argparse
from pathlib import Path
from langchain_core.documents.base import Blob
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser
from langchain_core.documents import Document
from agent.state import State

# set logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)

# load environement variable
dotenv.load_dotenv()

def transcribe_audio(state: State):
    """
    Transcribes an audio file into text using OpenAI's Whisper model.
    Handles cases where no audio is provided.

    Returns:
        dict: {"audio_transcription": "..."} or {"audio_transcription": ""}
    """
    if not state.customer_audio_file:
        logger.info("No audio file provided by the user.")
        return {"audio_transcription": ""}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        raise EnvironmentError("Missing OpenAI API Key. Please set OPENAI_API_KEY.")

    try:
        logger.info("Initializing Whisper model")
        whisper = OpenAIWhisperParser(
            model=state.speech2text_model_name,
            temperature=state.temperature,
            response_format='text',
            language=state.voice_language,
            api_key=api_key
        )

        # Define paths
        current_dir = Path(__file__).resolve().parent
        root_dir = current_dir.parent.parent
        audio_filepath = Path(root_dir, "datalake", "audio_reviews_dataset", state.customer_audio_file)


        if not audio_filepath.is_file():
            logger.warning("Audio file does not exist: %s", audio_filepath)
            return {"audio_transcription": ""}

        logger.info("Loading and transcribing audio: %s", audio_filepath)
        blob = Blob.from_path(audio_filepath)
        doc_transcription :  Document = whisper.parse(blob)
        text_transcription = doc_transcription[0].page_content

        output_dir = Path(root_dir, "datalake", "transcription_reviews")
        output_dir.mkdir(parents=True, exist_ok=True)

        transcription_file = output_dir / f"{audio_filepath.stem}.txt"
        with transcription_file.open("w", encoding="utf-8") as f:
            f.write(text_transcription)
        logger.info("Transcription saved to: %s", transcription_file)

        return {"audio_transcription": text_transcription}

    except Exception as e:
        logger.exception("An error occurred during transcription.")
        raise


def parser_arguments():
    # parse arguments from command line

    parser = argparse.ArgumentParser(
        description="whisper model input parameters",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--customer_audio_file",
        type=str,
        help="provide audio file name with extension, input provided by the user.",
        required=False,
        default="complaint3.mp3"
    )


    parser.add_argument(
        "--speech2text_model_name",
        type=str,
        help="Provide openai model name that can transcribe and translate audio into text.",
        required=False,
        default='whisper-1'
    )


    parser.add_argument(
        "--temperature",
        type=float,
        help="temperature value between 0 and 1 for LLM model for text generation",
        required=False,
        default=0.0,
    )


    parser.add_argument(
        "--voice_language",
        type=str,
        help="The language of the input audio.",
        required=False,
        default="en",
    )

    args = parser.parse_args()

    return args



def main():
     args = parser_arguments()
     state=State(
            customer_audio_file=args.customer_audio_file,
            temperature=args.temperature,
            speech2text_model_name=args.speech2text_model_name,
            voice_language=args.voice_language,

        )
     return state


state=main()
audio_transcription=transcribe_audio(state)
print(audio_transcription)
