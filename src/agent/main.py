import logging
import dotenv
import argparse

from agent.state import State
from agent.whisper import transcribe_audio
from agent.prompt_generation import prompt_template
from agent.gpt import classify_reviews
from langgraph.graph import StateGraph, START, END
from typing import Literal


# set logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)

# load environement variable
dotenv.load_dotenv()


def routing_input_mode(
    state: State,
) -> Literal["prompt_generation", "whisper_audio_transcription"]:
    # text input
    if state.customer_query:
       return "prompt_generation"
    # voice input
    elif state.customer_audio_file:
        return "whisper_audio_transcription"
    else:
        logger.error("No input provided by the user.")
        raise ValueError("No input detected. Please provide either text or audio.")


# Build graph
builder = StateGraph(State)

# define node (step)
builder.add_node("whisper_audio_transcription", transcribe_audio)  # whisper
builder.add_node("prompt_generation", prompt_template)  # prompt template
builder.add_node("gpt_reviews_classification", classify_reviews)  # gpt

# connect nodes
builder.add_conditional_edges(
    START, routing_input_mode
)  # Condition entry point : input can be text or audio


builder.add_edge(
    "whisper_audio_transcription", "prompt_generation"
)  # if audio input, then follow this logic

builder.add_edge(
    "prompt_generation", "gpt_reviews_classification"
)  # if text input, then follow this logic

builder.add_edge("gpt_reviews_classification", END)

#
# Compile graph
graph = builder.compile()


def parser_arguments():
    # parse arguments from command line

    parser = argparse.ArgumentParser(
        description="LLM input parameters",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--customer_audio_file",
        type=str,
        help="provide audio file name with extension, input provided by the user.",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--customer_query",
        type=str,
        help="Provide user reviews text.",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--speech2text_model_name",
        type=str,
        help="Provide openai model name that can transcribe and translate audio into text.",
        required=False,
        default="whisper-1",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        help="temperature value between 0 and 1 for LLM model for text generation",
        required=False,
        default=0.0,
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        help="maximum tokens for LLM model for text generation",
        required=False,
        default=4096,
    )

    parser.add_argument(
        "--gpt_model_name",
        type=str,
        help="maximum tokens for LLM model for text generation",
        required=False,
        default="gpt-4o-mini",
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

    response = graph.invoke(
        State(
            customer_audio_file=args.customer_audio_file,
            customer_query=args.customer_query,
            temperature=args.temperature,
            speech2text_model_name=args.speech2text_model_name,
            voice_language=args.voice_language,
            gpt_model_name=args.gpt_model_name,
            max_tokens=args.max_tokens,
        )
    )


    return response 


# run script
if __name__ == "__main__":
    # run the main function
    response = main()
    print("\n")
    print(response)
    print("\n")
