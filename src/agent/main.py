import logging
import dotenv
import argparse

from agent.state import State
from agent.whisper import transcribe_audio
from agent.prompt_generation import prompt_template
from agent.gpt import classify_reviews
from langgraph.graph import StateGraph, START, END

# set logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)

# load environement variable
dotenv.load_dotenv()


# Build graph
builder = StateGraph(State)

# define node (step)
builder.add_node("audio_transcription", transcribe_audio(State))
builder.add_node("prompt_generation", prompt_template(State))
builder.add_node("review_classification", classify_reviews(State))

# logic
builder.add_edge(START, "audio_transcription")
builder.add_edge(START, "prompt_generation")

builder.add_edge("text_retriever", "multimodal_llm")
builder.add_edge("image_retriever", "multimodal_llm")
builder.add_edge("review_classification", END)


# Compile graph
graph = builder.compile()


def parser_arguments():
    # parse arguments from command line

    parser = argparse.ArgumentParser(
        description="Prompt template input parameters",
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

    args = parser.parse_args()

    return args


def main():
    args = parser_arguments()

    # Initialisation de l'objet `state`
    state = State(
        customer_audio_file=args.customer_audio_file, customer_query=args.customer_query
    )
    return state
