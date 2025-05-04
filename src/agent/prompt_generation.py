import logging
import argparse
from langchain_core.messages import HumanMessage, SystemMessage
from agent.state import State
from agent.whisper import transcribe_audio

# set logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def prompt_template(state: State) -> dict:
    """
    Generate a prompt template.

    Builds the prompt messages based on either a text review or an audio file.

    """

    system_prompt = """ You are a helpful and analytical AI assistant.

Your goal is to:
1. Analyze the following customer review.
2. Return a structured JSON with:
   - `sentiment`: one of ["positive", "negative", "neutral"]
   - `topics`: key subjects mentioned in the review (e.g., "customer service", "quality")
   - `mentions`: specific phrases or aspects noted (e.g., "long delivery", "friendly staff")
   - `summary`: short one-sentence recap of the review
   - `response_to_customer`: a professional and empathetic message addressing the customer's review, adapted to the sentiment and topics.

Please respond only in JSON format like this:

{
  "sentiment": "negative",
  "topics": ["delivery", "customer service"],
  "mentions": ["late delivery", "no response from support"],
  "summary": "The customer complained about a late delivery and lack of support response.",
  "response_to_customer": "We're truly sorry for the delay in delivery and the trouble reaching our support team. We’re investigating this and will do our best to prevent it in the future. Thank you for your feedback."
}

"""

    if state.customer_query and not state.customer_audio_file:
        customer_review = state.customer_query

    elif state.customer_audio_file and not state.customer_query:
        transcription_result = transcribe_audio(state)
        customer_review = transcription_result.get("audio_transcription", "")

    else:
        logger.error("No input provided by the user.")
        raise ValueError(
            "You must provide either a text review or an audio file reviews."
        )

    return {
        "prompt": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=customer_review),
        ]
    }


def parser_arguments():
    # parse arguments from command line

    parser = argparse.ArgumentParser(
        description="prompt template input parameters", fromfile_prefix_chars="@"
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
        help="provide text reviews by the user.",
        required=False,
        default=None,
    )

    args = parser.parse_args()

    return args


def main():
    args = parser_arguments()

    state = State(
        customer_audio_file=args.customer_audio_file, customer_query=args.customer_query
    )
    return state


if __name__ == "__main__":
    state = main()
    prompt = prompt_template(state)
    print("\n")
    print(prompt)
    print("\n")
