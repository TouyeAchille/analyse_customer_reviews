import os
import dotenv
import logging
import argparse
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from agent.prompt_generation import prompt_template
from agent.state import State

# set logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)

# load environement variable
dotenv.load_dotenv()


def prompt(state: State):
    return prompt_template(state).prompt


def classify_reviews(state: State):
    """
    Classifies the customer reviews into a category or sentiment (positive, negative, neutral),
    extract key topics and product mentions.

    Classifies the customer reviews using GPT model
    and returns an updated State with the GPT answer.

    Returns:
    : json format.

    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables.")
        raise EnvironmentError("Missing OpenAI API Key. Please set OPENAI_API_KEY.")

    try:
        logger.info(f"Initializing ChatOpenAI model: {state.gpt_model_name}")
        gpt4o_mini = ChatOpenAI(
            model=state.gpt_model_name,
            temperature=state.temperature,
            max_tokens=state.max_tokens,
            # api_key=api_key,
        ).with_structured_output(method="json_mode")

        logger.info("Creating and running the prompt chain.")
        chain = RunnableLambda(prompt) | gpt4o_mini
        response = chain.invoke(state)
        logger.info("Review classification completed successfully.")

        return {"gpt_answer": response}  # final output

    except Exception as e:
        logger.exception("An error occurred during review classification.")
        raise RuntimeError(f"Review classification failed: {str(e)}")


def parser_arguments():
    # parse arguments from command line

    parser = argparse.ArgumentParser(
        description="gpt model input parameters",
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

    args = parser.parse_args()

    return args


def main():
    args = parser_arguments()

    state = State(
        customer_query=args.customer_query,
        customer_audio_file=args.customer_audio_file,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        gpt_model_name=args.gpt_model_name,
    )

    return state


if __name__ == "__main__":
    state = main()
    response = classify_reviews(state)
    print("-----------------------------------")
    print("GPT Answer:")
    print(response["gpt_answer"])
