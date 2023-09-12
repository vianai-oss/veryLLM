import openai
import tenacity
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

retry_decorator = tenacity.retry(
    stop=tenacity.stop_after_attempt(3),  # Number of retries before giving up
    wait=tenacity.wait_fixed(2),  # Time delay (in seconds) between retries
)


@retry_decorator
def create_text_embedding(text):
    # Calls the GPT-Ada API to create embeddings for a list of texts
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    embedding = response["data"][0]["embedding"]
    return embedding
