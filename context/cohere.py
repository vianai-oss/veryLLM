import os
from dotenv import load_dotenv
import weaviate

load_dotenv()

auth_config = weaviate.auth.AuthApiKey(api_key=os.environ.get("WEAVIATE_API_KEY"))

client = weaviate.Client(
    url="https://cohere-demo.weaviate.network/",
    auth_client_secret=auth_config,
    additional_headers={
        "X-Cohere-Api-Key": os.environ.get("COHERE_API_KEY"),
    },
)


def find_context(query, top_n=5):
    nearText = {"concepts": [query]}
    properties = ["text", "title", "url", "lang", "_additional {distance}"]

    where_filter = {
        "path": ["lang"],
        "operator": "Equal",
        "valueString": "en",
    }
    response = (
        client.query.get("Articles", properties)
        .with_where(where_filter)
        .with_near_text(nearText)
        .with_limit(top_n)
        .do()
    )

    result = response["data"]["Get"]["Articles"]

    for article in result:
        article["distance"] = article["_additional"]["distance"]
        del article["_additional"]

    return result
