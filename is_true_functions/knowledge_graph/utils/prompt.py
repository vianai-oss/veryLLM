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
def get_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content": 'Your task is to extract relationships from a provided document. Each relationship must be returned in a specific format: [ENTITY 1, RELATIONSHIP, ENTITY 2]. Each and every relationship returned must strictly have all these three parts, without any exceptions. The entities should be as concise as possible and the relationship is directed, so order matters. Importantly, do not include any extra information, commentary or explanations, and avoid using ellipsis ("...") in your output. If a relationship contains a compound object (e.g., "airstrip and testing facility"), please break it down into separate relationships (e.g., [["Area 51", "is", "airstrip"], ["Area 51", "is", "testing facility"]]). If no relationships are present, your response should be [["None"]].',
            },
            {
                "role": "user",
                "content": "Microsoft, founded by Bill Gates and Paul Allen, is a multinational technology company.",
            },
            {
                "role": "assistant",
                "content": '[["Microsoft", "founded by", "Bill Gates"], ["Microsoft", "founded by", "Paul Allen"], ["Microsoft", "is", "technology company"]]',
            },
            {
                "role": "user",
                "content": "Barack Obama was the 44th president of the United States. He was born in Hawaii.",
            },
            {
                "role": "assistant",
                "content": '[["Barack Obama", "was", "44th president"], ["Barack Obama", "born in", "Hawaii"]]',
            },
            {
                "role": "user",
                "content": "Python is a high-level, interpreted programming language. It was created by Guido van Rossum.",
            },
            {
                "role": "assistant",
                "content": '[["Python", "is", "programming language"], ["Python", "created by", "Guido van Rossum"]]',
            },
            {
                "role": "user",
                "content": "The Eiffel Tower, located in Paris, is one of the most famous landmarks in the world.",
            },
            {
                "role": "assistant",
                "content": '[["Eiffel Tower", "located in", "Paris"], ["Eiffel Tower", "is", "famous landmark"]]',
            },
            {
                "role": "user",
                "content": "The Amazon rainforest is the world's largest tropical rainforest. It is located in South America.",
            },
            {
                "role": "assistant",
                "content": '[["Amazon rainforest", "is", "largest rainforest"], ["Amazon rainforest", "located in", "South America"]]',
            },
            {
                "role": "user",
                "content": "Twitter",
            },
            {"role": "assistant", "content": '[["None"]]'},
            {
                "role": "user",
                "content": "The United States of America is large and diverse. It is home to many famous cities.",
            },
            {
                "role": "assistant",
                "content": '[["United States of America", "is", "large"], ["United States of America", "is", "diverse"], ["United States of America", "is home to", "many famous cities"]]',
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response
