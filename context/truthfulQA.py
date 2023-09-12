import gdown
import json
import pandas as pd
import os

from shared.embedding import create_text_embedding
from is_true_functions.embedding_similarity.main import cos_similarity

context_file_url = "https://drive.google.com/uc?id=1tHCh9IgbUDEAiba_BXxWSTenUOLF9nXW"
local_path = "data/truthfulQA_document_embeddings_no_citations_V2.json"

if not os.path.exists(local_path):
    print("Downloading truthfulQA document embeddings...")
    # Download the file using gdown
    gdown.download(context_file_url, local_path, quiet=False)

with open(local_path, "r") as json_file:
    data = json.load(json_file)

url_chunk = []
paragraphs = []
embeddings = []

for index, (url, values) in enumerate(data.items()):
    doc_paragraphs = values[0] if values[0] else []
    doc_embeddings = values[1] if values[1] else []

    if doc_paragraphs and doc_embeddings:
        url_chunk.append(url)
        paragraphs.append(doc_paragraphs)
        embeddings.append(doc_embeddings)

reorganized_data = {
    "url_chunk": url_chunk,
    "paragraphs": paragraphs,
    "embeddings": embeddings,
}

df_context = pd.DataFrame(reorganized_data)


def find_truthful_qa_context(query, top_n=5):
    qa_embedding = create_text_embedding(query)

    most_relevant_context = {
        "cos_similarity": 0,
    }

    # Iterate through the DataFrame using iterrows()
    for _, row in df_context.iterrows():
        doc_url_chunk = row["url_chunk"]
        doc_paragraphs = row["paragraphs"]
        doc_embeddings = row["embeddings"]

        qa_doc_cos_similarity = cos_similarity(doc_embeddings, qa_embedding)

        if qa_doc_cos_similarity > most_relevant_context["cos_similarity"]:
            most_relevant_context["cos_similarity"] = qa_doc_cos_similarity
            most_relevant_context["doc_url_chunk"] = doc_url_chunk
            most_relevant_context["doc_paragraphs"] = doc_paragraphs

    return most_relevant_context
