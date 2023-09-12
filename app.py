# (c) 2023 Vianai Systems, Inc.
from flask import Flask, request, jsonify, stream_with_context, Response
from flask_cors import CORS
import os

from shared.custom_types import DetailedQuestionAnswerPair
from shared.utils import int64_to_int
from classes.Validator import Validator
from verify.main import verify
from shared.gpt_3_5_turbo import get_gpt_3_5_turbo_response
from shared.split_sentences import split_sentences
from updates.update_validator import get_best_validator


app = Flask(__name__)
CORS(app)

id = get_best_validator()
print(f"Using validator {id}")
validator = Validator(id=id)
validator.load()


@app.route("/generate_answer", methods=["POST"])
def generate_answer():
    data = request.get_json()
    prompt = data.get("question", "")
    model = data.get("model", "")

    if model == "gpt-3.5-turbo":
        response = get_gpt_3_5_turbo_response(
            messages=[
                {"role": "user", "content": prompt + "\nRespond in a concise manner."}
            ]
        )
    else:
        return jsonify({"error": "Invalid model."})
    output = {"text": response, "sentences": split_sentences(response)}
    return jsonify(output)


@app.route("/verify_complete", methods=["POST"])
def verify_complete():
    data = request.get_json()

    question = data.get("question", "")
    answer = data.get("answer", "")
    answer_sentences = data.get("answer_sentences", [])

    query = DetailedQuestionAnswerPair(question, answer, answer_sentences)

    def generate():
        for result in verify(query, validator, top_n=8):
            yield jsonify(int64_to_int(result)).get_data(as_text=True) + "\n"

    return Response(stream_with_context(generate()), content_type="application/json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
