from shared.gpt_3_5_turbo import get_gpt_3_5_turbo_response


def is_relevant_llm(question: str, answer: str, context: str):
    messages = [
        {
            "role": "system",
            "content": (
                "Determine if the provided context contains enough information "
                "to accurately assess the correctness of the answer to the given question. "
                "Respond with 'relevant' if the context can be used to decide the truth value "
                "of the question-answer pair and 'insufficient' if not."
            ),
        },
        {
            "role": "user",
            "content": f"Context: {context}\nQuestion: {question}\nAnswer: {answer}",
        },
    ]

    response = get_gpt_3_5_turbo_response(
        messages=messages,
        temperature=0.3,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return 1 if "relevant" in response.lower() else 0
