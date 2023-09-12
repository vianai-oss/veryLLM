from shared.gpt_3_5_turbo import get_gpt_3_5_turbo_response


def is_true_llm(question: str, answer: str, context: str):
    messages = [
        {
            "role": "system",
            "content": "Using only the explicit details given in the context, assess the correctness of the answer. It is vital to prioritize accuracy: if there's any uncertainty about an answer being a direct match to the context, you should lean towards 'false'. Only indicate 'true' if you are entirely confident that the answer aligns perfectly with the context. Any slight deviations or ambiguities should be treated as 'false'.",
        },
        {
            "role": "user",
            "content": "Context: The sun is the central star of our solar system.\nQuestion: What is the sun in relation to our solar system?\nAnswer: The sun is the central star of our solar system.",
        },
        {
            "role": "assistant",
            "content": "true",
        },
        {
            "role": "user",
            "content": "Context: Elephants are known for their large ears and trunk.\nQuestion: What are elephants known for?\nAnswer: Elephants are recognized for their tusks and tail.",
        },
        {
            "role": "assistant",
            "content": "false",
        },
        {
            "role": "user",
            "content": "Context: Paris is the capital city of France.\nQuestion: Which city is the capital of France?\nAnswer: Paris is a major city in France.",
        },
        {
            "role": "assistant",
            "content": "false",
        },
        {
            "role": "user",
            "content": "Context: Chocolate is made from cocoa beans.\nQuestion: What is chocolate made from?\nAnswer: Chocolate comes from beans.",
        },
        {
            "role": "assistant",
            "content": "false",
        },
        {
            "role": "user",
            "content": "Context: The Eiffel Tower is a famous landmark in Paris.\nQuestion: Where is the Eiffel Tower located?\nAnswer: The Eiffel Tower is in France.",
        },
        {
            "role": "assistant",
            "content": "false",
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

    if "true" in response.lower():
        return 1
    else:
        return 0
