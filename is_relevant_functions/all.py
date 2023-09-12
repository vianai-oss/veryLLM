### Import your is relevant function here ###
from is_relevant_functions.llm.main import is_relevant_llm

### Add your is relevant function to the list below ###
is_relevant_funcs = [
    {
        "func": is_relevant_llm,
        "description": "Uses the LLM to predict whether the answer is true or false.",
        "author": "Andrew",
        "requires_threshold": False,
        "ignore": False,
    },
]

### Do not modify below this line ###``
is_relevant_funcs_dict = {
    is_relevant_func["func"].__name__: is_relevant_func
    for is_relevant_func in is_relevant_funcs
}
