### Import your prediction function here ###
from is_true_functions.embedding_similarity.main import (
    is_true_embedding,
    is_true_embedding_qa_negated,
)
from is_true_functions.entailment.main import (
    is_true_entailment,
    is_true_entailment_only_answer,
)
from is_true_functions.llm.main import is_true_llm
from is_true_functions.knowledge_graph.main import is_true_knowledge_graph

### Add your prediction function to the list below ###
prediction_funcs = [
    {
        "func": is_true_llm,
        "description": "Uses the LLM to predict whether the answer is true or false.",
        "author": "Andrew",
        "requires_threshold": False,
        "ignore": False,
    },
    {
        "func": is_true_embedding,
        "description": "Uses the embedding similarity to predict whether the answer is true or false.",
        "author": "Kevin",
        "requires_threshold": True,
        "ignore": False,
    },
    {
        "func": is_true_embedding_qa_negated,
        "description": "Uses the embedding similarity to predict whether the answer is true or false. Negates the question and answer.",
        "author": "Kevin",
        "requires_threshold": True,
        "ignore": False,
    },
    {
        "func": is_true_entailment,
        "description": "Uses the entailment model to predict whether the answer is true or false.",
        "author": "Andrew",
        "requires_threshold": True,
        "ignore": False,
    },
    {
        "func": is_true_entailment_only_answer,
        "description": "Uses the entailment model to predict whether the answer is true or false. Only uses the answer as the hypothesis.",
        "author": "Andrew",
        "requires_threshold": True,
        "ignore": False,
    },
    {
        "func": is_true_knowledge_graph,
        "description": "Uses the knowledge graph to predict whether the answer is true or false.",
        "author": "Andrew",
        "requires_threshold": True,
        "ignore": True,
    },
]
### Do not modify below this line ###

prediction_funcs_dict = {
    prediction_func["func"].__name__: prediction_func
    for prediction_func in prediction_funcs
}
