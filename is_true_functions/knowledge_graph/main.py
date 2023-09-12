from is_true_functions.knowledge_graph.utils.GraphGPT import GraphGPT
from is_true_functions.knowledge_graph.utils.compare import find_closest_edges


def is_true_knowledge_graph(question: str, answer: str, context: str):
    source_graph = GraphGPT()
    source_graph.create_graph(context)

    answer_graph = GraphGPT()
    answer_graph.create_graph(question + " " + answer)

    similarities = []
    for answer_edge in answer_graph.graph_state["edges"]:
        closest_edges = find_closest_edges(
            source_graph.graph_state["edges"],
            answer_edge,
            1,
        )
        closest_edge = closest_edges[0]
        similarity = closest_edge[1]
        similarities.append(similarity)

    return min(similarities) if similarities else 0
