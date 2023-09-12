import json
from is_true_functions.knowledge_graph.utils.prompt import get_response

DEFAULT_PARAMS = {
    "model": "gpt-3.5-turbo-16k",
    "temperature": 0.3,
    "max_tokens": 800,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


class GraphGPT:
    def __init__(self):
        self.graph_state = {"nodes": [], "edges": []}

    def clear_state(self):
        self.graph_state = {"nodes": [], "edges": []}

    def update_graph(self, updates):
        current_graph = self.graph_state.copy()

        if not updates:
            return

        if isinstance(updates[0], str):
            updates = [updates]

        for update in updates:
            if len(update) == 3:
                entity1, relation, entity2 = update
                node1 = next(
                    (node for node in current_graph["nodes"] if node == entity1), None
                )
                node2 = next(
                    (node for node in current_graph["nodes"] if node == entity2), None
                )

                if node1 is None:
                    current_graph["nodes"].append(entity1)

                if node2 is None:
                    current_graph["nodes"].append(entity2)

                edge = next(
                    (
                        edge
                        for edge in current_graph["edges"]
                        if edge["from"] == entity1 and edge["to"] == entity2
                    ),
                    None,
                )
                if edge is not None:
                    edge["relation"] = relation
                else:
                    current_graph["edges"].append(
                        {"from": entity1, "relation": relation, "to": entity2}
                    )

        self.graph_state = current_graph

    def query_prompt(self, prompt):
        response = get_response(prompt)
        text = response.choices[0].message.content
        try:
            updates = json.loads(text)
        except json.decoder.JSONDecodeError:
            text = text.replace("...", "")
            try:
                updates = json.loads(text)
            except json.decoder.JSONDecodeError:
                updates = None

        self.update_graph(updates)

    def create_graph(self, prompt):
        self.query_prompt(prompt)
