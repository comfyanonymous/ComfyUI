from comfy_execution.graph_utils import is_link


def _linked_children(dynprompt):
    children = {}
    for candidate_id in dynprompt.all_node_ids():
        node = dynprompt.get_node(candidate_id)
        for value in node.get("inputs", {}).values():
            if is_link(value):
                children.setdefault(value[0], set()).add(candidate_id)
    return children


def _walk_graph(start_ids, next_ids):
    found = set()
    stack = list(start_ids)
    while stack:
        node_id = stack.pop()
        if node_id in found:
            continue
        found.add(node_id)
        stack.extend(next_ids(node_id))
    return found


def descendants(dynprompt, node_id):
    children = _linked_children(dynprompt)
    return _walk_graph(children.get(node_id, ()), lambda child_id: children.get(child_id, ()))


def ascendants(dynprompt, node_id, stop_at=None):
    stop_at = stop_at or (lambda _node_id: False)

    def parent_ids(candidate_id):
        node = dynprompt.get_node(candidate_id)
        parents = []
        for value in node.get("inputs", {}).values():
            if not is_link(value):
                continue
            parent_id = value[0]
            if not stop_at(parent_id):
                parents.append(parent_id)
        return parents

    return _walk_graph(parent_ids(node_id), parent_ids)
