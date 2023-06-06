function isActiveNode(node) {
	if (node.mode !== LiteGraph.ALWAYS) {
		return false;
	}

	return true;
}

function getInnerGraphOutputByIndex(subgraph, outerOutputIndex) {
	const outputSlot = subgraph.getOutputInfo(outerOutputIndex)
	if (!outputSlot)
		return null;

	const graphOutput = subgraph.subgraph._nodes.find(n => {
		return n.type === "graph/output"
			&& n.properties.name === outputSlot.name
	})

	return graphOutput || null;
}

function followSubgraph(subgraph, link) {
    if (link.origin_id != subgraph.id)
        throw new Error("Invalid link and graph output!")

    const innerGraphOutput = getInnerGraphOutputByIndex(subgraph, link.origin_slot)
    if (innerGraphOutput == null)
        throw new Error("No inner graph input!")

    const nextLink = innerGraphOutput.getInputLink(0)
    return [innerGraphOutput.graph, nextLink];
}

function followGraphInput(graphInput, link) {
    if (link.origin_id != graphInput.id)
        throw new Error("Invalid link and graph input!")

    const outerSubgraph = graphInput.graph._subgraph_node
    if (outerSubgraph == null)
        throw new Error("No outer subgraph!")

    const outerInputIndex = outerSubgraph.inputs.findIndex(i => i.name === graphInput.name_in_graph)
    if (outerInputIndex === -1)
        throw new Error("No outer input slot!")

    const nextLink = outerSubgraph.getInputLink(outerInputIndex)
    return [outerSubgraph.graph, nextLink];
}

export function getUpstreamLink(parent, currentLink) {
    if (parent.type === "graph/subgraph") {
        console.debug("FollowSubgraph")
        return followSubgraph(parent, currentLink);
    }
    else if (parent.type === "graph/input") {
        console.debug("FollowGraphInput")
        return followGraphInput(parent, currentLink);
    }
    else if ("getUpstreamLink" in parent) {
        const link = parent.getUpstreamLink();
        return [parent.graph, link];
    }
    else if (parent.inputs && parent.inputs.length === 1) {
        // Only one input, so assume we can follow it backwards.
        const link = parent.getInputLink(0);
        if (link) {
            return [parent.graph, link]
        }
    }
    console.warn("[graphToPrompt] Node does not support getUpstreamLink", parent.type)
    return [null, null];
}

export function locateUpstreamNode(isTheTargetNodeCb, fromNode, inputIndex) {
	let parent = fromNode.getInputNode(inputIndex);
	if (!parent)
		return [null, null];

	const seen = {}
	let currentLink = fromNode.getInputLink(inputIndex);

	const shouldFollowParent = (parent, currentLink) => {
		return isActiveNode(parent) && !isTheTargetNodeCb(parent, currentLink);
	}

	// If there are non-target nodes between us and another
	// target node, we have to traverse them first. This
	// behavior is dependent on the type of node. Reroute nodes
	// will simply follow their single input, while branching
	// nodes have conditional logic that determines which link
	// to follow backwards.
	while (shouldFollowParent(parent, currentLink)) {
		const [nextGraph, nextLink] = getUpstreamLink(parent, currentLink);

		if (nextLink == null) {
			console.warn("[graphToPrompt] No upstream link found in frontend node", parent)
			break;
		}

		if (nextLink && !seen[nextLink.id]) {
			seen[nextLink.id] = true
			const nextParent = nextGraph.getNodeById(nextLink.origin_id);
			if (!isActiveNode(parent)) {
				parent = null;
			}
			else {
				console.debug("[graphToPrompt] Traverse upstream link", parent.id, nextParent?.id, (nextParent)?.comfyClass)
				currentLink = nextLink;
				parent = nextParent;
			}
		} else {
			parent = null;
		}
	}

	if (!isActiveNode(parent) || !isTheTargetNodeCb(parent, currentLink) || currentLink == null)
		return [null, currentLink];

	return [parent, currentLink]
}

export function promptToGraphVis(prompt) {
    let out = "digraph {\n"

    const ids = {}
    let nextID = 0;

    for (const pair of Object.entries(prompt.output)) {
        const [id, o] = pair;
        if (ids[id] == null)
            ids[id] = nextID++;

        if ("class_type" in o) {
            for (const pair2 of Object.entries(o.inputs)) {
                const [inpName, i] = pair2;

                if (Array.isArray(i) && i.length === 2 && typeof i[0] === "string" && typeof i[1] === "number") {
                    // Link
                    const [inpID, inpSlot] = i;
                    if (ids[inpID] == null)
                        ids[inpID] = nextID++;

                    const inpNode = prompt.output[inpID]
                    if (inpNode) {
                        out += `"${ids[inpID]}_${inpNode.class_type}" -> "${ids[id]}_${o.class_type}"\n`
                    }
                }
                else {
                    const value = String(i).substring(0, 20)
                    // Value
                    out += `"${ids[id]}-${inpName}-${value}" -> "${ids[id]}_${o.class_type}"\n`
                }
            }
        }
    }

    out += "}"
    return out
}
