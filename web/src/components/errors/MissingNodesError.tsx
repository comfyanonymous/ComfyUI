interface MissingNodesType {
    type: string;
    hint?: string;
    action?: {
        text: string;
        callback: () => void
    }
}

interface MissingNodesProps {
    hasAddedNodes?: boolean;
    nodeTypes: Set<string | MissingNodesType> | string[];
}

export function MissingNodesError({nodeTypes, hasAddedNodes = false}: MissingNodesProps) {
    const seenTypes = new Set();

    return (
        <div className="comfy-missing-nodes">
            <span>When loading the graph, the following node types were not found:</span>
            <ul>
                {Array.from(new Set(nodeTypes)).map(t => {
                    if (typeof t === 'object') {
                        if (seenTypes.has(t.type)) {
                            return null
                        }

                        seenTypes.add(t.type);
                        return (
                            <li key={t.type}>
                                <span>{t.type}</span>
                                {t.hint && <span>{t.hint}</span>}
                                {t.action && <button onClick={t.action.callback}>{t.action.text}</button>}
                            </li>
                        );
                    } else {
                        if (seenTypes.has(t)) return null;
                        seenTypes.add(t);
                        return (
                            <li key={t}>
                                <span>{t}</span>
                            </li>
                        );
                    }
                })}
            </ul>
            {
                hasAddedNodes
                && (<span>Nodes that have failed to load will show as red on the graph.</span>)
            }
        </div>
    )
}

