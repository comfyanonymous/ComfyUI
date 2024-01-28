import { serializedLGraph, SerializedLGraphGroup, SerializedLGraphNode, LGraphNode } from 'litegraph.js';

export type SerializedGraph = serializedLGraph<
    SerializedLGraphNode<LGraphNode>,
    [number, number, number, number, number, string],
    SerializedLGraphGroup
>;
