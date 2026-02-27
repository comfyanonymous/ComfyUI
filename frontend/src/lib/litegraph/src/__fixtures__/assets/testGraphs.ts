import type {
  ISerialisedGraph,
  ISerialisedNode,
  SerialisableGraph
} from '@/lib/litegraph/src/litegraph'

export const oldSchemaGraph: ISerialisedGraph = {
  id: 'b4e984f1-b421-4d24-b8b4-ff895793af13',
  revision: 0,
  version: 0.4,
  config: {},
  last_node_id: 0,
  last_link_id: 0,
  groups: [
    {
      id: 123,
      bounding: [20, 20, 1, 3],
      color: '#6029aa',
      font_size: 14,
      title: 'A group to test with'
    }
  ],
  nodes: [{ id: 1 } as Partial<ISerialisedNode> as ISerialisedNode],
  links: []
}

export const minimalSerialisableGraph: SerialisableGraph = {
  id: 'd175890f-716a-4ece-ba33-1d17a513b7be',
  revision: 0,
  version: 1,
  config: {},
  state: {
    lastNodeId: 0,
    lastLinkId: 0,
    lastGroupId: 0,
    lastRerouteId: 0
  },
  nodes: [],
  links: [],
  groups: []
}

export const basicSerialisableGraph: SerialisableGraph = {
  id: 'ca9da7d8-fddd-4707-ad32-67be9be13140',
  revision: 0,
  version: 1,
  config: {},
  state: {
    lastNodeId: 0,
    lastLinkId: 0,
    lastGroupId: 0,
    lastRerouteId: 0
  },
  groups: [
    {
      id: 123,
      bounding: [20, 20, 1, 3],
      color: '#6029aa',
      font_size: 14,
      title: 'A group to test with'
    }
  ],
  nodes: [
    { id: 1, type: 'mustBeSet' } as Partial<ISerialisedNode> as ISerialisedNode
  ],
  links: []
}
