import type { SubgraphInput } from '@/lib/litegraph/src/subgraph/SubgraphInput'
import type { SubgraphNode } from '@/lib/litegraph/src/subgraph/SubgraphNode'
import type { SubgraphOutput } from '@/lib/litegraph/src/subgraph/SubgraphOutput'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'

import type { LGraphEventMap } from './LGraphEventMap'

export interface SubgraphEventMap extends LGraphEventMap {
  'adding-input': {
    name: string
    type: string
  }
  'adding-output': {
    name: string
    type: string
  }

  'input-added': {
    input: SubgraphInput
  }
  'output-added': {
    output: SubgraphOutput
  }

  'removing-input': {
    input: SubgraphInput
    index: number
  }
  'removing-output': {
    output: SubgraphOutput
    index: number
  }

  'renaming-input': {
    input: SubgraphInput
    index: number
    oldName: string
    newName: string
  }
  'renaming-output': {
    output: SubgraphOutput
    index: number
    oldName: string
    newName: string
  }

  'widget-promoted': {
    widget: IBaseWidget
    subgraphNode: SubgraphNode
  }
  'widget-demoted': {
    widget: IBaseWidget
    subgraphNode: SubgraphNode
  }
}
