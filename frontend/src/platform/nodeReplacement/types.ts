interface InputMapOldId {
  new_id: string
  old_id: string
}

interface InputMapSetValue {
  new_id: string
  set_value: unknown
}

type InputMap = InputMapOldId | InputMapSetValue

interface OutputMap {
  new_idx: number
  old_idx: number
}

export interface NodeReplacement {
  new_node_id: string
  old_node_id: string
  old_widget_ids: string[] | null
  input_mapping: InputMap[] | null
  output_mapping: OutputMap[] | null
}

export type NodeReplacementResponse = Record<string, NodeReplacement[]>
