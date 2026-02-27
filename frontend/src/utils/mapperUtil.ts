import { transformInputSpecV1ToV2 } from '@/schemas/nodeDef/migration'
import type {
  ComfyNodeDef as ComfyNodeDefV2,
  InputSpec
} from '@/schemas/nodeDef/nodeDefSchemaV2'
import type { ComfyNodeDef as ComfyNodeDefV1 } from '@/schemas/nodeDefSchema'
import type { components } from '@/types/comfyRegistryTypes'

const registryToFrontendV2NodeOutputs = (
  registryDef: components['schemas']['ComfyNode']
): ComfyNodeDefV2['outputs'] => {
  const returnTypes = JSON.parse(registryDef.return_types ?? '{}')
  if (!returnTypes.length) return []

  const returnNames = JSON.parse(registryDef.return_names ?? '{}')
  const outputsIsList = registryDef.output_is_list ?? []

  const outputs = []
  for (let i = 0; i < returnTypes.length; i++) {
    outputs.push({
      type: returnTypes[i],
      name: returnNames[i] || returnTypes[i],
      is_list: outputsIsList[i] ?? false,
      index: i
    })
  }

  return outputs
}

const registryToFrontendV2NodeInputs = (
  registryDef: components['schemas']['ComfyNode']
): ComfyNodeDefV2['inputs'] => {
  const inputTypes = JSON.parse(
    registryDef.input_types ?? '{}'
  ) as ComfyNodeDefV1['input']
  if (!inputTypes || !Object.keys(inputTypes).length) return {}

  const inputsV2: Record<string, InputSpec> = {}

  if (inputTypes.required) {
    Object.entries(inputTypes.required).forEach(([name, inputSpecV1]) => {
      inputsV2[name] = transformInputSpecV1ToV2(inputSpecV1, {
        name,
        isOptional: false
      })
    })
  }

  if (inputTypes.optional) {
    Object.entries(inputTypes.optional).forEach(([name, inputSpecV1]) => {
      inputsV2[name] = transformInputSpecV1ToV2(inputSpecV1, {
        name,
        isOptional: true
      })
    })
  }

  return inputsV2
}

export const registryToFrontendV2NodeDef = (
  nodeDef: components['schemas']['ComfyNode'],
  nodePack: components['schemas']['Node']
): ComfyNodeDefV2 => {
  const name = nodeDef.comfy_node_name ?? 'Node Name'
  return {
    category: nodeDef.category ?? 'unknown',
    description: nodeDef.description ?? '',
    display_name: name,
    name,
    inputs: registryToFrontendV2NodeInputs(nodeDef),
    outputs: registryToFrontendV2NodeOutputs(nodeDef),
    output_node: false,
    python_module: nodePack.name ?? nodePack.id ?? 'unknown'
  }
}
