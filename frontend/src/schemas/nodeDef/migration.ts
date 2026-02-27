import type {
  ComfyNodeDef as ComfyNodeDefV2,
  InputSpec as InputSpecV2,
  OutputSpec as OutputSpecV2
} from '@/schemas/nodeDef/nodeDefSchemaV2'
import type {
  ComfyNodeDef as ComfyNodeDefV1,
  InputSpec as InputSpecV1
} from '@/schemas/nodeDefSchema'
import {
  getComboSpecComboOptions,
  isComboInputSpec,
  isComboInputSpecV1
} from '@/schemas/nodeDefSchema'

/**
 * Transforms a V1 node definition to V2 format
 * @param nodeDefV1 The V1 node definition to transform
 * @returns The transformed V2 node definition
 */
export function transformNodeDefV1ToV2(
  nodeDefV1: ComfyNodeDefV1
): ComfyNodeDefV2 {
  // Transform inputs
  const inputs: Record<string, InputSpecV2> = {}

  // Process required inputs
  if (nodeDefV1.input?.required) {
    Object.entries(nodeDefV1.input.required).forEach(([name, inputSpecV1]) => {
      inputs[name] = transformInputSpecV1ToV2(inputSpecV1, {
        name,
        isOptional: false
      })
    })
  }

  // Process optional inputs
  if (nodeDefV1.input?.optional) {
    Object.entries(nodeDefV1.input.optional).forEach(([name, inputSpecV1]) => {
      inputs[name] = transformInputSpecV1ToV2(inputSpecV1, {
        name,
        isOptional: true
      })
    })
  }

  // Transform outputs
  const outputs: OutputSpecV2[] = []

  if (nodeDefV1.output) {
    if (Array.isArray(nodeDefV1.output)) {
      nodeDefV1.output.forEach((outputType, index) => {
        const outputSpec: OutputSpecV2 = {
          index,
          name: nodeDefV1.output_name?.[index] || `output_${index}`,
          type: Array.isArray(outputType) ? 'COMBO' : outputType,
          is_list: nodeDefV1.output_is_list?.[index] || false,
          tooltip: nodeDefV1.output_tooltips?.[index]
        }

        // Add options for combo outputs
        if (Array.isArray(outputType)) {
          outputSpec.options = outputType
        }

        outputs.push(outputSpec)
      })
    } else {
      console.warn('nodeDefV1.output is not an array:', nodeDefV1.output)
    }
  }

  // Create the V2 node definition
  return {
    inputs,
    outputs,
    hidden: nodeDefV1.input?.hidden,
    name: nodeDefV1.name,
    display_name: nodeDefV1.display_name,
    description: nodeDefV1.description,
    category: nodeDefV1.category,
    output_node: nodeDefV1.output_node,
    python_module: nodeDefV1.python_module,
    deprecated: nodeDefV1.deprecated,
    experimental: nodeDefV1.experimental
  }
}

/**
 * Transforms a V1 input specification to V2 format
 * @param inputSpecV1 The V1 input specification to transform
 * @param name The name of the input
 * @param isOptional Whether the input is optional
 * @returns The transformed V2 input specification
 */
export function transformInputSpecV1ToV2(
  inputSpecV1: InputSpecV1,
  kwargs: {
    name: string
    isOptional?: boolean
  }
): InputSpecV2 {
  const { name, isOptional = false } = kwargs

  // Extract options from the input spec
  const options = inputSpecV1[1] || {}

  // Base properties for all input types
  const baseProps = {
    name,
    isOptional,
    ...options
  }

  // Handle different input types
  if (isComboInputSpec(inputSpecV1)) {
    return {
      type: 'COMBO',
      ...baseProps,
      options: isComboInputSpecV1(inputSpecV1)
        ? inputSpecV1[0]
        : getComboSpecComboOptions(inputSpecV1)
    }
  } else if (typeof inputSpecV1[0] === 'string') {
    // Handle standard types (INT, FLOAT, BOOLEAN, STRING) and custom types
    return {
      type: inputSpecV1[0],
      ...baseProps
    }
  }

  // Fallback for any unhandled cases
  return {
    type: 'UNKNOWN',
    ...baseProps
  }
}

export function transformInputSpecV2ToV1(
  inputSpecV2: InputSpecV2
): InputSpecV1 {
  return [inputSpecV2.type, inputSpecV2]
}
