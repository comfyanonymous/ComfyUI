import { z } from 'zod'
import { fromZodError } from 'zod-validation-error'

import { resultItemType } from '@/schemas/apiSchema'
import { CONTROL_OPTIONS } from '@/types/simplifiedWidget'

const zComboOption = z.union([z.string(), z.number()])
const zRemoteWidgetConfig = z.object({
  route: z.string().url().or(z.string().startsWith('/')),
  refresh: z.number().gte(128).safe().or(z.number().lte(0).safe()).optional(),
  response_key: z.string().optional(),
  query_params: z.record(z.string(), z.string()).optional(),
  refresh_button: z.boolean().optional(),
  control_after_refresh: z.enum(['first', 'last']).optional(),
  timeout: z.number().gte(0).optional(),
  max_retries: z.number().gte(0).optional()
})
const zMultiSelectOption = z.object({
  placeholder: z.string().optional(),
  chip: z.boolean().optional()
})

export const zBaseInputOptions = z
  .object({
    default: z.any().optional(),
    defaultInput: z.boolean().optional(),
    display_name: z.string().optional(),
    forceInput: z.boolean().optional(),
    tooltip: z.string().optional(),
    socketless: z.boolean().optional(),
    hidden: z.boolean().optional(),
    advanced: z.boolean().optional(),
    widgetType: z.string().optional(),
    /** Backend-only properties. */
    rawLink: z.boolean().optional(),
    lazy: z.boolean().optional()
  })
  .passthrough()

const zNumericInputOptions = zBaseInputOptions.extend({
  min: z.number().optional(),
  max: z.number().optional(),
  step: z.number().optional(),
  /** Note: Many node authors are using INT/FLOAT to pass list of INT/FLOAT. */
  default: z.union([z.number(), z.array(z.number())]).optional(),
  display: z.enum(['slider', 'number', 'knob', 'gradientslider']).optional()
})

export const zIntInputOptions = zNumericInputOptions.extend({
  /**
   * If true, a linked widget will be added to the node to select the mode
   * of `control_after_generate`.
   */
  control_after_generate: z
    .union([z.boolean(), z.enum(CONTROL_OPTIONS)])
    .optional()
})

export const zFloatInputOptions = zNumericInputOptions.extend({
  round: z.union([z.number(), z.literal(false)]).optional(),
  gradient_stops: z
    .array(
      z.object({
        offset: z.number(),
        color: z.tuple([z.number(), z.number(), z.number()])
      })
    )
    .optional()
})

export const zBooleanInputOptions = zBaseInputOptions.extend({
  label_on: z.string().optional(),
  label_off: z.string().optional(),
  default: z.boolean().optional()
})

export const zStringInputOptions = zBaseInputOptions.extend({
  default: z.string().optional(),
  multiline: z.boolean().optional(),
  dynamicPrompts: z.boolean().optional(),

  // Multiline-only fields
  defaultVal: z.string().optional(),
  placeholder: z.string().optional()
})

export const zComboInputOptions = zBaseInputOptions.extend({
  control_after_generate: z
    .union([z.boolean(), z.enum(CONTROL_OPTIONS)])
    .optional(),
  image_upload: z.boolean().optional(),
  image_folder: resultItemType.optional(),
  allow_batch: z.boolean().optional(),
  video_upload: z.boolean().optional(),
  audio_upload: z.boolean().optional(),
  mesh_upload: z.boolean().optional(),
  upload_subfolder: z.string().optional(),
  animated_image_upload: z.boolean().optional(),
  options: z.array(zComboOption).optional(),
  remote: zRemoteWidgetConfig.optional(),
  /** Whether the widget is a multi-select widget. */
  multi_select: zMultiSelectOption.optional()
})

const zIntInputSpec = z.tuple([z.literal('INT'), zIntInputOptions.optional()])
const zFloatInputSpec = z.tuple([
  z.literal('FLOAT'),
  zFloatInputOptions.optional()
])
const zBooleanInputSpec = z.tuple([
  z.literal('BOOLEAN'),
  zBooleanInputOptions.optional()
])
const zStringInputSpec = z.tuple([
  z.literal('STRING'),
  zStringInputOptions.optional()
])
/**
 * Legacy combo syntax.
 * @deprecated Use `zComboInputSpecV2` instead.
 */
const zComboInputSpec = z.tuple([
  z.array(zComboOption),
  zComboInputOptions.optional()
])
const zComboInputSpecV2 = z.tuple([
  z.literal('COMBO'),
  zComboInputOptions.optional()
])

export function isComboInputSpecV1(
  inputSpec: InputSpec
): inputSpec is ComboInputSpec {
  return Array.isArray(inputSpec[0])
}

export function isIntInputSpec(
  inputSpec: InputSpec
): inputSpec is IntInputSpec {
  return inputSpec[0] === 'INT'
}

export function isFloatInputSpec(
  inputSpec: InputSpec
): inputSpec is FloatInputSpec {
  return inputSpec[0] === 'FLOAT'
}

export function isComboInputSpecV2(
  inputSpec: InputSpec
): inputSpec is ComboInputSpecV2 {
  return inputSpec[0] === 'COMBO'
}

export function isComboInputSpec(
  inputSpec: InputSpec
): inputSpec is ComboInputSpec | ComboInputSpecV2 {
  return isComboInputSpecV1(inputSpec) || isComboInputSpecV2(inputSpec)
}

export function isMediaUploadComboInput(inputSpec: InputSpec): boolean {
  const [inputName, inputOptions] = inputSpec
  if (!inputOptions) return false

  const isUploadInput =
    inputOptions['image_upload'] === true ||
    inputOptions['video_upload'] === true ||
    inputOptions['animated_image_upload'] === true

  return (
    isUploadInput && (isComboInputSpecV1(inputSpec) || inputName === 'COMBO')
  )
}

/**
 * Get the type of an input spec.
 *
 * @param inputSpec - The input spec to get the type of.
 * @returns The type of the input spec.
 */
export function getInputSpecType(inputSpec: InputSpec): string {
  return isComboInputSpec(inputSpec) ? 'COMBO' : inputSpec[0]
}

/**
 * Get the combo options from a combo input spec.
 *
 * @param inputSpec - The input spec to get the combo options from.
 * @returns The combo options.
 */
export function getComboSpecComboOptions(
  inputSpec: ComboInputSpec | ComboInputSpecV2
): (number | string)[] {
  return (
    (isComboInputSpecV2(inputSpec) ? inputSpec[1]?.options : inputSpec[0]) ?? []
  )
}

const excludedLiterals = new Set(['INT', 'FLOAT', 'BOOLEAN', 'STRING', 'COMBO'])
const zCustomInputSpec = z.tuple([
  z.string().refine((value) => !excludedLiterals.has(value)),
  zBaseInputOptions.optional()
])

const zInputSpec = z.union([
  zIntInputSpec,
  zFloatInputSpec,
  zBooleanInputSpec,
  zStringInputSpec,
  zComboInputSpec,
  zComboInputSpecV2,
  zCustomInputSpec
])

const zComfyInputsSpec = z.object({
  required: z.record(zInputSpec).optional(),
  optional: z.record(zInputSpec).optional(),
  // Frontend repo is not using it, but some custom nodes are using the
  // hidden field to pass various values.
  hidden: z.record(z.any()).optional()
})

const zComfyNodeDataType = z.string()
const zComfyComboOutput = z.array(zComboOption)
const zComfyOutputTypesSpec = z.array(
  z.union([zComfyNodeDataType, zComfyComboOutput])
)

/**
 * Widget dependency with type information.
 * Provides strong type enforcement for JSONata evaluation context.
 */
const zWidgetDependency = z.object({
  name: z.string(),
  type: z.string()
})

export type WidgetDependency = z.infer<typeof zWidgetDependency>

/**
 * Schema for price badge depends_on field.
 * Specifies which widgets and inputs the pricing expression depends on.
 * Widgets must be specified as objects with name and type.
 */
const zPriceBadgeDepends = z.object({
  widgets: z.array(zWidgetDependency).optional().default([]),
  inputs: z.array(z.string()).optional().default([]),
  /**
   * Autogrow input group names to track.
   * For each group, the count of connected inputs will be available in the
   * JSONata context as `g.<groupName>`.
   * Example: `input_groups: ["reference_videos"]` makes `g.reference_videos`
   * available with the count of connected inputs like `reference_videos.character1`, etc.
   */
  input_groups: z.array(z.string()).optional().default([])
})

/**
 * Schema for price badge definition.
 * Used to calculate and display pricing information for API nodes.
 * The `expr` field contains a JSONata expression that returns a PricingResult.
 */
const zPriceBadge = z.object({
  engine: z.literal('jsonata').optional().default('jsonata'),
  depends_on: zPriceBadgeDepends
    .optional()
    .default({ widgets: [], inputs: [], input_groups: [] }),
  expr: z.string()
})

export type PriceBadge = z.infer<typeof zPriceBadge>

export const zComfyNodeDef = z.object({
  input: zComfyInputsSpec.optional(),
  output: zComfyOutputTypesSpec.optional(),
  output_is_list: z.array(z.boolean()).optional(),
  output_name: z.array(z.string()).optional(),
  output_tooltips: z.array(z.string()).optional(),
  output_matchtypes: z.array(z.string().optional()).optional(),
  name: z.string(),
  display_name: z.string(),
  description: z.string(),
  help: z.string().optional(),
  category: z.string(),
  main_category: z.string().optional(),
  output_node: z.boolean(),
  python_module: z.string(),
  deprecated: z.boolean().optional(),
  experimental: z.boolean().optional(),
  dev_only: z.boolean().optional(),
  /**
   * Whether the node is an API node. Running API nodes requires login to
   * Comfy Org account.
   * https://docs.comfy.org/tutorials/api-nodes/overview
   */
  api_node: z.boolean().optional(),
  /**
   * Specifies the order of inputs for each input category.
   * Used to ensure consistent widget ordering regardless of JSON serialization.
   * Keys are 'required', 'optional', etc., values are arrays of input names.
   */
  input_order: z.record(z.array(z.string())).optional(),
  /**
   * Alternative names for search. Useful for synonyms, abbreviations,
   * or old names after renaming a node.
   */
  search_aliases: z.array(z.string()).optional(),
  /**
   * Price badge definition for API nodes.
   * Contains a JSONata expression to calculate pricing based on widget values
   * and input connectivity.
   */
  price_badge: zPriceBadge.optional(),
  /** Category for the Essentials tab. If set, the node appears in Essentials. */
  essentials_category: z.string().optional(),
  /** Whether the blueprint is a global/installed blueprint (not user-created). */
  isGlobal: z.boolean().optional()
})

export const zAutogrowOptions = z.object({
  ...zBaseInputOptions.shape,
  template: z.object({
    input: zComfyInputsSpec,
    names: z.array(z.string()).optional(),
    max: z.number().optional(),
    //Backend defines as mandatory with min 1, Frontend is more forgiving
    min: z.number().optional(),
    prefix: z.string().optional()
  })
})

export const zDynamicComboInputSpec = z.tuple([
  z.literal('COMFY_DYNAMICCOMBO_V3'),
  zBaseInputOptions.extend({
    options: z.array(
      z.object({
        inputs: zComfyInputsSpec,
        key: z.string()
      })
    )
  })
])

// `/object_info`
export type ComfyInputsSpec = z.infer<typeof zComfyInputsSpec>
export type ComfyOutputTypesSpec = z.infer<typeof zComfyOutputTypesSpec>
export type ComfyNodeDef = z.infer<typeof zComfyNodeDef>
export type RemoteWidgetConfig = z.infer<typeof zRemoteWidgetConfig>

export type ComboInputOptions = z.infer<typeof zComboInputOptions>
export type NumericInputOptions = z.infer<typeof zNumericInputOptions>

export type IntInputSpec = z.infer<typeof zIntInputSpec>
export type FloatInputSpec = z.infer<typeof zFloatInputSpec>
export type ComboInputSpec = z.infer<typeof zComboInputSpec>
export type ComboInputSpecV2 = z.infer<typeof zComboInputSpecV2>
export type InputSpec = z.infer<typeof zInputSpec>

export function validateComfyNodeDef(
  data: unknown,
  onError: (error: string) => void = console.warn
): ComfyNodeDef | null {
  const result = zComfyNodeDef.safeParse(data)
  if (!result.success) {
    const zodError = fromZodError(result.error)
    onError(
      `Invalid ComfyNodeDef: ${JSON.stringify(data)}\n${zodError.message}`
    )
    return null
  }
  return result.data
}
