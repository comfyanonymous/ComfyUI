import _ from 'es-toolkit/compat'

import type {
  ComboInputSpec,
  ComboInputSpecV2,
  FloatInputSpec,
  InputSpec,
  IntInputSpec,
  NumericInputOptions
} from '@/schemas/nodeDefSchema'
import {
  getComboSpecComboOptions,
  getInputSpecType,
  isComboInputSpec,
  isFloatInputSpec,
  isIntInputSpec
} from '@/schemas/nodeDefSchema'

import { lcm } from './mathUtil'

const IGNORE_KEYS = new Set<string>([
  'default',
  'forceInput',
  'defaultInput',
  'control_after_generate',
  'multiline',
  'tooltip',
  'dynamicPrompts'
])

const getRange = (options: NumericInputOptions) => {
  const min = options.min ?? -Infinity
  const max = options.max ?? Infinity
  return { min, max }
}

const mergeNumericInputSpec = <T extends IntInputSpec | FloatInputSpec>(
  spec1: T,
  spec2: T
): T | null => {
  const type = spec1[0]
  const options1 = spec1[1] ?? {}
  const options2 = spec2[1] ?? {}

  const range1 = getRange(options1)
  const range2 = getRange(options2)

  // If the ranges do not overlap, return null
  if (range1.min > range2.max || range1.max < range2.min) {
    return null
  }

  const step1 = options1.step ?? 1
  const step2 = options2.step ?? 1

  const mergedOptions = {
    // Take intersection of ranges
    min: Math.max(range1.min, range2.min),
    max: Math.min(range1.max, range2.max),
    step: lcm(step1, step2)
  }

  return mergeCommonInputSpec(
    [type, { ...options1, ...mergedOptions }] as T,
    [type, { ...options2, ...mergedOptions }] as T
  )
}

const mergeComboInputSpec = <T extends ComboInputSpec | ComboInputSpecV2>(
  spec1: T,
  spec2: T
): T | null => {
  const options1 = spec1[1] ?? {}
  const options2 = spec2[1] ?? {}

  const comboOptions1 = getComboSpecComboOptions(spec1)
  const comboOptions2 = getComboSpecComboOptions(spec2)

  const intersection = _.intersection(comboOptions1, comboOptions2)

  // If the intersection is empty, return null
  if (intersection.length === 0) {
    return null
  }

  return mergeCommonInputSpec(
    ['COMBO', { ...options1, options: intersection }] as T,
    ['COMBO', { ...options2, options: intersection }] as T
  )
}

const mergeCommonInputSpec = <T extends InputSpec>(
  spec1: T,
  spec2: T
): T | null => {
  const type = getInputSpecType(spec1)
  const options1 = spec1[1] ?? {}
  const options2 = spec2[1] ?? {}

  const compareKeys = _.union(_.keys(options1), _.keys(options2)).filter(
    (key) => !IGNORE_KEYS.has(key)
  )

  const mergeIsValid = compareKeys.every((key) => {
    const value1 = options1[key]
    const value2 = options2[key]
    return value1 === value2 || (_.isNil(value1) && _.isNil(value2))
  })

  return mergeIsValid ? ([type, { ...options1, ...options2 }] as T) : null
}

/**
 * Merges two input specs.
 *
 * @param spec1 - The first input spec.
 * @param spec2 - The second input spec.
 * @returns The merged input spec, or null if the specs are not mergeable.
 */
export const mergeInputSpec = (
  spec1: InputSpec,
  spec2: InputSpec
): InputSpec | null => {
  const type1 = getInputSpecType(spec1)
  const type2 = getInputSpecType(spec2)

  if (type1 !== type2) {
    return null
  }

  if (isIntInputSpec(spec1) || isFloatInputSpec(spec1)) {
    return mergeNumericInputSpec(spec1, spec2 as typeof spec1)
  }

  if (isComboInputSpec(spec1)) {
    return mergeComboInputSpec(spec1, spec2 as typeof spec1)
  }

  return mergeCommonInputSpec(spec1, spec2)
}
