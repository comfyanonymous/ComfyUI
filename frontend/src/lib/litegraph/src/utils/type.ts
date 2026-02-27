import { without } from 'es-toolkit'

import type { IColorable, ISlotType } from '@/lib/litegraph/src/interfaces'
import type { NodeBindable } from '@/lib/litegraph/src/types/widgets'

/**
 * Converts a plain object to a class instance if it is not already an instance of the class.
 *
 * Requires specific constructor signature; first parameter must be the object to convert.
 * @param cls The class to convert to
 * @param args The object to convert, followed by any other constructor arguments
 * @returns The class instance
 */
export function toClass<P, C extends P, Args extends unknown[]>(
  cls: new (instance: P, ...args: Args) => C,
  ...args: [P, ...Args]
): C {
  return args[0] instanceof cls ? args[0] : new cls(...args)
}

/**
 * Checks if an object is an instance of {@link IColorable}.
 */
export function isColorable(obj: unknown): obj is IColorable {
  return (
    typeof obj === 'object' &&
    obj !== null &&
    'setColorOption' in obj &&
    'getColorOption' in obj
  )
}

export function isNodeBindable(widget: unknown): widget is NodeBindable {
  return (
    typeof widget === 'object' &&
    widget !== null &&
    'setNodeId' in widget &&
    typeof widget.setNodeId === 'function'
  )
}

export function commonType(...types: ISlotType[]): ISlotType | undefined {
  if (!isStrings(types)) return undefined

  const withoutWildcards = without(types, '*')
  if (withoutWildcards.length === 0) return '*'

  const typeLists: string[][] = withoutWildcards.map((type) => type.split(','))

  const combinedTypes = intersection(...typeLists)
  if (combinedTypes.length === 0) return undefined

  return combinedTypes.join(',')
}

function intersection(...sets: string[][]): string[] {
  const itemCounts: Record<string, number> = {}
  for (const set of sets)
    for (const item of new Set(set))
      itemCounts[item] = (itemCounts[item] ?? 0) + 1
  return Object.entries(itemCounts)
    .filter(([, count]) => count === sets.length)
    .map(([key]) => key)
}

function isStrings(types: unknown[]): types is string[] {
  return types.every((t) => typeof t === 'string')
}
