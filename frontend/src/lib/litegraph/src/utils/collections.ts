import { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { parseSlotTypes } from '@/lib/litegraph/src/strings'

import type { ISlotType, Positionable } from '../interfaces'

/**
 * Creates a flat set of all positionable items by recursively iterating through all child items.
 *
 * Does not include or recurse into pinned items.
 * @param items The original set of items to iterate through
 * @returns All unpinned items in the original set, and recursively, their children
 */
export function getAllNestedItems(
  items: ReadonlySet<Positionable>
): Set<Positionable> {
  const allItems = new Set<Positionable>()
  if (items) {
    for (const item of items) addRecursively(item, allItems)
  }
  return allItems

  function addRecursively(
    item: Positionable,
    flatSet: Set<Positionable>
  ): void {
    if (flatSet.has(item) || item.pinned) return
    flatSet.add(item)
    if (item.children) {
      for (const child of item.children) addRecursively(child, flatSet)
    }
  }
}

/**
 * Iterates through a collection of {@link Positionable} items, returning the first {@link LGraphNode}.
 * @param items The items to search through
 * @returns The first node found in {@link items}, otherwise `undefined`
 */
export function findFirstNode(
  items: Iterable<Positionable>
): LGraphNode | undefined {
  for (const item of items) {
    if (item instanceof LGraphNode) return item
  }
}

type FreeSlotResult<T extends { type: ISlotType }> =
  | { index: number; slot: T }
  | undefined

/**
 * Finds the first free in/out slot with any of the comma-delimited types in {@link type}.
 *
 * If no slots are free, falls back in order to:
 * - The first free wildcard slot
 * - The first occupied slot
 * - The first occupied wildcard slot
 * @param slots The iterable of node slots slots to search through
 * @param type The {@link ISlotType type} of slot to find
 * @param hasNoLinks A predicate that returns `true` if the slot is free.
 * @returns The index and slot if found, otherwise `undefined`.
 */
export function findFreeSlotOfType<T extends { type: ISlotType }>(
  slots: T[],
  type: ISlotType,
  hasNoLinks: (slot: T) => boolean
) {
  if (!slots?.length) return

  let occupiedSlot: FreeSlotResult<T>
  let wildSlot: FreeSlotResult<T>
  let occupiedWildSlot: FreeSlotResult<T>

  const validTypes = parseSlotTypes(type)

  for (const [index, slot] of slots.entries()) {
    const slotTypes = parseSlotTypes(slot.type)

    for (const validType of validTypes) {
      for (const slotType of slotTypes) {
        if (slotType === validType) {
          if (hasNoLinks(slot)) {
            // Exact match - short circuit
            return { index, slot }
          }
          // In case we can't find a free slot.
          occupiedSlot ??= { index, slot }
        } else if (!wildSlot && (validType === '*' || slotType === '*')) {
          // Save the first free wildcard slot as a fallback
          if (hasNoLinks(slot)) {
            wildSlot = { index, slot }
          } else {
            occupiedWildSlot ??= { index, slot }
          }
        }
      }
    }
  }
  return wildSlot ?? occupiedSlot ?? occupiedWildSlot
}
