export interface SpaceRequest {
  minSize: number
  maxSize?: number
}

/**
 * Distributes available space among items with min/max size constraints
 * @param totalSpace Total space available to distribute
 * @param requests Array of space requests with size constraints
 * @returns Array of space allocations
 */
export function distributeSpace(
  totalSpace: number,
  requests: SpaceRequest[]
): number[] {
  // Handle edge cases
  if (requests.length === 0) return []

  // Calculate total minimum space needed
  const totalMinSize = requests.reduce((sum, req) => sum + req.minSize, 0)

  // If we can't meet minimum requirements, return the minimum sizes
  if (totalSpace < totalMinSize) {
    return requests.map((req) => req.minSize)
  }

  // Initialize allocations with minimum sizes
  let allocations = requests.map((req) => ({
    computedSize: req.minSize,
    maxSize: req.maxSize ?? Infinity,
    remaining: (req.maxSize ?? Infinity) - req.minSize
  }))

  // Calculate remaining space to distribute
  let remainingSpace = totalSpace - totalMinSize

  // Distribute remaining space iteratively
  while (
    remainingSpace > 0 &&
    allocations.some((alloc) => alloc.remaining > 0)
  ) {
    // Count items that can still grow
    const growableItems = allocations.filter(
      (alloc) => alloc.remaining > 0
    ).length

    if (growableItems === 0) break

    // Calculate fair share per item
    const sharePerItem = remainingSpace / growableItems

    // Track how much space was actually used in this iteration
    let spaceUsedThisRound = 0

    // Distribute space
    allocations = allocations.map((alloc) => {
      if (alloc.remaining <= 0) return alloc

      const growth = Math.min(sharePerItem, alloc.remaining)
      spaceUsedThisRound += growth

      return {
        ...alloc,
        computedSize: alloc.computedSize + growth,
        remaining: alloc.remaining - growth
      }
    })

    remainingSpace -= spaceUsedThisRound

    // Break if we couldn't distribute any more space
    if (spaceUsedThisRound === 0) break
  }

  // Return only the computed sizes
  return allocations.map(({ computedSize }) => computedSize)
}
