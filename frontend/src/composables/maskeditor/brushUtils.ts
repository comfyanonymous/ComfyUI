/**
 * Calculates the effective brush size based on the base size and hardness.
 * As hardness decreases, the effective size increases to allow for a softer falloff.
 *
 * @param size - The base radius of the brush
 * @param hardness - The hardness of the brush (0.0 to 1.0)
 * @returns The effective radius of the brush
 */
export function getEffectiveBrushSize(size: number, hardness: number): number {
  // Scale factor for maximum softness
  const MAX_SCALE = 1.5
  const scale = 1.0 + (1.0 - hardness) * (MAX_SCALE - 1.0)
  return size * scale
}

/**
 * Calculates the effective hardness to maintain the visual "hard core" of the brush.
 * Since the effective size is larger, we need to adjust the hardness value so that
 * the inner hard circle remains at the same physical radius as the original size * hardness.
 *
 * @param size - The base radius of the brush
 * @param hardness - The base hardness of the brush
 * @param effectiveSize - The effective radius (calculated by getEffectiveBrushSize)
 * @returns The adjusted hardness value (0.0 to 1.0)
 */
export function getEffectiveHardness(
  size: number,
  hardness: number,
  effectiveSize: number
): number {
  if (effectiveSize <= 0) return 0
  // Adjust hardness to maintain the physical radius of the hard core
  return (size * hardness) / effectiveSize
}
