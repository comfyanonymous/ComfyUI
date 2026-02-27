export function getSlotColor(type?: string | number | null): string {
  if (!type) return '#AAA'
  const typeStr = String(type).toUpperCase()
  return `var(--color-datatype-${typeStr}, #AAA)`
}
