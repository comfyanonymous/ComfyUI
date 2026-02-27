export function splitTextAtWordBoundary(
  text: string,
  ratio: number
): [string, string] {
  if (ratio >= 1) return [text, '']
  const estimate = Math.floor(text.length * ratio)
  const breakIndex = text.lastIndexOf(' ', estimate)
  if (breakIndex <= 0) return [text, '']
  return [text.substring(0, breakIndex), text.substring(breakIndex + 1)]
}
