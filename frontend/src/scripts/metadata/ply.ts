/**
 * PLY (Polygon File Format) decoder
 * Parses ASCII PLY files and extracts vertex positions and colors
 */

interface PLYHeader {
  vertexCount: number
  hasColor: boolean
  propertyIndices: {
    x: number
    y: number
    z: number
    red: number
    green: number
    blue: number
  }
  headerEndLine: number
}

interface PLYData {
  positions: Float32Array
  colors: Float32Array | null
  vertexCount: number
}

function parsePLYHeader(lines: string[]): PLYHeader | null {
  let vertexCount = 0
  let headerEndLine = 0
  let hasColor = false
  let xIndex = -1
  let yIndex = -1
  let zIndex = -1
  let redIndex = -1
  let greenIndex = -1
  let blueIndex = -1
  let propertyIndex = 0

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim()

    if (line.startsWith('element vertex')) {
      vertexCount = parseInt(line.split(/\s+/)[2])
    } else if (line.startsWith('property')) {
      const parts = line.split(/\s+/)
      const propName = parts[parts.length - 1]

      if (propName === 'x') xIndex = propertyIndex
      else if (propName === 'y') yIndex = propertyIndex
      else if (propName === 'z') zIndex = propertyIndex
      else if (propName === 'red') {
        hasColor = true
        redIndex = propertyIndex
      } else if (propName === 'green') greenIndex = propertyIndex
      else if (propName === 'blue') blueIndex = propertyIndex

      propertyIndex++
    } else if (line === 'end_header') {
      headerEndLine = i
      break
    }
  }

  if (vertexCount === 0 || xIndex < 0 || yIndex < 0 || zIndex < 0) {
    return null
  }

  return {
    vertexCount,
    hasColor,
    propertyIndices: {
      x: xIndex,
      y: yIndex,
      z: zIndex,
      red: redIndex,
      green: greenIndex,
      blue: blueIndex
    },
    headerEndLine
  }
}

function parsePLYVertices(lines: string[], header: PLYHeader): PLYData {
  const { vertexCount, hasColor, propertyIndices, headerEndLine } = header
  const { x: xIndex, y: yIndex, z: zIndex } = propertyIndices
  const { red: redIndex, green: greenIndex, blue: blueIndex } = propertyIndices

  const positions = new Float32Array(vertexCount * 3)
  const colors = hasColor ? new Float32Array(vertexCount * 3) : null

  let vertexIndex = 0

  for (
    let i = headerEndLine + 1;
    i < lines.length && vertexIndex < vertexCount;
    i++
  ) {
    const line = lines[i].trim()
    if (!line) continue

    const parts = line.split(/\s+/)
    if (parts.length < 3) continue

    const posIndex = vertexIndex * 3

    positions[posIndex] = parseFloat(parts[xIndex])
    positions[posIndex + 1] = parseFloat(parts[yIndex])
    positions[posIndex + 2] = parseFloat(parts[zIndex])

    if (
      hasColor &&
      colors &&
      redIndex >= 0 &&
      greenIndex >= 0 &&
      blueIndex >= 0
    ) {
      if (parts.length > Math.max(redIndex, greenIndex, blueIndex)) {
        colors[posIndex] = parseInt(parts[redIndex]) / 255
        colors[posIndex + 1] = parseInt(parts[greenIndex]) / 255
        colors[posIndex + 2] = parseInt(parts[blueIndex]) / 255
      }
    }

    vertexIndex++
  }

  return {
    positions,
    colors,
    vertexCount: vertexIndex
  }
}

/**
 * Parse ASCII PLY data from an ArrayBuffer
 * Returns positions and colors as typed arrays
 */
export function parseASCIIPLY(arrayBuffer: ArrayBuffer): PLYData | null {
  const text = new TextDecoder().decode(arrayBuffer)
  const lines = text.split('\n')

  const header = parsePLYHeader(lines)
  if (!header) return null

  return parsePLYVertices(lines, header)
}

/**
 * Check if PLY data is in ASCII format
 */
export function isPLYAsciiFormat(arrayBuffer: ArrayBuffer): boolean {
  const header = new TextDecoder().decode(arrayBuffer.slice(0, 500))
  return header.includes('format ascii')
}
