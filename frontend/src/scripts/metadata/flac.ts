/** @knipIgnoreUnusedButUsedByCustomNodes */
export function getFromFlacBuffer(buffer: ArrayBuffer): Record<string, string> {
  const dataView = new DataView(buffer)

  // Verify the FLAC signature
  const signature = String.fromCharCode(...new Uint8Array(buffer, 0, 4))
  if (signature !== 'fLaC') {
    console.error('Not a valid FLAC file')
    // @ts-expect-error fixme ts strict error
    return
  }

  // Parse metadata blocks
  let offset = 4
  let vorbisComment = null
  while (offset < dataView.byteLength) {
    const isLastBlock = dataView.getUint8(offset) & 0x80
    const blockType = dataView.getUint8(offset) & 0x7f
    const blockSize = dataView.getUint32(offset, false) & 0xffffff
    offset += 4

    if (blockType === 4) {
      // Vorbis Comment block type
      vorbisComment = parseVorbisComment(
        new DataView(buffer, offset, blockSize)
      )
    }

    offset += blockSize
    if (isLastBlock) break
  }

  // @ts-expect-error fixme ts strict error
  return vorbisComment
}

export function getFromFlacFile(file: File): Promise<Record<string, string>> {
  return new Promise((r) => {
    const reader = new FileReader()
    reader.onload = function (event) {
      // @ts-expect-error fixme ts strict error
      const arrayBuffer = event.target.result as ArrayBuffer
      r(getFromFlacBuffer(arrayBuffer))
    }
    reader.readAsArrayBuffer(file)
  })
}

// Function to parse the Vorbis Comment block
function parseVorbisComment(dataView: DataView): Record<string, string> {
  let offset = 0
  const vendorLength = dataView.getUint32(offset, true)
  offset += 4
  // @ts-expect-error unused variable
  const vendorString = getString(dataView, offset, vendorLength)
  offset += vendorLength

  const userCommentListLength = dataView.getUint32(offset, true)
  offset += 4
  const comments = {}
  for (let i = 0; i < userCommentListLength; i++) {
    const commentLength = dataView.getUint32(offset, true)
    offset += 4
    const comment = getString(dataView, offset, commentLength)
    offset += commentLength

    const ind = comment.indexOf('=')
    const key = comment.substring(0, ind)

    // @ts-expect-error fixme ts strict error
    comments[key] = comment.substring(ind + 1)
  }

  return comments
}

function getString(dataView: DataView, offset: number, length: number): string {
  let string = ''
  for (let i = 0; i < length; i++) {
    string += String.fromCharCode(dataView.getUint8(offset + i))
  }
  return string
}
