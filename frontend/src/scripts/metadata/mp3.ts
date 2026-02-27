export async function getMp3Metadata(file: File) {
  const reader = new FileReader()
  const read_process = new Promise(
    (r) => (reader.onload = (event) => r(event?.target?.result))
  )
  reader.readAsArrayBuffer(file)
  const arrayBuffer = (await read_process) as ArrayBuffer
  //https://stackoverflow.com/questions/7302439/how-can-i-determine-that-a-particular-file-is-in-fact-an-mp3-file#7302482
  const sig_bytes = new Uint8Array(arrayBuffer, 0, 3)
  if (
    (sig_bytes[0] != 0xff && sig_bytes[1] != 0xfb) ||
    (sig_bytes[0] != 0x49 && sig_bytes[1] != 0x44 && sig_bytes[2] != 0x33)
  )
    console.error('Invalid file signature.')
  let header = ''
  while (header.length < arrayBuffer.byteLength) {
    const page = String.fromCharCode(
      ...new Uint8Array(arrayBuffer, header.length, header.length + 4096)
    )
    header += page
    if (page.match('\u00ff\u00fb')) break
  }
  let workflow, prompt
  let prompt_s = header.match(/prompt\u0000(\{.*?\})\u0000/s)?.[1]
  if (prompt_s) prompt = JSON.parse(prompt_s)
  let workflow_s = header.match(/workflow\u0000(\{.*?\})\u0000/s)?.[1]
  if (workflow_s) workflow = JSON.parse(workflow_s)
  return { prompt, workflow }
}
