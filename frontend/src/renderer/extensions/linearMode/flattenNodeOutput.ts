import type { NodeExecutionOutput, ResultItem } from '@/schemas/apiSchema'
import { ResultItemImpl } from '@/stores/queueStore'

export function flattenNodeOutput([nodeId, nodeOutput]: [
  string | number,
  NodeExecutionOutput
]): ResultItemImpl[] {
  const knownOutputs: Record<string, ResultItem[]> = {}
  if (nodeOutput.audio) knownOutputs.audio = nodeOutput.audio
  if (nodeOutput.images) knownOutputs.images = nodeOutput.images
  if (nodeOutput.video) knownOutputs.video = nodeOutput.video
  if (nodeOutput.gifs) knownOutputs.gifs = nodeOutput.gifs as ResultItem[]
  if (nodeOutput['3d']) knownOutputs['3d'] = nodeOutput['3d'] as ResultItem[]

  return Object.entries(knownOutputs).flatMap(([mediaType, outputs]) =>
    outputs.map(
      (output) => new ResultItemImpl({ ...output, mediaType, nodeId })
    )
  )
}
