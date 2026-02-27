import type { LGraphNode } from '@/lib/litegraph/src/litegraph'

type DragHandler = (e: DragEvent) => boolean
type DropHandler<T> = (files: File[]) => Promise<T[]>

interface DragAndDropOptions<T> {
  onDragOver?: DragHandler
  onDrop: DropHandler<T>
  fileFilter?: (file: File) => boolean
}

/**
 * Adds drag and drop file handling to a node
 */
export const useNodeDragAndDrop = <T>(
  node: LGraphNode,
  options: DragAndDropOptions<T>
) => {
  const { onDragOver, onDrop, fileFilter = () => true } = options

  const hasFiles = (items: DataTransferItemList) =>
    !!Array.from(items).find((f) => f.kind === 'file')

  const filterFiles = (files: FileList) => Array.from(files).filter(fileFilter)

  const hasValidFiles = (files: FileList) => filterFiles(files).length > 0

  const isDraggingFiles = (e: DragEvent | undefined) => {
    if (!e?.dataTransfer?.items) return false
    return onDragOver?.(e) ?? hasFiles(e.dataTransfer.items)
  }

  const isDraggingValidFiles = (e: DragEvent | undefined) => {
    if (!e?.dataTransfer?.files) return false
    return hasValidFiles(e.dataTransfer.files)
  }

  node.onDragOver = isDraggingFiles

  node.onDragDrop = function (e: DragEvent) {
    if (!isDraggingValidFiles(e)) return false

    const files = filterFiles(e.dataTransfer!.files)
    void onDrop(files)
    return true
  }
}
