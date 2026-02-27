import { useNodeDragAndDrop } from '@/composables/node/useNodeDragAndDrop'
import { useNodeFileInput } from '@/composables/node/useNodeFileInput'
import { useNodePaste } from '@/composables/node/useNodePaste'
import { t } from '@/i18n'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { useToastStore } from '@/platform/updates/common/toastStore'
import type { ResultItemType } from '@/schemas/apiSchema'
import { api } from '@/scripts/api'
import { useAssetsStore } from '@/stores/assetsStore'

const PASTED_IMAGE_EXPIRY_MS = 2000

interface ImageUploadFormFields {
  /**
   * The folder to upload the file to.
   * @example 'input', 'output', 'temp'
   */
  type: ResultItemType
}

const uploadFile = async (
  file: File,
  isPasted: boolean,
  formFields: Partial<ImageUploadFormFields> = {}
) => {
  const body = new FormData()
  body.append('image', file)
  if (isPasted) body.append('subfolder', 'pasted')
  if (formFields.type) body.append('type', formFields.type)

  const resp = await api.fetchApi('/upload/image', {
    method: 'POST',
    body
  })

  if (resp.status !== 200) {
    useToastStore().addAlert(resp.status + ' - ' + resp.statusText)
    return
  }

  const data = await resp.json()

  // Update AssetsStore input assets when files are uploaded to input folder
  if (formFields.type === 'input' || (!formFields.type && !isPasted)) {
    const assetsStore = useAssetsStore()
    await assetsStore.updateInputs()
  }

  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name
}

interface ImageUploadOptions {
  fileFilter?: (file: File) => boolean
  onUploadComplete: (paths: string[]) => void
  allow_batch?: boolean
  /**
   * The file types to accept.
   * @example 'image/png,image/jpeg,image/webp,video/webm,video/mp4'
   */
  accept?: string
  /**
   * The folder to upload the file to.
   * @example 'input', 'output', 'temp'
   */
  folder?: ResultItemType
  onUploadStart?: (files: File[]) => void
  onUploadError?: () => void
}

/**
 * Adds image upload to a node via drag & drop, paste, and file input.
 */
export const useNodeImageUpload = (
  node: LGraphNode,
  options: ImageUploadOptions
) => {
  const { fileFilter, onUploadComplete, allow_batch, accept } = options

  const isPastedFile = (file: File): boolean =>
    file.name === 'image.png' &&
    file.lastModified - Date.now() < PASTED_IMAGE_EXPIRY_MS

  const handleUpload = async (file: File) => {
    try {
      const path = await uploadFile(file, isPastedFile(file), {
        type: options.folder
      })
      if (!path) return
      return path
    } catch (error) {
      useToastStore().addAlert(String(error))
    }
  }

  const handleUploadBatch = async (files: File[]) => {
    if (node.isUploading) {
      useToastStore().addAlert(t('g.uploadAlreadyInProgress'))
      return []
    }
    node.isUploading = true

    try {
      node.imgs = undefined
      node.graph?.setDirtyCanvas(true)
      options.onUploadStart?.(files)

      const paths = await Promise.all(files.map(handleUpload))
      const validPaths = paths.filter((p): p is string => !!p)
      if (validPaths.length) {
        onUploadComplete(validPaths)
      } else {
        options.onUploadError?.()
      }
      return validPaths
    } finally {
      node.isUploading = false
      node.graph?.setDirtyCanvas(true)
    }
  }

  // Handle drag & drop
  useNodeDragAndDrop(node, {
    fileFilter,
    onDrop: handleUploadBatch
  })

  // Handle paste
  useNodePaste(node, {
    fileFilter,
    allow_batch,
    onPaste: handleUploadBatch
  })

  // Handle file input
  const { openFileSelection } = useNodeFileInput(node, {
    fileFilter,
    allow_batch,
    accept,
    onSelect: handleUploadBatch
  })

  return { openFileSelection, handleUpload }
}
