import { useNodeImage, useNodeVideo } from '@/composables/node/useNodeImage'
import { useNodeImageUpload } from '@/composables/node/useNodeImageUpload'
import { t } from '@/i18n'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { IComboWidget } from '@/lib/litegraph/src/types/widgets'
import type { ResultItemType } from '@/schemas/apiSchema'
import type { InputSpec } from '@/schemas/nodeDefSchema'
import type { ComfyWidgetConstructor } from '@/scripts/widgets'
import { useNodeOutputStore } from '@/stores/imagePreviewStore'
import { isImageUploadInput } from '@/types/nodeDefAugmentation'
import { createAnnotatedPath } from '@/utils/createAnnotatedPath'
import { addToComboValues } from '@/utils/litegraphUtil'

const ACCEPTED_IMAGE_TYPES = 'image/png,image/jpeg,image/webp'
const ACCEPTED_VIDEO_TYPES = 'video/webm,video/mp4'

const isImageFile = (file: File) => file.type.startsWith('image/')
const isVideoFile = (file: File) => file.type.startsWith('video/')

const findFileComboWidget = (
  node: LGraphNode,
  inputName: string
): IComboWidget | undefined =>
  node.widgets?.find((w): w is IComboWidget => w.name === inputName)

export const useImageUploadWidget = () => {
  const widgetConstructor: ComfyWidgetConstructor = (
    node: LGraphNode,
    inputName: string,
    inputData: InputSpec
  ) => {
    if (!isImageUploadInput(inputData)) {
      throw new Error(
        'Image upload widget requires imageInputName augmentation'
      )
    }

    const inputOptions = inputData[1]
    const { imageInputName, allow_batch, image_folder = 'input' } = inputOptions
    const folder: ResultItemType | undefined = image_folder
    const nodeOutputStore = useNodeOutputStore()

    const isAnimated = !!inputOptions.animated_image_upload
    const isVideo = !!inputOptions.video_upload
    const accept = isVideo ? ACCEPTED_VIDEO_TYPES : ACCEPTED_IMAGE_TYPES
    const { showPreview } = isVideo ? useNodeVideo(node) : useNodeImage(node)

    const fileFilter = isVideo ? isVideoFile : isImageFile
    const fileComboWidget = findFileComboWidget(node, imageInputName)
    if (!fileComboWidget) {
      throw new Error(`Widget "${imageInputName}" not found on node`)
    }
    const formatPath = (value: string) =>
      createAnnotatedPath(value, { rootFolder: image_folder })

    // Setup file upload handling
    let rollback: (() => void) | undefined
    const { openFileSelection } = useNodeImageUpload(node, {
      allow_batch,
      fileFilter,
      accept,
      folder,
      onUploadStart: (files) => {
        if (files.length > 0) {
          const prev = fileComboWidget.value
          fileComboWidget.value = files[0].name
          rollback = () => {
            fileComboWidget.value = prev
          }
        }
      },
      onUploadError: () => {
        rollback?.()
        rollback = undefined
      },
      onUploadComplete: (output) => {
        rollback = undefined
        const annotated = output.map(formatPath)
        annotated.forEach((path) => {
          addToComboValues(fileComboWidget, path)
        })

        const newValue = allow_batch ? annotated : annotated[0]

        // @ts-expect-error litegraph combo value type does not support arrays yet
        fileComboWidget.value = newValue
        fileComboWidget.callback?.(newValue)
      }
    })

    // Create the button widget for selecting the files
    const uploadWidget = node.addWidget(
      'button',
      inputName,
      'image',
      () => openFileSelection(),
      {
        serialize: false,
        canvasOnly: true
      }
    )
    uploadWidget.label = t('g.choose_file_to_upload')

    // Add our own callback to the combo widget to render an image when it changes
    fileComboWidget.callback = function () {
      node.imgs = undefined
      nodeOutputStore.setNodeOutputs(node, String(fileComboWidget.value), {
        isAnimated
      })
      node.graph?.setDirtyCanvas(true)
    }

    // On load if we have a value then render the image
    // The value isn't set immediately so we need to wait a moment
    // No change callbacks seem to be fired on initial setting of the value
    requestAnimationFrame(() => {
      nodeOutputStore.setNodeOutputs(node, String(fileComboWidget.value), {
        isAnimated
      })
      showPreview({ block: false })
    })

    return { widget: uploadWidget }
  }

  return widgetConstructor
}
