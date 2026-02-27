/**
 * Frontend augmentations for node definitions.
 *
 * This module defines type extensions that augment the backend node definition
 * types with frontend-specific properties. These augmentations are applied at
 * runtime and are not part of the backend API contract.
 */
import type { ComboInputOptions, InputSpec } from '@/schemas/nodeDefSchema'

/**
 * Frontend augmentation for image upload combo inputs.
 * This extends ComboInputOptions with properties injected by the uploadImage extension.
 */
interface ImageUploadComboOptions extends ComboInputOptions {
  /**
   * Reference to the associated filename combo widget.
   * Injected by uploadImage.ts to link upload buttons with their combo widgets.
   *
   * @remarks This property exists only in the frontend runtime.
   */
  imageInputName: string
}

/**
 * Type guard to check if an InputSpec has image upload augmentations.
 * Narrows from base InputSpec to augmented type.
 */
export function isImageUploadInput(
  inputData: InputSpec
): inputData is [string, ImageUploadComboOptions] {
  const options = inputData[1]
  return (
    options !== undefined &&
    typeof options === 'object' &&
    'imageInputName' in options &&
    typeof options.imageInputName === 'string'
  )
}
