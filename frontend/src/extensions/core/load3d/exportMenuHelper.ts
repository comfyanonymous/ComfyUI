import { t } from '@/i18n'
import type { IContextMenuValue } from '@/lib/litegraph/src/interfaces'
import { useToastStore } from '@/platform/updates/common/toastStore'
import Load3d from '@/extensions/core/load3d/Load3d'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'

const EXPORT_FORMATS = [
  { label: 'GLB', value: 'glb' },
  { label: 'OBJ', value: 'obj' },
  { label: 'STL', value: 'stl' }
] as const

/**
 * Creates export menu items for a 3D node using the new extension API.
 * Returns an array of context menu items including a separator and export submenu.
 */
export function createExportMenuItems(
  load3d: Load3d
): (IContextMenuValue | null)[] {
  return [
    null, // Separator
    {
      content: 'Save',
      has_submenu: true,
      callback: (_value, _options, event, prev_menu) => {
        const submenuOptions: IContextMenuValue[] = EXPORT_FORMATS.map(
          (format) => ({
            content: format.label,
            callback: () => {
              void (async () => {
                try {
                  await load3d.exportModel(format.value)
                  useToastStore().add({
                    severity: 'success',
                    summary: t('toastMessages.exportSuccess', {
                      format: format.label
                    })
                  })
                } catch (error) {
                  console.error('Export failed:', error)
                  useToastStore().addAlert(
                    t('toastMessages.failedToExportModel', {
                      format: format.label
                    })
                  )
                }
              })()
            }
          })
        )

        new LiteGraph.ContextMenu(submenuOptions, {
          event,
          parentMenu: prev_menu
        })
      }
    }
  ]
}
