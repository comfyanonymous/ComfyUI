import { downloadBlob } from '@/base/common/downloadUtil'
import { t } from '@/i18n'
import type { IContextMenuValue } from '@/lib/litegraph/src/interfaces'
import type { LGraphCanvas } from '@/lib/litegraph/src/litegraph'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { useDialogService } from '@/services/dialogService'
import type { ComfyExtension } from '@/types/comfy'
import { deserialiseAndCreate } from '@/utils/vintageClipboard'

import { api } from '../../scripts/api'
import { app } from '../../scripts/app'
import { $el, ComfyDialog } from '../../scripts/ui'
import { GroupNodeConfig, GroupNodeHandler } from './groupNode'

// Adds the ability to save and add multiple nodes as a template
// To save:
// Select multiple nodes (ctrl + drag to select a region or ctrl+click individual nodes)
// Right click the canvas
// Save Node Template -> give it a name
//
// To add:
// Right click the canvas
// Node templates -> click the one to add
//
// To delete/rename:
// Right click the canvas
// Node templates -> Manage
//
// To rearrange:
// Open the manage dialog and Drag and drop elements using the "Name:" label as handle

const id = 'Comfy.NodeTemplates'
const file = 'comfy.templates.json'

interface NodeTemplate {
  name: string
  data: string
}

class ManageTemplates extends ComfyDialog {
  templates: NodeTemplate[] = []
  draggedEl: HTMLElement | null
  saveVisualCue: number | null
  emptyImg: HTMLImageElement
  importInput: HTMLInputElement

  constructor() {
    super()
    this.load().then((v) => {
      this.templates = v
    })

    this.element.classList.add('comfy-manage-templates')
    this.draggedEl = null
    this.saveVisualCue = null
    this.emptyImg = new Image()
    this.emptyImg.src =
      'data:image/gif;base64,R0lGODlhAQABAIAAAAUEBAAAACwAAAAAAQABAAACAkQBADs='

    this.importInput = $el('input', {
      type: 'file',
      accept: '.json',
      multiple: true,
      style: { display: 'none' },
      parent: document.body,
      onchange: () => this.importAll()
    }) as HTMLInputElement
  }

  override createButtons() {
    const btns = super.createButtons()
    btns[0].textContent = 'Close'
    btns[0].onclick = () => {
      // @ts-expect-error fixme ts strict error
      clearTimeout(this.saveVisualCue)
      this.close()
    }
    btns.unshift(
      $el('button', {
        type: 'button',
        textContent: 'Export',
        onclick: () => this.exportAll()
      })
    )
    btns.unshift(
      $el('button', {
        type: 'button',
        textContent: 'Import',
        onclick: () => {
          this.importInput.click()
        }
      })
    )
    return btns
  }

  async load() {
    let templates = []
    const res = await api.getUserData(file)
    if (res.status === 200) {
      try {
        templates = await res.json()
      } catch (error) {}
    } else if (res.status !== 404) {
      console.error(res.status + ' ' + res.statusText)
    }
    return templates ?? []
  }

  async store() {
    const templates = JSON.stringify(this.templates, undefined, 4)
    try {
      await api.storeUserData(file, templates, { stringify: false })
    } catch (error) {
      console.error(error)
      // @ts-expect-error fixme ts strict error
      useToastStore().addAlert(error.message)
    }
  }

  async importAll() {
    // @ts-expect-error fixme ts strict error
    for (const file of this.importInput.files) {
      if (file.type === 'application/json' || file.name.endsWith('.json')) {
        const reader = new FileReader()
        reader.onload = async () => {
          const importFile = JSON.parse(reader.result as string)
          if (importFile?.templates) {
            for (const template of importFile.templates) {
              if (template?.name && template?.data) {
                this.templates.push(template)
              }
            }
            await this.store()
          }
        }
        await reader.readAsText(file)
      }
    }

    // @ts-expect-error fixme ts strict error
    this.importInput.value = null

    this.close()
  }

  exportAll() {
    if (this.templates.length == 0) {
      useToastStore().addAlert(t('toastMessages.noTemplatesToExport'))
      return
    }

    const json = JSON.stringify({ templates: this.templates }, null, 2) // convert the data to a JSON string
    const blob = new Blob([json], { type: 'application/json' })
    downloadBlob('node_templates.json', blob)
  }

  override show() {
    // Show list of template names + delete button
    super.show(
      $el(
        'div',
        {},
        this.templates.flatMap((t, i) => {
          // @ts-expect-error fixme ts strict error
          let nameInput
          return [
            $el(
              'div',
              {
                dataset: { id: i.toString() },
                className: 'templateManagerRow',
                style: {
                  display: 'grid',
                  gridTemplateColumns: '1fr auto',
                  border: '1px dashed transparent',
                  gap: '5px',
                  backgroundColor: 'var(--comfy-menu-bg)'
                },
                // @ts-expect-error fixme ts strict error
                ondragstart: (e) => {
                  this.draggedEl = e.currentTarget
                  e.currentTarget.style.opacity = '0.6'
                  e.currentTarget.style.border = '1px dashed yellow'
                  e.dataTransfer.effectAllowed = 'move'
                  e.dataTransfer.setDragImage(this.emptyImg, 0, 0)
                },
                // @ts-expect-error fixme ts strict error
                ondragend: (e) => {
                  e.target.style.opacity = '1'
                  e.currentTarget.style.border = '1px dashed transparent'
                  e.currentTarget.removeAttribute('draggable')

                  // rearrange the elements
                  this.element
                    .querySelectorAll('.templateManagerRow')
                    // @ts-expect-error fixme ts strict error
                    .forEach((el: HTMLElement, i) => {
                      // @ts-expect-error fixme ts strict error
                      var prev_i = Number.parseInt(el.dataset.id)

                      if (el == this.draggedEl && prev_i != i) {
                        this.templates.splice(
                          i,
                          0,
                          this.templates.splice(prev_i, 1)[0]
                        )
                      }
                      el.dataset.id = i.toString()
                    })
                  this.store()
                },
                // @ts-expect-error fixme ts strict error
                ondragover: (e) => {
                  e.preventDefault()
                  if (e.currentTarget == this.draggedEl) return

                  let rect = e.currentTarget.getBoundingClientRect()
                  if (e.clientY > rect.top + rect.height / 2) {
                    e.currentTarget.parentNode.insertBefore(
                      this.draggedEl,
                      e.currentTarget.nextSibling
                    )
                  } else {
                    e.currentTarget.parentNode.insertBefore(
                      this.draggedEl,
                      e.currentTarget
                    )
                  }
                }
              },
              [
                $el(
                  'label',
                  {
                    textContent: 'Name: ',
                    style: {
                      cursor: 'grab'
                    },
                    // @ts-expect-error fixme ts strict error
                    onmousedown: (e) => {
                      // enable dragging only from the label
                      if (e.target.localName == 'label')
                        e.currentTarget.parentNode.draggable = 'true'
                    }
                  },
                  [
                    $el('input', {
                      value: t.name,
                      dataset: { name: t.name },
                      style: {
                        transitionProperty: 'background-color',
                        transitionDuration: '0s'
                      },
                      // @ts-expect-error fixme ts strict error
                      onchange: (e) => {
                        // @ts-expect-error fixme ts strict error
                        clearTimeout(this.saveVisualCue)
                        var el = e.target
                        var row = el.parentNode.parentNode
                        this.templates[row.dataset.id].name =
                          el.value.trim() || 'untitled'
                        this.store()
                        el.style.backgroundColor = 'rgb(40, 95, 40)'
                        el.style.transitionDuration = '0s'
                        // @ts-expect-error
                        // In browser env the return value is number.
                        this.saveVisualCue = setTimeout(function () {
                          el.style.transitionDuration = '.7s'
                          el.style.backgroundColor = 'var(--comfy-input-bg)'
                        }, 15)
                      },
                      // @ts-expect-error fixme ts strict error
                      onkeypress: (e) => {
                        var el = e.target
                        // @ts-expect-error fixme ts strict error
                        clearTimeout(this.saveVisualCue)
                        el.style.transitionDuration = '0s'
                        el.style.backgroundColor = 'var(--comfy-input-bg)'
                      },
                      $: (el) => (nameInput = el)
                    })
                  ]
                ),
                $el('div', {}, [
                  $el('button', {
                    textContent: 'Export',
                    style: {
                      fontSize: '12px',
                      fontWeight: 'normal'
                    },
                    onclick: () => {
                      const json = JSON.stringify({ templates: [t] }, null, 2) // convert the data to a JSON string
                      const blob = new Blob([json], {
                        type: 'application/json'
                      })
                      // @ts-expect-error fixme ts strict error
                      const name = (nameInput.value || t.name) + '.json'
                      downloadBlob(name, blob)
                    }
                  }),
                  $el('button', {
                    textContent: 'Delete',
                    style: {
                      fontSize: '12px',
                      color: 'red',
                      fontWeight: 'normal'
                    },
                    // @ts-expect-error fixme ts strict error
                    onclick: (e) => {
                      const item = e.target.parentNode.parentNode
                      item.parentNode.removeChild(item)
                      this.templates.splice(item.dataset.id * 1, 1)
                      this.store()
                      // update the rows index, setTimeout ensures that the list is updated
                      var that = this
                      setTimeout(function () {
                        that.element
                          .querySelectorAll('.templateManagerRow')
                          // @ts-expect-error fixme ts strict error
                          .forEach((el: HTMLElement, i) => {
                            el.dataset.id = i.toString()
                          })
                      }, 0)
                    }
                  })
                ])
              ]
            )
          ]
        })
      )
    )
  }
}

const manage = new ManageTemplates()

// @ts-expect-error fixme ts strict error
const clipboardAction = async (cb) => {
  // We use the clipboard functions but dont want to overwrite the current user clipboard
  // Restore it after we've run our callback
  const old = localStorage.getItem('litegrapheditor_clipboard')
  await cb()
  // @ts-expect-error fixme ts strict error
  localStorage.setItem('litegrapheditor_clipboard', old)
}

const ext: ComfyExtension = {
  name: id,

  getCanvasMenuItems(_canvas: LGraphCanvas): IContextMenuValue[] {
    const items: IContextMenuValue[] = []

    // @ts-expect-error fixme ts strict error
    items.push(null)
    items.push({
      content: `Save Selected as Template`,
      disabled: !Object.keys(app.canvas.selected_nodes || {}).length,
      callback: async () => {
        const name = await useDialogService().prompt({
          title: t('nodeTemplates.saveAsTemplate'),
          message: t('nodeTemplates.enterName'),
          defaultValue: ''
        })
        if (!name?.trim()) return

        clipboardAction(() => {
          app.canvas.copyToClipboard()
          let data = localStorage.getItem('litegrapheditor_clipboard')
          data = JSON.parse(data || '{}')
          const nodeIds = Object.keys(app.canvas.selected_nodes)
          for (let i = 0; i < nodeIds.length; i++) {
            const node = app.canvas.graph?.getNodeById(nodeIds[i])
            const nodeData = node?.constructor.nodeData

            if (!node) continue
            const groupConfig = GroupNodeHandler.getGroupData(node)
            if (groupConfig) {
              const groupData = groupConfig.nodeData
              // @ts-expect-error
              if (!data.groupNodes) {
                // @ts-expect-error
                data.groupNodes = {}
              }
              if (nodeData == null) throw new TypeError('nodeData is not set')
              // @ts-expect-error
              data.groupNodes[nodeData.name] = groupData
              // @ts-expect-error
              data.nodes[i].type = nodeData.name
            }
          }

          manage.templates.push({
            name,
            data: JSON.stringify(data)
          })
          manage.store()
        })
      }
    })

    // Map each template to a menu item
    const subItems = manage.templates.map((t) => {
      return {
        content: t.name,
        callback: () => {
          clipboardAction(async () => {
            const data = JSON.parse(t.data)
            await GroupNodeConfig.registerFromWorkflow(
              data.groupNodes ?? {},
              []
            )

            // Check for old clipboard format
            if (!data.reroutes) {
              deserialiseAndCreate(t.data, app.canvas)
            } else {
              localStorage.setItem('litegrapheditor_clipboard', t.data)
              app.canvas.pasteFromClipboard()
            }
          })
        }
      }
    })

    // @ts-expect-error fixme ts strict error
    subItems.push(null, {
      content: 'Manage',
      callback: () => manage.show()
    })

    items.push({
      content: 'Node Templates',
      submenu: {
        options: subItems
      }
    })

    return items
  }
}

app.registerExtension(ext)
