import { app } from '../../scripts/app'
import { ComfyApp } from '../../scripts/app'
import { $el, ComfyDialog } from '../../scripts/ui'

export class ClipspaceDialog extends ComfyDialog {
  static items: Array<
    HTMLButtonElement & {
      contextPredicate?: () => boolean
    }
  > = []
  static instance: ClipspaceDialog | null = null

  static registerButton(
    name: string,
    contextPredicate: () => boolean,
    callback: () => void
  ) {
    const item = $el('button', {
      type: 'button',
      textContent: name,
      contextPredicate: contextPredicate,
      onclick: callback
    })

    ClipspaceDialog.items.push(item)
  }

  static invalidatePreview() {
    if (
      ComfyApp.clipspace &&
      ComfyApp.clipspace.imgs &&
      ComfyApp.clipspace.imgs.length > 0
    ) {
      const img_preview = document.getElementById(
        'clipspace_preview'
      ) as HTMLImageElement
      if (img_preview) {
        img_preview.src =
          ComfyApp.clipspace.imgs[ComfyApp.clipspace['selectedIndex']].src
        img_preview.style.maxHeight = '100%'
        img_preview.style.maxWidth = '100%'
      }
    }
  }

  static invalidate() {
    if (ClipspaceDialog.instance) {
      const self = ClipspaceDialog.instance
      // allow reconstruct controls when copying from non-image to image content.
      const imgSettings = self.createImgSettings()
      const children = $el('div.comfy-modal-content', [
        ...(imgSettings ? [imgSettings] : []),
        ...self.createButtons()
      ])

      if (self.element) {
        // update
        if (self.element.firstChild) {
          self.element.removeChild(self.element.firstChild)
        }
        self.element.appendChild(children)
      } else {
        // new
        self.element = $el('div.comfy-modal', { parent: document.body }, [
          children
        ])
      }

      if (self.element.children[0].children.length <= 1) {
        self.element.children[0].appendChild(
          $el('p', {}, [
            'Unable to find the features to edit content of a format stored in the current Clipspace.'
          ])
        )
      }

      ClipspaceDialog.invalidatePreview()
    }
  }

  constructor() {
    super()
  }

  override createButtons() {
    const buttons = []

    for (let idx in ClipspaceDialog.items) {
      const item = ClipspaceDialog.items[idx]
      if (!item.contextPredicate || item.contextPredicate())
        buttons.push(ClipspaceDialog.items[idx])
    }

    buttons.push(
      $el('button', {
        type: 'button',
        textContent: 'Close',
        onclick: () => {
          this.close()
        }
      })
    )

    return buttons
  }

  createImgSettings(): HTMLTableElement | null {
    if (ComfyApp.clipspace?.imgs) {
      const combo_items = []
      const imgs = ComfyApp.clipspace.imgs

      for (let i = 0; i < imgs.length; i++) {
        combo_items.push($el('option', { value: i }, [`${i}`]))
      }

      const combo1 = $el(
        'select',
        {
          id: 'clipspace_img_selector',
          onchange: (event: Event) => {
            if (event.target && ComfyApp.clipspace) {
              ComfyApp.clipspace['selectedIndex'] = (
                event.target as HTMLSelectElement
              ).selectedIndex
              ClipspaceDialog.invalidatePreview()
            }
          }
        },
        combo_items
      )

      const row1 = $el('tr', {}, [
        $el('td', {}, [$el('font', { color: 'white' }, ['Select Image'])]),
        $el('td', {}, [combo1])
      ])

      const combo2 = $el(
        'select',
        {
          id: 'clipspace_img_paste_mode',
          onchange: (event: Event) => {
            if (event.target && ComfyApp.clipspace) {
              ComfyApp.clipspace['img_paste_mode'] = (
                event.target as HTMLSelectElement
              ).value
            }
          }
        },
        [
          $el('option', { value: 'selected' }, 'selected'),
          $el('option', { value: 'all' }, 'all')
        ]
      ) as HTMLSelectElement
      combo2.value = ComfyApp.clipspace['img_paste_mode']

      const row2 = $el('tr', {}, [
        $el('td', {}, [$el('font', { color: 'white' }, ['Paste Mode'])]),
        $el('td', {}, [combo2])
      ])

      const td = $el(
        'td',
        { align: 'center', width: '100px', height: '100px', colSpan: '2' },
        [$el('img', { id: 'clipspace_preview', ondragstart: () => false }, [])]
      )

      const row3 = $el('tr', {}, [td])

      return $el('table', {}, [row1, row2, row3])
    } else {
      return null
    }
  }

  createImgPreview(): HTMLImageElement | null {
    if (ComfyApp.clipspace?.imgs) {
      return $el('img', { id: 'clipspace_preview', ondragstart: () => false })
    } else return null
  }

  override show() {
    ClipspaceDialog.invalidate()

    this.element.style.display = 'block'
  }
}

app.registerExtension({
  name: 'Comfy.Clipspace',
  init() {
    app.openClipspace = function () {
      if (!ClipspaceDialog.instance) {
        ClipspaceDialog.instance = new ClipspaceDialog()
        ComfyApp.clipspace_invalidate_handler = ClipspaceDialog.invalidate
      }

      if (ComfyApp.clipspace) {
        ClipspaceDialog.instance.show()
      } else app.ui.dialog.show('Clipspace is Empty!')
    }
  }
})
