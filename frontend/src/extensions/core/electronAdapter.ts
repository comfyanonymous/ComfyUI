import log from 'loglevel'

import { useExternalLink } from '@/composables/useExternalLink'
import { PYTHON_MIRROR } from '@/constants/uvMirrors'
import { t } from '@/i18n'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { app } from '@/scripts/app'
import { useDialogService } from '@/services/dialogService'
import { checkMirrorReachable } from '@/utils/electronMirrorCheck'
import { isDesktop } from '@/platform/distribution/types'
import { electronAPI as getElectronAPI } from '@/utils/envUtil'
;(async () => {
  if (!isDesktop) return

  const electronAPI = getElectronAPI()
  const desktopAppVersion = await electronAPI.getElectronVersion()
  const workflowStore = useWorkflowStore()
  const toastStore = useToastStore()
  const { staticUrls, buildDocsUrl } = useExternalLink()

  const onChangeRestartApp = (newValue: unknown, oldValue: unknown) => {
    // Add a delay to allow changes to take effect before restarting.
    if (oldValue !== undefined && newValue !== oldValue) {
      electronAPI.restartApp('Restart ComfyUI to apply changes.', 1500)
    }
  }

  app.registerExtension({
    name: 'Comfy.ElectronAdapter',
    settings: [
      {
        id: 'Comfy-Desktop.AutoUpdate',
        category: ['Comfy-Desktop', 'General', 'AutoUpdate'],
        name: 'Automatically check for updates',
        type: 'boolean',
        defaultValue: true,
        onChange: onChangeRestartApp
      },
      {
        id: 'Comfy-Desktop.SendStatistics',
        category: ['Comfy-Desktop', 'General', 'Send Statistics'],
        name: 'Send anonymous usage metrics',
        type: 'boolean',
        defaultValue: true,
        onChange: onChangeRestartApp
      },
      {
        id: 'Comfy-Desktop.WindowStyle',
        category: ['Comfy-Desktop', 'General', 'Window Style'],
        name: 'Window Style',
        tooltip: "Custom: Replace the system title bar with ComfyUI's Top menu",
        type: 'combo',
        experimental: true,
        defaultValue: 'default',
        options: ['default', 'custom'],
        onChange: (
          newValue: 'default' | 'custom',
          oldValue?: 'default' | 'custom'
        ) => {
          if (!oldValue) return

          electronAPI.Config.setWindowStyle(newValue)
        }
      },
      {
        id: 'Comfy-Desktop.UV.PythonInstallMirror',
        name: 'Python Install Mirror',
        tooltip: `Managed Python installations are downloaded from the Astral python-build-standalone project. This variable can be set to a mirror URL to use a different source for Python installations. The provided URL will replace https://github.com/astral-sh/python-build-standalone/releases/download in, e.g., https://github.com/astral-sh/python-build-standalone/releases/download/20240713/cpython-3.12.4%2B20240713-aarch64-apple-darwin-install_only.tar.gz. Distributions can be read from a local directory by using the file:// URL scheme.`,
        type: 'url',
        defaultValue: '',
        attrs: {
          validateUrlFn(mirror: string) {
            return checkMirrorReachable(
              mirror + PYTHON_MIRROR.validationPathSuffix
            )
          }
        }
      },
      {
        id: 'Comfy-Desktop.UV.PypiInstallMirror',
        name: 'Pypi Install Mirror',
        tooltip: `Default pip install mirror`,
        type: 'url',
        defaultValue: '',
        attrs: {
          validateUrlFn: checkMirrorReachable
        }
      },
      {
        id: 'Comfy-Desktop.UV.TorchInstallMirror',
        name: 'Torch Install Mirror',
        tooltip: `Pip install mirror for pytorch`,
        type: 'url',
        defaultValue: '',
        attrs: {
          validateUrlFn: checkMirrorReachable
        }
      }
    ],

    commands: [
      {
        id: 'Comfy-Desktop.Folders.OpenLogsFolder',
        label: 'Open Logs Folder',
        icon: 'pi pi-folder-open',
        function() {
          electronAPI.openLogsFolder()
        }
      },
      {
        id: 'Comfy-Desktop.Folders.OpenModelsFolder',
        label: 'Open Models Folder',
        icon: 'pi pi-folder-open',
        function() {
          electronAPI.openModelsFolder()
        }
      },
      {
        id: 'Comfy-Desktop.Folders.OpenOutputsFolder',
        label: 'Open Outputs Folder',
        icon: 'pi pi-folder-open',
        function() {
          electronAPI.openOutputsFolder()
        }
      },
      {
        id: 'Comfy-Desktop.Folders.OpenInputsFolder',
        label: 'Open Inputs Folder',
        icon: 'pi pi-folder-open',
        function() {
          electronAPI.openInputsFolder()
        }
      },
      {
        id: 'Comfy-Desktop.Folders.OpenCustomNodesFolder',
        label: 'Open Custom Nodes Folder',
        icon: 'pi pi-folder-open',
        function() {
          electronAPI.openCustomNodesFolder()
        }
      },
      {
        id: 'Comfy-Desktop.Folders.OpenModelConfig',
        label: 'Open extra_model_paths.yaml',
        icon: 'pi pi-file',
        function() {
          electronAPI.openModelConfig()
        }
      },
      {
        id: 'Comfy-Desktop.OpenDevTools',
        label: 'Open DevTools',
        icon: 'pi pi-code',
        function() {
          electronAPI.openDevTools()
        }
      },
      {
        id: 'Comfy-Desktop.OpenUserGuide',
        label: 'Desktop User Guide',
        icon: 'pi pi-book',
        function() {
          window.open(
            buildDocsUrl('/installation/desktop', {
              includeLocale: true,
              platform: true
            }),
            '_blank'
          )
        }
      },
      {
        id: 'Comfy-Desktop.CheckForUpdates',
        label: 'Check for Updates',
        icon: 'pi pi-sync',
        async function() {
          try {
            const updateInfo = await electronAPI.checkForUpdates({
              disableUpdateReadyAction: true
            })

            if (!updateInfo.isUpdateAvailable) {
              toastStore.add({
                severity: 'info',
                summary: t('desktopUpdate.noUpdateFound'),
                life: 5_000
              })
              return
            }

            const proceed = await useDialogService().confirm({
              title: t('desktopUpdate.updateFoundTitle', {
                version: updateInfo.version
              }),
              message: t('desktopUpdate.updateAvailableMessage'),
              type: 'default'
            })
            if (proceed) {
              try {
                electronAPI.restartAndInstall()
              } catch (error) {
                log.error('Error installing update:', error)
                toastStore.add({
                  severity: 'error',
                  summary: t('g.error'),
                  detail: t('desktopUpdate.errorInstallingUpdate'),
                  life: 10_000
                })
              }
            }
          } catch (error) {
            log.error('Error checking for updates:', error)
            toastStore.add({
              severity: 'error',
              summary: t('g.error'),
              detail: t('desktopUpdate.errorCheckingUpdate'),
              life: 10_000
            })
          }
        }
      },
      {
        id: 'Comfy-Desktop.Reinstall',
        label: 'Reinstall',
        icon: 'pi pi-refresh',
        async function() {
          const proceed = await useDialogService().confirm({
            message: t('desktopMenu.confirmReinstall'),
            title: t('desktopMenu.reinstall'),
            type: 'reinstall'
          })

          if (proceed) electronAPI.reinstall()
        }
      },
      {
        id: 'Comfy-Desktop.Restart',
        label: 'Restart',
        icon: 'pi pi-refresh',
        function() {
          electronAPI.restartApp()
        }
      },
      {
        id: 'Comfy-Desktop.Quit',
        label: 'Quit',
        icon: 'pi pi-sign-out',
        async function() {
          // Confirm if unsaved workflows are open
          if (workflowStore.modifiedWorkflows.length > 0) {
            const confirmed = await useDialogService().confirm({
              message: t('desktopMenu.confirmQuit'),
              title: t('desktopMenu.quit'),
              type: 'default'
            })

            if (!confirmed) return
          }

          electronAPI.quit()
        }
      }
    ],

    menuCommands: [
      {
        path: ['Help'],
        commands: ['Comfy-Desktop.OpenUserGuide']
      },
      {
        path: ['Help'],
        commands: ['Comfy-Desktop.OpenDevTools']
      },
      {
        path: ['Help', 'Open Folder'],
        commands: [
          'Comfy-Desktop.Folders.OpenLogsFolder',
          'Comfy-Desktop.Folders.OpenModelsFolder',
          'Comfy-Desktop.Folders.OpenOutputsFolder',
          'Comfy-Desktop.Folders.OpenInputsFolder',
          'Comfy-Desktop.Folders.OpenCustomNodesFolder',
          'Comfy-Desktop.Folders.OpenModelConfig'
        ]
      },
      {
        path: ['Help'],
        commands: ['Comfy-Desktop.CheckForUpdates', 'Comfy-Desktop.Reinstall']
      }
    ],

    keybindings: [
      {
        commandId: 'Workspace.CloseWorkflow',
        combo: {
          key: 'w',
          ctrl: true
        }
      }
    ],

    aboutPageBadges: [
      {
        label: 'ComfyUI_desktop v' + desktopAppVersion,
        url: staticUrls.githubElectron,
        icon: 'pi pi-github'
      }
    ]
  })
})()
