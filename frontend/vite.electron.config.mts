import { defineConfig, mergeConfig } from 'vite'
import type { Plugin } from 'vite'

import baseConfig from './vite.config.mts'

const mockElectronAPI: Plugin = {
  name: 'mock-electron-api',
  transformIndexHtml() {
    return [
      {
        tag: 'script',
        children: `window.electronAPI = {
          restartApp: () => {
            alert('restartApp')
          },
          sendReady: () => {},
          onShowSelectDirectory: () => {},
          onFirstTimeSetupComplete: () => {},
          onProgressUpdate: () => {},
          onLogMessage: () => {},
          isFirstTimeSetup: () => Promise.resolve(true),
          getSystemPaths: () =>
            Promise.resolve({
              appData: 'C:/Users/username/AppData/Roaming',
              appPath: 'C:/Program Files/comfyui-electron/resources/app',
              defaultInstallPath: 'bad'
            }),
          validateInstallPath: (path) => {
            if (path === 'bad') {
              return { isValid: false, error: 'Bad path!' }
            }
            return { isValid: true }
          },
          migrationItems: () =>
            Promise.resolve([
              {
                id: 'user_files',
                label: 'User Files',
                description: 'Settings and user-created workflows'
              }
            ]),
          validateComfyUISource: (path) => {
            if (path === 'bad') {
              return { isValid: false, error: 'Bad path!' }
            }
            return { isValid: true }
          },
          showDirectoryPicker: () => Promise.resolve('C:/Users/username/comfyui-electron/1'),
          DownloadManager: {
            getAllDownloads: () => Promise.resolve([]),
            onDownloadProgress: () => {}
          },
          getElectronVersion: () => Promise.resolve('1.0.0'),
          getComfyUIVersion: () => '9.9.9',
          getPlatform: () => 'win32',
          changeTheme: () => {},
          Config: {
            setWindowStyle: () => {},
            getWindowStyle: () => Promise.resolve('default'),
            getDetectedGpu: () => Promise.resolve('nvidia')
          },
          Events: {
            trackEvent: (event_name, event_data) => {
              console.log('trackEvent', event_name, event_data)
            },
            incrementUserProperty: (property, value) => {
              console.log('incrementUserProperty', property, value)
            }
          },
          NetWork: {
            canAccessUrl: (url) => {
              const canAccess = url.includes('good')
              console.log('canAccessUrl', url, canAccess)
              return new Promise((resolve) => setTimeout(() => resolve(canAccess), 10000))
            }
          },
          setMetricsConsent: (consent) => {}
        };`
      }
    ]
  }
}

export default mergeConfig(
  baseConfig,
  defineConfig({
    plugins: [mockElectronAPI]
  })
)
