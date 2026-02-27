import { PrimeIcons } from '@primevue/core'

import type { MaintenanceTask } from '@/types/desktop/maintenanceTypes'
import { electronAPI } from '@/utils/envUtil'

const electron = electronAPI()

const openUrl = (url: string) => {
  window.open(url, '_blank')
  return true
}

export const DESKTOP_MAINTENANCE_TASKS: Readonly<MaintenanceTask>[] = [
  {
    id: 'basePath',
    execute: async () => await electron.setBasePath(),
    name: 'Base path',
    shortDescription: 'Change the application base path.',
    errorDescription:
      'The current base path is invalid or unsafe. Please select a new location.',
    description:
      'The base path is the default location where ComfyUI stores data. It is the location for the python environment, and may also contain models, custom nodes, and other extensions.',
    isInstallationFix: true,
    button: {
      icon: PrimeIcons.QUESTION,
      text: 'Select'
    }
  },
  {
    id: 'git',
    headerImg: 'assets/images/Git-Logo-White.svg',
    execute: () => openUrl('https://git-scm.com/downloads/'),
    name: 'Download git',
    shortDescription: 'Open the git download page.',
    errorDescription:
      'Git is missing. Please download and install git, then restart ComfyUI Desktop.',
    description:
      'Git is required to download and manage custom nodes and other extensions. This task opens the download page in your default browser, where you can download the latest version of git. Once you have installed git, please restart ComfyUI Desktop.',
    button: {
      icon: PrimeIcons.EXTERNAL_LINK,
      text: 'Download'
    }
  },
  {
    id: 'vcRedist',
    execute: () => openUrl('https://aka.ms/vs/17/release/vc_redist.x64.exe'),
    name: 'Download VC++ Redist',
    shortDescription: 'Download the latest VC++ Redistributable runtime.',
    description:
      'The Visual C++ runtime libraries are required to run ComfyUI. You will need to download and install this file.',
    button: {
      icon: PrimeIcons.EXTERNAL_LINK,
      text: 'Download'
    }
  },
  {
    id: 'reinstall',
    severity: 'danger',
    requireConfirm: true,
    execute: async () => {
      await electron.reinstall()
      return true
    },
    name: 'Reinstall ComfyUI',
    shortDescription:
      'Deletes the desktop app config and load the welcome screen.',
    description:
      'Delete the desktop app config, restart the app, and load the installation screen.',
    confirmText: 'Delete all saved config and reinstall?',
    button: {
      icon: PrimeIcons.EXCLAMATION_TRIANGLE,
      text: 'Reinstall'
    }
  },
  {
    id: 'pythonPackages',
    requireConfirm: true,
    execute: async () => {
      try {
        await electron.uv.installRequirements()
        return true
      } catch (error) {
        return false
      }
    },
    name: 'Install python packages',
    shortDescription:
      'Installs the base python packages required to run ComfyUI.',
    errorDescription:
      'Python packages that are required to run ComfyUI are not installed.',
    description:
      'This will install the python packages required to run ComfyUI. This includes torch, torchvision, and other dependencies.',
    usesTerminal: true,
    isInstallationFix: true,
    button: {
      icon: PrimeIcons.DOWNLOAD,
      text: 'Install'
    }
  },
  {
    id: 'uv',
    execute: () =>
      openUrl('https://docs.astral.sh/uv/getting-started/installation/'),
    name: 'uv executable',
    shortDescription: 'uv installs and maintains the python environment.',
    description:
      "This will open the download page for Astral's uv tool. uv is used to install python and manage python packages.",
    button: {
      icon: 'pi pi-asterisk',
      text: 'Download'
    }
  },
  {
    id: 'uvCache',
    severity: 'danger',
    requireConfirm: true,
    execute: async () => await electron.uv.clearCache(),
    name: 'uv cache',
    shortDescription: 'Remove the Astral uv cache of python packages.',
    description:
      'This will remove the uv cache directory and its contents. All downloaded python packages will need to be downloaded again.',
    confirmText: 'Delete uv cache of python packages?',
    usesTerminal: true,
    isInstallationFix: true,
    button: {
      icon: PrimeIcons.TRASH,
      text: 'Clear cache'
    }
  },
  {
    id: 'venvDirectory',
    severity: 'danger',
    requireConfirm: true,
    execute: async () => await electron.uv.resetVenv(),
    name: 'Reset virtual environment',
    shortDescription:
      'Remove and recreate the .venv directory. This removes all python packages.',
    description:
      'The python environment is where ComfyUI installs python and python packages. It is used to run the ComfyUI server.',
    confirmText: 'Delete the .venv directory?',
    usesTerminal: true,
    isInstallationFix: true,
    button: {
      icon: PrimeIcons.FOLDER,
      text: 'Recreate'
    }
  }
] as const
