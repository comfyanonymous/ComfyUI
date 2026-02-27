import type Load3d from '@/extensions/core/load3d/Load3d'
import { t } from '@/i18n'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { api } from '@/scripts/api'
import { app } from '@/scripts/app'

class Load3dUtils {
  static async generateThumbnailIfNeeded(
    load3d: Load3d,
    modelPath: string,
    folderType: 'input' | 'output'
  ): Promise<void> {
    const [subfolder, filename] = this.splitFilePath(modelPath)
    const thumbnailFilename = this.getThumbnailFilename(filename)

    const exists = await this.fileExists(
      subfolder,
      thumbnailFilename,
      folderType
    )
    if (exists) return

    const imageData = await load3d.captureThumbnail(256, 256)
    await this.uploadThumbnail(
      imageData,
      subfolder,
      thumbnailFilename,
      folderType
    )
  }

  static async uploadTempImage(
    imageData: string,
    prefix: string,
    fileType: string = 'png'
  ) {
    const blob = await fetch(imageData).then((r) => r.blob())
    const name = `${prefix}_${Date.now()}.${fileType}`
    const file = new File([blob], name, {
      type: fileType === 'mp4' ? 'video/mp4' : 'image/png'
    })

    const body = new FormData()
    body.append('image', file)
    body.append('subfolder', 'threed')
    body.append('type', 'temp')

    const resp = await api.fetchApi('/upload/image', {
      method: 'POST',
      body
    })

    if (resp.status !== 200) {
      const err = `Error uploading temp file: ${resp.status} - ${resp.statusText}`
      useToastStore().addAlert(err)
      throw new Error(err)
    }

    return await resp.json()
  }

  static readonly MAX_UPLOAD_SIZE_MB = 100

  static async uploadFile(file: File, subfolder: string) {
    let uploadPath

    const fileSizeMB = file.size / 1024 / 1024
    if (fileSizeMB > this.MAX_UPLOAD_SIZE_MB) {
      const message = t('toastMessages.fileTooLarge', {
        size: fileSizeMB.toFixed(1),
        maxSize: this.MAX_UPLOAD_SIZE_MB
      })
      console.warn(
        '[Load3D] uploadFile: file too large',
        fileSizeMB.toFixed(2),
        'MB'
      )
      useToastStore().addAlert(message)
      return undefined
    }

    try {
      const body = new FormData()
      body.append('image', file)

      body.append('subfolder', subfolder)

      const resp = await api.fetchApi('/upload/image', {
        method: 'POST',
        body
      })

      if (resp.status === 200) {
        const data = await resp.json()
        let path = data.name

        if (data.subfolder) {
          path = data.subfolder + '/' + path
        }

        uploadPath = path
      } else {
        useToastStore().addAlert(resp.status + ' - ' + resp.statusText)
      }
    } catch (error) {
      console.error('[Load3D] uploadFile: exception', error)
      useToastStore().addAlert(
        error instanceof Error
          ? error.message
          : t('toastMessages.fileUploadFailed')
      )
    }

    return uploadPath
  }

  static splitFilePath(path: string): [string, string] {
    const folder_separator = path.lastIndexOf('/')
    if (folder_separator === -1) {
      return ['', path]
    }
    return [
      path.substring(0, folder_separator),
      path.substring(folder_separator + 1)
    ]
  }

  static getResourceURL(
    subfolder: string,
    filename: string,
    type: string = 'input'
  ): string {
    const params = [
      'filename=' + encodeURIComponent(filename),
      'type=' + type,
      'subfolder=' + subfolder,
      app.getRandParam().substring(1)
    ].join('&')

    return `/view?${params}`
  }

  static async uploadMultipleFiles(files: FileList, subfolder: string = '3d') {
    const uploadPromises = Array.from(files).map((file) =>
      this.uploadFile(file, subfolder)
    )

    await Promise.all(uploadPromises)
  }

  static getThumbnailFilename(modelFilename: string): string {
    return `${modelFilename}.png`
  }

  static async fileExists(
    subfolder: string,
    filename: string,
    type: string = 'input'
  ): Promise<boolean> {
    try {
      const url = api.apiURL(this.getResourceURL(subfolder, filename, type))
      const response = await fetch(url, { method: 'HEAD' })
      return response.ok
    } catch {
      return false
    }
  }

  static async uploadThumbnail(
    imageData: string,
    subfolder: string,
    filename: string,
    type: string = 'input'
  ): Promise<boolean> {
    const blob = await fetch(imageData).then((r) => r.blob())
    const file = new File([blob], filename, { type: 'image/png' })

    const body = new FormData()
    body.append('image', file)
    body.append('subfolder', subfolder)
    body.append('type', type)

    const resp = await api.fetchApi('/upload/image', {
      method: 'POST',
      body
    })

    return resp.status === 200
  }
}

export default Load3dUtils
