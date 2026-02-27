import { isObject } from 'es-toolkit/compat'

export function getDataFromJSON(
  file: File
): Promise<Record<string, object> | undefined> {
  return new Promise<Record<string, object> | undefined>((resolve) => {
    const reader = new FileReader()
    reader.onload = async () => {
      const readerResult = reader.result as string
      const jsonContent = JSON.parse(readerResult)
      if (jsonContent?.templates) {
        resolve({ templates: jsonContent.templates })
        return
      }
      if (isApiJson(jsonContent)) {
        resolve({ prompt: jsonContent })
        return
      }
      resolve({ workflow: jsonContent })
      return
    }
    reader.readAsText(file)
    return
  })
}

function isApiJson(data: unknown) {
  return isObject(data) && Object.values(data).every((v) => v.class_type)
}
