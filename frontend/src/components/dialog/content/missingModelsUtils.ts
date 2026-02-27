import { isDesktop } from '@/platform/distribution/types'
import { useElectronDownloadStore } from '@/stores/electronDownloadStore'

const ALLOWED_SOURCES = [
  'https://civitai.com/',
  'https://huggingface.co/',
  'http://localhost:'
] as const

const ALLOWED_SUFFIXES = [
  '.safetensors',
  '.sft',
  '.ckpt',
  '.pth',
  '.pt'
] as const

const WHITE_LISTED_URLS: ReadonlySet<string> = new Set([
  'https://huggingface.co/stabilityai/stable-zero123/resolve/main/stable_zero123.ckpt',
  'https://huggingface.co/TencentARC/T2I-Adapter/resolve/main/models/t2iadapter_depth_sd14v1.pth?download=true',
  'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
])

const DIRECTORY_BADGE_MAP = {
  vae: 'VAE',
  diffusion_models: 'DIFFUSION',
  text_encoders: 'TEXT ENCODER',
  loras: 'LORA',
  checkpoints: 'CHECKPOINT'
} as const

export interface ModelWithUrl {
  name: string
  url: string
  directory: string
}

export function isModelDownloadable(model: ModelWithUrl): boolean {
  if (WHITE_LISTED_URLS.has(model.url)) return true
  if (!ALLOWED_SOURCES.some((source) => model.url.startsWith(source)))
    return false
  if (!ALLOWED_SUFFIXES.some((suffix) => model.name.endsWith(suffix)))
    return false
  return true
}

export function hasValidDirectory(
  model: ModelWithUrl,
  paths: Record<string, string[]>
): boolean {
  return !!paths[model.directory]
}

export function getBadgeLabel(directory: string): string {
  if (directory in DIRECTORY_BADGE_MAP) {
    return DIRECTORY_BADGE_MAP[directory as keyof typeof DIRECTORY_BADGE_MAP]
  }
  return directory.toUpperCase()
}

export function downloadModel(
  model: ModelWithUrl,
  paths: Record<string, string[]>
): void {
  if (!isDesktop) {
    const link = document.createElement('a')
    link.href = model.url
    link.download = model.name
    link.target = '_blank'
    link.rel = 'noopener noreferrer'
    link.click()
    return
  }

  const modelPaths = paths[model.directory]
  if (modelPaths?.[0]) {
    void useElectronDownloadStore().start({
      url: model.url,
      savePath: modelPaths[0],
      filename: model.name
    })
  }
}
