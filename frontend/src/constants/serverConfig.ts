import type { FormItem } from '@/platform/settings/types'
import {
  CrossAttentionMethod,
  CudaMalloc,
  FloatingPointPrecision,
  HashFunction,
  LatentPreviewMethod,
  LogLevel,
  VramManagement
} from '@/types/serverArgs'

export type ServerConfigValue = string | number | boolean | null | undefined

export interface ServerConfig<T> extends FormItem {
  id: string
  defaultValue: T
  category?: string[]
  // Override the default value getter with a custom function.
  getValue?: (value: T) => Record<string, ServerConfigValue>
}

export const SERVER_CONFIG_ITEMS = [
  // Network settings
  {
    id: 'listen',
    name: 'Host: The IP address to listen on',
    category: ['Network'],
    type: 'text',
    defaultValue: '127.0.0.1'
  },
  {
    id: 'port',
    name: 'Port: The port to listen on',
    category: ['Network'],
    type: 'number',
    // The default launch port for desktop app is 8000 instead of 8188.
    defaultValue: 8000
  },
  {
    id: 'tls-keyfile',
    name: 'TLS Key File: Path to TLS key file for HTTPS',
    category: ['Network'],
    type: 'text',
    defaultValue: ''
  },
  {
    id: 'tls-certfile',
    name: 'TLS Certificate File: Path to TLS certificate file for HTTPS',
    category: ['Network'],
    type: 'text',
    defaultValue: ''
  },
  {
    id: 'enable-cors-header',
    name: 'Enable CORS header: Use "*" for all origins or specify domain',
    category: ['Network'],
    type: 'text',
    defaultValue: ''
  },
  {
    id: 'max-upload-size',
    name: 'Maximum upload size (MB)',
    category: ['Network'],
    type: 'number',
    defaultValue: 100
  },

  // CUDA settings
  {
    id: 'cuda-device',
    name: 'CUDA device index to use',
    category: ['CUDA'],
    type: 'number',
    defaultValue: null
  },
  {
    id: 'cuda-malloc',
    name: 'Use CUDA malloc for memory allocation',
    category: ['CUDA'],
    type: 'combo',
    options: Object.values(CudaMalloc),
    defaultValue: CudaMalloc.Auto,
    getValue: (value: CudaMalloc) => {
      switch (value) {
        case CudaMalloc.Auto:
          return {}
        case CudaMalloc.Enable:
          return {
            ['cuda-malloc']: true
          }
        case CudaMalloc.Disable:
          return {
            ['disable-cuda-malloc']: true
          }
      }
    }
  },

  // Precision settings
  {
    id: 'global-precision',
    name: 'Global floating point precision',
    category: ['Inference'],
    type: 'combo',
    options: [
      FloatingPointPrecision.AUTO,
      FloatingPointPrecision.FP32,
      FloatingPointPrecision.FP16
    ],
    defaultValue: FloatingPointPrecision.AUTO,
    tooltip: 'Global floating point precision',
    getValue: (value: FloatingPointPrecision) => {
      switch (value) {
        case FloatingPointPrecision.AUTO:
          return {}
        case FloatingPointPrecision.FP32:
          return {
            ['force-fp32']: true
          }
        case FloatingPointPrecision.FP16:
          return {
            ['force-fp16']: true
          }
        default:
          return {}
      }
    }
  },

  // UNET precision
  {
    id: 'unet-precision',
    name: 'UNET precision',
    category: ['Inference'],
    type: 'combo',
    options: [
      FloatingPointPrecision.AUTO,
      FloatingPointPrecision.FP64,
      FloatingPointPrecision.FP32,
      FloatingPointPrecision.FP16,
      FloatingPointPrecision.BF16,
      FloatingPointPrecision.FP8E4M3FN,
      FloatingPointPrecision.FP8E5M2
    ],
    defaultValue: FloatingPointPrecision.AUTO,
    tooltip: 'UNET precision',
    getValue: (value: FloatingPointPrecision) => {
      switch (value) {
        case FloatingPointPrecision.AUTO:
          return {}
        default:
          return {
            [`${value.toLowerCase()}-unet`]: true
          }
      }
    }
  },

  // VAE settings
  {
    id: 'vae-precision',
    name: 'VAE precision',
    category: ['Inference'],
    type: 'combo',
    options: [
      FloatingPointPrecision.AUTO,
      FloatingPointPrecision.FP16,
      FloatingPointPrecision.FP32,
      FloatingPointPrecision.BF16
    ],
    defaultValue: FloatingPointPrecision.AUTO,
    tooltip: 'VAE precision',
    getValue: (value: FloatingPointPrecision) => {
      switch (value) {
        case FloatingPointPrecision.AUTO:
          return {}
        default:
          return {
            [`${value.toLowerCase()}-vae`]: true
          }
      }
    }
  },
  {
    id: 'cpu-vae',
    name: 'Run VAE on CPU',
    category: ['Inference'],
    type: 'boolean',
    defaultValue: false
  },

  // Text Encoder settings
  {
    id: 'text-encoder-precision',
    name: 'Text Encoder precision',
    category: ['Inference'],
    type: 'combo',
    options: [
      FloatingPointPrecision.AUTO,
      FloatingPointPrecision.FP8E4M3FN,
      FloatingPointPrecision.FP8E5M2,
      FloatingPointPrecision.FP16,
      FloatingPointPrecision.FP32
    ],
    defaultValue: FloatingPointPrecision.AUTO,
    tooltip: 'Text Encoder precision',
    getValue: (value: FloatingPointPrecision) => {
      switch (value) {
        case FloatingPointPrecision.AUTO:
          return {}
        default:
          return {
            [`${value.toLowerCase()}-text-enc`]: true
          }
      }
    }
  },

  // Memory and performance settings
  {
    id: 'force-channels-last',
    name: 'Force channels-last memory format',
    category: ['Memory'],
    type: 'boolean',
    defaultValue: false
  },
  {
    id: 'directml',
    name: 'DirectML device index',
    category: ['Memory'],
    type: 'number',
    defaultValue: null
  },
  {
    id: 'disable-ipex-optimize',
    name: 'Disable IPEX optimization',
    category: ['Memory'],
    type: 'boolean',
    defaultValue: false
  },

  // Preview settings
  {
    id: 'preview-method',
    name: 'Method used for latent previews',
    category: ['Preview'],
    type: 'combo',
    options: Object.values(LatentPreviewMethod),
    defaultValue: LatentPreviewMethod.NoPreviews
  },
  {
    id: 'preview-size',
    name: 'Size of preview images',
    category: ['Preview'],
    type: 'slider',
    defaultValue: 512,
    attrs: {
      min: 128,
      max: 2048,
      step: 128
    }
  },

  // Cache settings
  {
    id: 'cache-classic',
    name: 'Use classic cache system',
    category: ['Cache'],
    type: 'boolean',
    defaultValue: false
  },
  {
    id: 'cache-lru',
    name: 'Use LRU caching with a maximum of N node results cached.',
    category: ['Cache'],
    type: 'number',
    defaultValue: null,
    tooltip: 'May use more RAM/VRAM.'
  },

  // Attention settings
  {
    id: 'cross-attention-method',
    name: 'Cross attention method',
    category: ['Attention'],
    type: 'combo',
    options: Object.values(CrossAttentionMethod),
    defaultValue: CrossAttentionMethod.Auto,
    getValue: (value: CrossAttentionMethod) => {
      switch (value) {
        case CrossAttentionMethod.Auto:
          return {}
        default:
          return {
            [`use-${value.toLowerCase()}-cross-attention`]: true
          }
      }
    }
  },
  {
    id: 'disable-xformers',
    name: 'Disable xFormers optimization',
    type: 'boolean',
    defaultValue: false
  },
  {
    id: 'force-upcast-attention',
    name: 'Force attention upcast',
    category: ['Attention'],
    type: 'boolean',
    defaultValue: false
  },
  {
    id: 'dont-upcast-attention',
    name: 'Prevent attention upcast',
    category: ['Attention'],
    type: 'boolean',
    defaultValue: false
  },

  // VRAM management
  {
    id: 'vram-management',
    name: 'VRAM management mode',
    category: ['Memory'],
    type: 'combo',
    options: Object.values(VramManagement),
    defaultValue: VramManagement.Auto,
    getValue: (value: VramManagement) => {
      switch (value) {
        case VramManagement.Auto:
          return {}
        default:
          return {
            [value]: true
          }
      }
    }
  },
  {
    id: 'reserve-vram',
    name: 'Reserved VRAM (GB)',
    category: ['Memory'],
    type: 'number',
    defaultValue: null,
    tooltip:
      'Set the amount of vram in GB you want to reserve for use by your OS/other software. By default some amount is reserved depending on your OS.'
  },

  // Misc settings
  {
    id: 'default-hashing-function',
    name: 'Default hashing function for model files',
    type: 'combo',
    options: Object.values(HashFunction),
    defaultValue: HashFunction.SHA256
  },
  {
    id: 'disable-smart-memory',
    name: 'Disable smart memory management',
    tooltip:
      'Force ComfyUI to aggressively offload to regular ram instead of keeping models in vram when it can.',
    category: ['Memory'],
    type: 'boolean',
    defaultValue: false
  },
  {
    id: 'deterministic',
    name: 'Make pytorch use slower deterministic algorithms when it can.',
    type: 'boolean',
    defaultValue: false,
    tooltip: 'Note that this might not make images deterministic in all cases.'
  },
  {
    id: 'fast',
    name: 'Enable some untested and potentially quality deteriorating optimizations.',
    type: 'boolean',
    defaultValue: false
  },
  {
    id: 'dont-print-server',
    name: "Don't print server output to console.",
    type: 'boolean',
    defaultValue: false
  },
  {
    id: 'disable-metadata',
    name: 'Disable saving prompt metadata in files.',
    type: 'boolean',
    defaultValue: false
  },
  {
    id: 'enable-manager-legacy-ui',
    name: 'Use legacy Manager UI',
    tooltip: 'Uses the legacy ComfyUI-Manager UI instead of the new UI.',
    type: 'boolean',
    defaultValue: false
  },
  {
    id: 'disable-all-custom-nodes',
    name: 'Disable loading all custom nodes.',
    type: 'boolean',
    defaultValue: false
  },
  {
    id: 'log-level',
    name: 'Logging verbosity level',
    type: 'combo',
    options: Object.values(LogLevel),
    defaultValue: LogLevel.INFO,
    getValue: (value: LogLevel) => {
      return {
        verbose: value
      }
    }
  },
  // Directories
  {
    id: 'input-directory',
    name: 'Input directory',
    category: ['Directories'],
    type: 'text',
    defaultValue: ''
  },
  {
    id: 'output-directory',
    name: 'Output directory',
    category: ['Directories'],
    type: 'text',
    defaultValue: ''
  }
]
