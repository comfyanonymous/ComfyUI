import type { Meta, StoryObj } from '@storybook/vue3-vite'

import type { ComfyNodeDef as ComfyNodeDefV2 } from '@/schemas/nodeDef/nodeDefSchemaV2'

import NodePreview from './NodePreview.vue'

// Mock node definition data for a typical ComfyUI node
const mockNodeDef: ComfyNodeDefV2 = {
  name: 'KSampler',
  display_name: 'KSampler',
  description:
    'The main sampler node for generating images. Controls the denoising process using various samplers and schedulers.',
  category: 'sampling',
  output_node: false,
  python_module: 'nodes',
  inputs: {
    model: {
      type: 'MODEL',
      name: 'model',
      isOptional: false
    },
    positive: {
      type: 'CONDITIONING',
      name: 'positive',
      isOptional: false
    },
    negative: {
      type: 'CONDITIONING',
      name: 'negative',
      isOptional: false
    },
    latent_image: {
      type: 'LATENT',
      name: 'latent_image',
      isOptional: false
    },
    seed: {
      type: 'INT',
      name: 'seed',
      default: 0,
      min: 0,
      max: Number.MAX_SAFE_INTEGER,
      isOptional: false
    },
    steps: {
      type: 'INT',
      name: 'steps',
      default: 20,
      min: 1,
      max: 10000,
      isOptional: false
    },
    cfg: {
      type: 'FLOAT',
      name: 'cfg',
      default: 8.0,
      min: 0.0,
      max: 100.0,
      step: 0.1,
      isOptional: false
    },
    sampler_name: {
      type: 'COMBO',
      name: 'sampler_name',
      options: [
        'euler',
        'euler_ancestral',
        'heun',
        'dpm_2',
        'dpm_2_ancestral',
        'lms',
        'dpm_fast',
        'dpm_adaptive',
        'dpmpp_2s_ancestral',
        'dpmpp_sde',
        'dpmpp_2m'
      ],
      default: 'euler',
      isOptional: false
    },
    scheduler: {
      type: 'COMBO',
      name: 'scheduler',
      options: [
        'normal',
        'karras',
        'exponential',
        'sgm_uniform',
        'simple',
        'ddim_uniform'
      ],
      default: 'normal',
      isOptional: false
    },
    denoise: {
      type: 'FLOAT',
      name: 'denoise',
      default: 1.0,
      min: 0.0,
      max: 1.0,
      step: 0.01,
      isOptional: false
    }
  },
  outputs: [
    {
      index: 0,
      name: 'LATENT',
      type: 'LATENT',
      is_list: false
    }
  ]
}

// Simpler text node for comparison
const mockTextNodeDef: ComfyNodeDefV2 = {
  name: 'CLIPTextEncode',
  display_name: 'CLIP Text Encode',
  description:
    'Encode text using CLIP to create conditioning for image generation.',
  category: 'conditioning',
  output_node: false,
  python_module: 'nodes',
  inputs: {
    clip: {
      type: 'CLIP',
      name: 'clip',
      isOptional: false
    },
    text: {
      type: 'STRING',
      name: 'text',
      multiline: true,
      default: '',
      isOptional: false
    }
  },
  outputs: [
    {
      index: 0,
      name: 'CONDITIONING',
      type: 'CONDITIONING',
      is_list: false
    }
  ]
}

// Image processing node with multiple outputs
const mockImageNodeDef: ComfyNodeDefV2 = {
  name: 'VAEDecode',
  display_name: 'VAE Decode',
  description:
    'Decode latent representation back to image using a Variational Autoencoder.',
  category: 'latent',
  output_node: false,
  python_module: 'nodes',
  inputs: {
    samples: {
      type: 'LATENT',
      name: 'samples',
      isOptional: false
    },
    vae: {
      type: 'VAE',
      name: 'vae',
      isOptional: false
    }
  },
  outputs: [
    {
      index: 0,
      name: 'IMAGE',
      type: 'IMAGE',
      is_list: false
    }
  ]
}

const meta: Meta<typeof NodePreview> = {
  title: 'Components/Node/NodePreview',
  component: NodePreview,
  parameters: {
    layout: 'centered',
    docs: {
      description: {
        component:
          'NodePreview displays a preview of a ComfyUI node with its inputs, outputs, and parameters. This component is used to show node information in sidebars and tooltips.'
      }
    }
  },
  argTypes: {
    nodeDef: {
      description: 'Node definition object containing all node metadata',
      control: { type: 'object' }
    }
  },
  tags: ['autodocs']
}

export default meta
type Story = StoryObj<typeof NodePreview>

export const KSamplerNode: Story = {
  args: {
    nodeDef: mockNodeDef
  },
  parameters: {
    docs: {
      description: {
        story:
          'KSampler node - the main sampling node with multiple inputs including model, conditioning, and sampling parameters.'
      }
    }
  }
}

export const TextEncodeNode: Story = {
  args: {
    nodeDef: mockTextNodeDef
  },
  parameters: {
    docs: {
      description: {
        story:
          'CLIP Text Encode node - simple text input node for creating conditioning.'
      }
    }
  }
}

export const ImageProcessingNode: Story = {
  args: {
    nodeDef: mockImageNodeDef
  },
  parameters: {
    docs: {
      description: {
        story:
          'VAE Decode node - converts latent representation back to images.'
      }
    }
  }
}

export const WithLongDescription: Story = {
  args: {
    nodeDef: {
      ...mockNodeDef,
      description:
        'This is an example of a node with a very long description that should wrap properly and display in the description area below the node preview. It demonstrates how the component handles extensive documentation text and formatting. The description supports **markdown** formatting including *italics* and other styling elements.'
    }
  },
  parameters: {
    docs: {
      description: {
        story:
          'Node preview with a longer, markdown-formatted description to test text wrapping and styling.'
      }
    }
  }
}
