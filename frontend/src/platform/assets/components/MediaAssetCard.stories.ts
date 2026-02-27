import type { Meta, StoryObj } from '@storybook/vue3-vite'

import ResultGallery from '@/components/sidebar/tabs/queue/ResultGallery.vue'

import { useMediaAssetGalleryStore } from '../composables/useMediaAssetGalleryStore'
import type { AssetItem } from '../schemas/assetSchema'
import MediaAssetCard from './MediaAssetCard.vue'

const meta: Meta<typeof MediaAssetCard> = {
  title: 'Platform/Assets/MediaAssetCard',
  component: MediaAssetCard,
  decorators: [
    () => ({
      components: { ResultGallery },
      setup() {
        const galleryStore = useMediaAssetGalleryStore()
        return { galleryStore }
      },
      template: `
        <div>
          <story />
          <ResultGallery 
            v-model:active-index="galleryStore.activeIndex"
            :all-gallery-items="galleryStore.items"
          />
        </div>
      `
    })
  ],
  argTypes: {
    loading: {
      control: 'boolean'
    }
  }
}

export default meta
type Story = StoryObj<typeof meta>

// Public sample media URLs
const SAMPLE_MEDIA = {
  image1: 'https://i.imgur.com/OB0y6MR.jpg',
  image2: 'https://i.imgur.com/CzXTtJV.jpg',
  image3: 'https://farm9.staticflickr.com/8505/8441256181_4e98d8bff5_z_d.jpg',
  video:
    'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4',
  videoThumbnail:
    'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/images/BigBuckBunny.jpg',
  audio: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3'
}

const sampleAsset: AssetItem = {
  id: 'asset-1',
  name: 'sample-image.png',
  size: 2048576,
  created_at: new Date().toISOString(),
  preview_url: SAMPLE_MEDIA.image1,
  tags: ['input'],
  user_metadata: {
    duration: 3345,
    dimensions: {
      width: 1920,
      height: 1080
    }
  }
}

export const ImageAsset: Story = {
  decorators: [
    () => ({
      template: '<div style="max-width: 280px;"><story /></div>'
    })
  ],
  args: {
    asset: sampleAsset,
    loading: false
  }
}

export const VideoAsset: Story = {
  decorators: [
    () => ({
      template: '<div style="max-width: 280px;"><story /></div>'
    })
  ],
  args: {
    asset: {
      ...sampleAsset,
      id: 'asset-2',
      name: 'Big_Buck_Bunny.mp4',
      size: 10485760,
      preview_url: SAMPLE_MEDIA.videoThumbnail,
      user_metadata: {
        duration: 13425,
        dimensions: {
          width: 1280,
          height: 720
        }
      }
    }
  }
}

export const Model3DAsset: Story = {
  decorators: [
    () => ({
      template: '<div style="max-width: 280px;"><story /></div>'
    })
  ],
  args: {
    asset: {
      ...sampleAsset,
      id: 'asset-3',
      name: 'Asset-3d-model.glb',
      size: 7340032,
      preview_url: '',
      user_metadata: {
        duration: 18023
      }
    }
  }
}

export const AudioAsset: Story = {
  decorators: [
    () => ({
      template: '<div style="max-width: 280px;"><story /></div>'
    })
  ],
  args: {
    asset: {
      ...sampleAsset,
      id: 'asset-4',
      name: 'SoundHelix-Song.mp3',
      size: 5242880,
      preview_url: SAMPLE_MEDIA.audio,
      user_metadata: {
        duration: 23180
      }
    }
  }
}

export const TextAsset: Story = {
  decorators: [
    () => ({
      template: '<div style="max-width: 280px;"><story /></div>'
    })
  ],
  args: {
    asset: {
      ...sampleAsset,
      id: 'asset-5',
      name: 'generation-notes.txt',
      size: 2048,
      preview_url: SAMPLE_MEDIA.image1
    }
  }
}

export const OtherAsset: Story = {
  decorators: [
    () => ({
      template: '<div style="max-width: 280px;"><story /></div>'
    })
  ],
  args: {
    asset: {
      ...sampleAsset,
      id: 'asset-6',
      name: 'workflow-payload.bin',
      size: 8192,
      preview_url: SAMPLE_MEDIA.image1
    }
  }
}

export const LoadingState: Story = {
  decorators: [
    () => ({
      template: '<div style="max-width: 280px;"><story /></div>'
    })
  ],
  args: {
    asset: sampleAsset,
    loading: true
  }
}

export const LongFileName: Story = {
  decorators: [
    () => ({
      template: '<div style="max-width: 280px;"><story /></div>'
    })
  ],
  args: {
    asset: {
      ...sampleAsset,
      name: 'very-long-file-name-that-should-be-truncated-in-the-ui-to-prevent-overflow.png'
    }
  }
}

export const SelectedState: Story = {
  decorators: [
    () => ({
      template: '<div style="max-width: 280px;"><story /></div>'
    })
  ],
  args: {
    asset: sampleAsset,
    selected: true
  }
}

export const WebMVideo: Story = {
  decorators: [
    () => ({
      template: '<div style="max-width: 280px;"><story /></div>'
    })
  ],
  args: {
    asset: {
      id: 'asset-webm',
      name: 'animated-clip.webm',
      size: 3145728,
      created_at: new Date().toISOString(),
      preview_url: SAMPLE_MEDIA.image1,
      tags: ['input'],
      user_metadata: {
        duration: 620,
        dimensions: {
          width: 640,
          height: 360
        }
      }
    }
  }
}

export const GifAnimation: Story = {
  decorators: [
    () => ({
      template: '<div style="max-width: 280px;"><story /></div>'
    })
  ],
  args: {
    asset: {
      id: 'asset-gif',
      name: 'animation.gif',
      size: 1572864,
      created_at: new Date().toISOString(),
      preview_url: 'https://media.giphy.com/media/3o7aCTPPm4OHfRLSH6/giphy.gif',
      tags: ['input'],
      user_metadata: {
        duration: 1345,
        dimensions: {
          width: 480,
          height: 270
        }
      }
    }
  }
}

export const GridLayout: Story = {
  render: () => ({
    components: { MediaAssetCard },
    setup() {
      const assets: AssetItem[] = [
        {
          id: 'grid-1',
          name: 'image-file.jpg',
          size: 2097152,
          created_at: new Date().toISOString(),
          preview_url: SAMPLE_MEDIA.image1,
          tags: ['input'],
          user_metadata: {
            duration: 4500,
            dimensions: { width: 1920, height: 1080 }
          }
        },
        {
          id: 'grid-2',
          name: 'image-file.jpg',
          size: 2097152,
          created_at: new Date().toISOString(),
          preview_url: SAMPLE_MEDIA.image2,
          tags: ['input'],
          user_metadata: {
            duration: 4500,
            dimensions: { width: 1920, height: 1080 }
          }
        },
        {
          id: 'grid-3',
          name: 'video-file.mp4',
          size: 10485760,
          created_at: new Date().toISOString(),
          preview_url: SAMPLE_MEDIA.videoThumbnail,
          tags: ['input'],
          user_metadata: {
            duration: 13425,
            dimensions: { width: 1280, height: 720 }
          }
        },
        {
          id: 'grid-4',
          name: 'audio-file.mp3',
          size: 5242880,
          created_at: new Date().toISOString(),
          preview_url: SAMPLE_MEDIA.audio,
          tags: ['input'],
          user_metadata: {
            duration: 180
          }
        },
        {
          id: 'grid-5',
          name: 'animation.gif',
          size: 3145728,
          created_at: new Date().toISOString(),
          preview_url:
            'https://media.giphy.com/media/l0HlNaQ6gWfllcjDO/giphy.gif',
          tags: ['input'],
          user_metadata: {
            duration: 1345,
            dimensions: { width: 480, height: 360 }
          }
        },
        {
          id: 'grid-6',
          name: 'Asset-3d-model.glb',
          size: 7340032,
          preview_url: '',
          created_at: new Date().toISOString(),
          tags: ['input'],
          user_metadata: {
            duration: 18023
          }
        },
        {
          id: 'grid-7',
          name: 'image-file.jpg',
          size: 2097152,
          created_at: new Date().toISOString(),
          preview_url: SAMPLE_MEDIA.image3,
          tags: ['input'],
          user_metadata: {
            duration: 4500,
            dimensions: { width: 1920, height: 1080 }
          }
        }
      ]
      return { assets }
    },
    template: `
      <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 16px; padding: 16px;">
        <MediaAssetCard
          v-for="asset in assets"
          :key="asset.id"
          :asset="asset"
        />
      </div>
    `
  })
}
