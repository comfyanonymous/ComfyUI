import type { Meta, StoryObj } from '@storybook/vue3-vite'

import type { ReleaseNote } from '../common/releaseService'
import { useReleaseStore } from '../common/releaseStore'
import ReleaseNotificationToast from './ReleaseNotificationToast.vue'

// Mock release data with realistic CMS content
const mockReleases: ReleaseNote[] = [
  {
    id: 1,
    project: 'comfyui',
    version: '1.2.3',
    attention: 'medium',
    published_at: '2024-01-15T10:00:00Z',
    content: `# ComfyUI 1.2.3 Release

**What's new**

New features and improvements for better workflow management.

- **Enhanced Node Editor**: Improved performance for large workflows with 100+ nodes
- **Auto-save Feature**: Your work is now automatically saved every 30 seconds 
- **New Model Support**: Added support for FLUX.1-dev and FLUX.1-schnell models
- **Bug Fixes**: Resolved memory leak issues in the backend processing`
  },
  {
    id: 2,
    project: 'comfyui',
    version: '1.2.4',
    attention: 'high',
    published_at: '2024-02-01T14:30:00Z',
    content: `# ComfyUI 1.2.4 Major Release

**What's new**

Revolutionary updates that change how you create with ComfyUI.

- **Real-time Collaboration**: Share and edit workflows with your team in real-time
- **Advanced Upscaling**: New ESRGAN and Real-ESRGAN models built-in
- **Custom Node Store**: Browse and install community nodes directly from the interface
- **Performance Boost**: 40% faster generation times for SDXL models
- **Dark Mode**: Beautiful new dark interface theme`
  },
  {
    id: 3,
    project: 'comfyui',
    version: '1.3.0',
    attention: 'high',
    published_at: '2024-03-10T09:15:00Z',
    content: `# ComfyUI 1.3.0 - The Biggest Update Yet

**What's new**

Introducing powerful new features that unlock creative possibilities.

- **AI-Powered Node Suggestions**: Get intelligent recommendations while building workflows
- **Workflow Templates**: Start from professionally designed templates 
- **Advanced Queuing**: Batch process multiple generations with queue management
- **Mobile Preview**: Preview your workflows on mobile devices
- **API Improvements**: Enhanced REST API with better documentation
- **Community Hub**: Share workflows and discover creations from other users`
  }
]

interface StoryArgs {
  releaseData: ReleaseNote
}

const meta: Meta<StoryArgs> = {
  title: 'Platform/Updates/ReleaseNotificationToast',
  component: ReleaseNotificationToast,
  parameters: {
    layout: 'fullscreen',
    backgrounds: { default: 'dark' }
  },
  argTypes: {
    releaseData: {
      control: 'object',
      description: 'Release data with version and markdown content'
    }
  },
  decorators: [
    (_, context) => {
      // Set up the store with mock data for this story
      const releaseStore = useReleaseStore()

      // Patch store state directly for Storybook
      releaseStore.$patch({
        releases: [context.args.releaseData]
      })
      // Override shouldShowToast getter for Storybook
      Object.defineProperty(releaseStore, 'shouldShowToast', {
        get: () => true,
        configurable: true
      })
      // Override recentRelease getter for Storybook
      Object.defineProperty(releaseStore, 'recentRelease', {
        get: () => context.args.releaseData,
        configurable: true
      })

      // Mock the store methods to prevent errors
      releaseStore.handleSkipRelease = async () => {
        // Mock implementation for Storybook
      }
      releaseStore.handleShowChangelog = async () => {
        // Mock implementation for Storybook
      }

      return {
        template: `
          <div class="min-h-screen flex items-center justify-center bg-base-background p-8">
            <story />
          </div>
        `
      }
    }
  ]
}

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    releaseData: mockReleases[0]
  }
}

export const MajorRelease: Story = {
  args: {
    releaseData: mockReleases[1]
  }
}

export const ExtensiveFeatures: Story = {
  args: {
    releaseData: mockReleases[2]
  }
}

export const LongContent: Story = {
  args: {
    releaseData: {
      id: 4,
      project: 'comfyui',
      version: '1.4.0',
      attention: 'high',
      published_at: '2024-04-05T11:00:00Z',
      content: `# ComfyUI 1.4.0 - Comprehensive Update

**What's new**

This is a comprehensive update with many new features and improvements. This release includes extensive changes across the entire platform.

- **Revolutionary Workflow Engine**: Complete rewrite of the workflow processing engine with 300% performance improvements
- **Advanced Model Management**: Sophisticated model organization with tagging, favorites, and automatic duplicate detection
- **Real-time Collaboration Suite**: Complete collaboration platform with user management, permissions, and shared workspaces
- **Professional Animation Tools**: Timeline-based animation system with keyframes and interpolation
- **Cloud Integration**: Seamless cloud storage integration with automatic backup and sync
- **Advanced Debugging Tools**: Comprehensive debugging suite with step-through execution and variable inspection`
    }
  }
}

export const EmptyContent: Story = {
  args: {
    releaseData: {
      id: 5,
      project: 'comfyui',
      version: '1.0.0',
      attention: 'low',
      published_at: '2024-01-01T00:00:00Z',
      content: ''
    }
  }
}
