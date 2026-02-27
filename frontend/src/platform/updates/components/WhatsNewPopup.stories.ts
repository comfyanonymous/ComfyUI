import type { Meta, StoryObj } from '@storybook/vue3-vite'

import type { ReleaseNote } from '../common/releaseService'
import { useReleaseStore } from '../common/releaseStore'
import WhatsNewPopup from './WhatsNewPopup.vue'

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
    content: `![Featured Image](https://images.unsplash.com/photo-1633356122544-f134324a6cee?w=400&h=200&fit=crop&fm=jpg)

# ComfyUI 1.2.4 Major Release

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
    content: `![Release Image](https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=400&h=200&fit=crop&fm=jpg)

# ComfyUI 1.3.0 - The Biggest Update Yet

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
  title: 'Platform/Updates/WhatsNewPopup',
  component: WhatsNewPopup,
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
    (_story, context) => {
      // Set up the store with mock data for this story
      const releaseStore = useReleaseStore()

      // Override store data with story args
      releaseStore.releases = [context.args.releaseData]

      // Force the computed properties to return the values we want
      Object.defineProperty(releaseStore, 'recentRelease', {
        value: context.args.releaseData,
        writable: true
      })
      Object.defineProperty(releaseStore, 'shouldShowPopup', {
        value: true,
        writable: true
      })

      // Mock the store methods to prevent errors
      releaseStore.handleWhatsNewSeen = async () => {
        // Mock implementation for Storybook
      }

      return {
        template: `
          <div class="min-h-screen flex items-center justify-center bg-gray-900 p-8">
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

export const WithImage: Story = {
  args: {
    releaseData: mockReleases[1]
  }
}

export const MajorRelease: Story = {
  args: {
    releaseData: mockReleases[2]
  }
}

export const LongContent: Story = {
  args: {
    releaseData: {
      id: 4,
      project: 'comfyui',
      version: '2.0.0',
      attention: 'high',
      published_at: '2024-04-20T16:00:00Z',
      content: `![Major Update](https://images.unsplash.com/photo-1633356122544-f134324a6cee?w=400&h=200&fit=crop)

# ComfyUI 2.0.0 - Complete Rewrite

**What's new**

The most significant update in ComfyUI history with complete platform rewrite.

## Core Engine Improvements

- **Next-Generation Workflow Engine**: Completely rewritten from the ground up with 500% performance improvements for complex workflows
- **Advanced Memory Management**: Intelligent memory allocation reducing VRAM usage by up to 60% while maintaining quality
- **Multi-Threading Support**: Full multi-core CPU utilization for preprocessing and post-processing tasks
- **GPU Optimization**: Advanced GPU scheduling with automatic optimization for different hardware configurations

## New User Interface

- **Modern Design Language**: Beautiful new interface with improved accessibility and mobile responsiveness
- **Customizable Workspace**: Fully customizable layout with dockable panels and saved workspace configurations
- **Advanced Node Browser**: Intelligent node search with AI-powered suggestions and visual node previews
- **Real-time Preview**: Live preview of changes as you build your workflow without needing to execute

## Professional Features

- **Version Control Integration**: Native Git integration for workflow version control and collaboration
- **Enterprise Security**: Advanced security features including end-to-end encryption and audit logging
- **Scalable Architecture**: Designed to handle enterprise-scale deployments with thousands of concurrent users
- **Plugin Ecosystem**: Robust plugin system with hot-loading and automatic dependency management`
    }
  }
}

export const MinimalContent: Story = {
  args: {
    releaseData: {
      id: 5,
      project: 'comfyui',
      version: '1.0.1',
      attention: 'low',
      published_at: '2024-01-05T12:00:00Z',
      content: `# ComfyUI 1.0.1

**What's new**

Quick patch release.

- **Bug Fix**: Fixed critical save issue`
    }
  }
}

export const EmptyContent: Story = {
  args: {
    releaseData: {
      id: 6,
      project: 'comfyui',
      version: '1.0.0',
      attention: 'low',
      published_at: '2024-01-01T00:00:00Z',
      content: ''
    }
  }
}
