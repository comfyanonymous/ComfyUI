import type { Meta, StoryObj } from '@storybook/vue3-vite'
import { computed, provide, ref } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import MoreButton from '@/components/button/MoreButton.vue'
import CardBottom from '@/components/card/CardBottom.vue'
import CardContainer from '@/components/card/CardContainer.vue'
import CardTop from '@/components/card/CardTop.vue'
import SquareChip from '@/components/chip/SquareChip.vue'
import MultiSelect from '@/components/input/MultiSelect.vue'
import SearchBox from '@/components/common/SearchBox.vue'
import SingleSelect from '@/components/input/SingleSelect.vue'
import type { NavGroupData, NavItemData } from '@/types/navTypes'
import { OnCloseKey } from '@/types/widgetTypes'
import { createGridStyle } from '@/utils/gridUtil'

import LeftSidePanel from '../panel/LeftSidePanel.vue'
import BaseModalLayout from './BaseModalLayout.vue'

interface StoryArgs {
  contentTitle: string
  hasLeftPanel: boolean
  hasRightPanel: boolean
  hasHeader: boolean
  hasContentFilter: boolean
  hasHeaderRightArea: boolean
  cardCount: number
}

const meta: Meta<StoryArgs> = {
  title: 'Components/Widget/Layout/BaseModalLayout',
  argTypes: {
    contentTitle: {
      control: 'text',
      description: 'Title shown when no left panel is present'
    },
    hasLeftPanel: {
      control: 'boolean',
      description: 'Toggle left panel visibility'
    },
    hasRightPanel: {
      control: 'boolean',
      description: 'Toggle right panel visibility'
    },
    hasHeader: {
      control: 'boolean',
      description: 'Toggle header visibility'
    },
    hasContentFilter: {
      control: 'boolean',
      description: 'Toggle content filter visibility'
    },
    hasHeaderRightArea: {
      control: 'boolean',
      description: 'Toggle header right area visibility'
    },
    cardCount: {
      control: { type: 'range', min: 0, max: 50, step: 1 },
      description: 'Number of cards to display in content'
    }
  }
}

export default meta
type Story = StoryObj<typeof meta>

const createStoryTemplate = (args: StoryArgs) => ({
  components: {
    BaseModalLayout,
    LeftSidePanel,
    SearchBox,
    MultiSelect,
    SingleSelect,
    Button,
    MoreButton,
    CardContainer,
    CardTop,
    CardBottom,
    SquareChip
  },
  setup() {
    const t = (k: string) => k

    const onClose = () => {
      // OnClose handler for story
    }
    provide(OnCloseKey, onClose)

    const tempNavigation = ref<(NavItemData | NavGroupData)[]>([
      {
        id: 'installed',
        label: 'Installed',
        icon: 'icon-[lucide--folder]'
      },
      {
        title: 'TAGS',
        items: [
          {
            id: 'tag-sd15',
            label: 'SD 1.5',
            icon: 'icon-[lucide--tag]'
          },
          {
            id: 'tag-sdxl',
            label: 'SDXL',
            icon: 'icon-[lucide--tag]'
          },
          {
            id: 'tag-utility',
            label: 'Utility',
            icon: 'icon-[lucide--tag]'
          }
        ]
      },
      {
        title: 'CATEGORIES',
        items: [
          {
            id: 'cat-models',
            label: 'Models',
            icon: 'icon-[lucide--layers]'
          },
          {
            id: 'cat-nodes',
            label: 'Nodes',
            icon: 'icon-[lucide--grid-3x3]'
          }
        ]
      }
    ])
    const selectedNavItem = ref<string | null>('installed')

    const searchQuery = ref<string>('')

    const frameworkOptions = ref([
      { name: 'Vue', value: 'vue' },
      { name: 'React', value: 'react' },
      { name: 'Angular', value: 'angular' },
      { name: 'Svelte', value: 'svelte' }
    ])
    const projectOptions = ref([
      { name: 'Project A', value: 'proj-a' },
      { name: 'Project B', value: 'proj-b' },
      { name: 'Project C', value: 'proj-c' }
    ])
    const sortOptions = ref([
      { name: 'Popular', value: 'popular' },
      { name: 'Latest', value: 'latest' },
      { name: 'A → Z', value: 'az' }
    ])

    const selectedFrameworks = ref<string[]>([])
    const selectedProjects = ref<string[]>([])
    const selectedSort = ref<string>('popular')

    const gridStyle = computed(() => createGridStyle())

    return {
      args,
      t,
      tempNavigation,
      selectedNavItem,
      searchQuery,
      frameworkOptions,
      projectOptions,
      sortOptions,
      selectedFrameworks,
      selectedProjects,
      selectedSort,
      gridStyle
    }
  },
  template: `
    <div>
      <BaseModalLayout v-if="!args.hasRightPanel" :content-title="args.contentTitle || 'Content Title'">
        <!-- Left Panel Header Title -->
        <template v-if="args.hasLeftPanel" #leftPanelHeaderTitle>
          <i class="icon-[lucide--puzzle] size-4 text-neutral" />
          <span class="text-neutral text-base">Title</span>
        </template>

        <!-- Left Panel -->
        <template v-if="args.hasLeftPanel" #leftPanel>
          <LeftSidePanel v-model="selectedNavItem" :nav-items="tempNavigation" />
        </template>

        <!-- Header -->
        <template v-if="args.hasHeader" #header>
          <SearchBox
            class="max-w-[384px]"
            size="lg"
            :modelValue="searchQuery"
            @update:modelValue="searchQuery = $event"
          />
        </template>

        <!-- Header Right Area -->
        <template v-if="args.hasHeaderRightArea" #header-right-area>
          <div class="flex gap-2">
            <Button variant="primary" @click="() => {}">
                <i class="icon-[lucide--upload] size-3" />
              <span> Upload Model </span>
            </Button>

            <MoreButton>
              <template #default="{ close }">
                <Button
                  variant="secondary"
                  label="Settings"
                  @click="() => { close() }"
                >
                  <template #icon>
                    <i class="icon-[lucide--download] size-3" />
                  </template>
                </Button>

                <Button
                  variant="primary"
                  label="Profile"
                  @click="() => { close() }"
                >
                  <template #icon>
                    <i class="icon-[lucide--scroll] size-3" />
                  </template>
                </Button>
              </template>
            </MoreButton>
          </div>
        </template>

        <!-- Content Filter -->
        <template v-if="args.hasContentFilter" #contentFilter>
          <div class="relative px-6 py-4 flex gap-2">
            <MultiSelect
              v-model="selectedFrameworks"
              label="Select Frameworks"
              :options="frameworkOptions"
              :has-search-box="true"
              :show-selected-count="true"
              :has-clear-button="true"
            />
            <MultiSelect
              v-model="selectedProjects"
              label="Select Projects"
              :options="projectOptions"
            />
            <SingleSelect
              v-model="selectedSort"
              label="Sorting Type"
              :options="sortOptions"
              class="w-[135px]"
            >
              <template #icon>
                <i class="icon-[lucide--filter] size-3" />
              </template>
            </SingleSelect>
          </div>
        </template>

        <!-- Content -->
        <template #content>
          <div :style="gridStyle">
            <CardContainer
              v-for="i in args.cardCount"
              :key="i"
              ratio="square"
            >
              <template #top>
                <CardTop ratio="landscape">
                  <template #default>
                    <div class="w-full h-full bg-blue-500"></div>
                  </template>
                  <template #top-right>
                    <Button size="icon" class="!bg-white !text-neutral-900" @click="() => {}">
                      <i class="icon-[lucide--info] size-4" />
                    </Button>
                  </template>
                  <template #bottom-right>
                    <SquareChip label="png" />
                    <SquareChip label="1.2 MB" />
                    <SquareChip label="LoRA">
                      <template #icon>
                        <i class="icon-[lucide--folder] size-3" />
                      </template>
                    </SquareChip>
                  </template>
                </CardTop>
              </template>
              <template #bottom>
                <CardBottom />
              </template>
            </CardContainer>
          </div>
        </template>
      </BaseModalLayout>

      <BaseModalLayout v-else :content-title="args.contentTitle || 'Content Title'">
        <!-- Same content but WITH right panel -->
        <!-- Left Panel Header Title -->
        <template v-if="args.hasLeftPanel" #leftPanelHeaderTitle>
          <i class="icon-[lucide--puzzle] size-4 text-neutral" />
          <span class="text-neutral text-base">Title</span>
        </template>

        <!-- Left Panel -->
        <template v-if="args.hasLeftPanel" #leftPanel>
          <LeftSidePanel v-model="selectedNavItem" :nav-items="tempNavigation" />
        </template>

        <!-- Header -->
        <template v-if="args.hasHeader" #header>
          <SearchBox
            class="max-w-[384px]"
            size="lg"
            :modelValue="searchQuery"
            @update:modelValue="searchQuery = $event"
          />
        </template>

        <!-- Header Right Area -->
        <template v-if="args.hasHeaderRightArea" #header-right-area>
          <div class="flex gap-2">
            <Button variant="primary" @click="() => {}">
                <i class="icon-[lucide--upload] size-3" />
                <span>Upload Model</span>
            </Button>

            <MoreButton>
              <template #default="{ close }">
                <Button
                  variant="secondary"
                  @click="() => { close() }"
                >
                    <i class="icon-[lucide--download] size-3" />
                    <span>Settings</span>
                </Button>

                <Button
                  variant="primary"
                  @click="() => { close() }"
                >
                    <i class="icon-[lucide--scroll] size-3" />
                    <span>Profile</span>
                </Button>
              </template>
            </MoreButton>
          </div>
        </template>

        <!-- Content Filter -->
        <template v-if="args.hasContentFilter" #contentFilter>
          <div class="relative px-6 py-4 flex gap-2">
            <MultiSelect
              v-model="selectedFrameworks"
              label="Select Frameworks"
              :options="frameworkOptions"
            />
            <MultiSelect
              v-model="selectedProjects"
              label="Select Projects"
              :options="projectOptions"
            />
            <SingleSelect
              v-model="selectedSort"
              label="Sorting Type"
              :options="sortOptions"
              class="w-[135px]"
            >
              <template #icon>
                <i class="icon-[lucide--filter] size-3" />
              </template>
            </SingleSelect>
          </div>
        </template>

        <!-- Content -->
        <template #content>
          <div :style="gridStyle">
            <CardContainer
              v-for="i in args.cardCount"
              :key="i"
              ratio="square"
            >
              <template #top>
                <CardTop ratio="landscape">
                  <template #default>
                    <div class="w-full h-full bg-blue-500"></div>
                  </template>
                  <template #top-right>
                    <Button size="icon" class="!bg-white !text-neutral-900" @click="() => {}">
                      <i class="icon-[lucide--info] size-4" />
                    </Button>
                  </template>
                  <template #bottom-right>
                    <SquareChip label="png" />
                    <SquareChip label="1.2 MB" />
                    <SquareChip label="LoRA">
                      <template #icon>
                        <i class="icon-[lucide--folder] size-3" />
                      </template>
                    </SquareChip>
                  </template>
                </CardTop>
              </template>
              <template #bottom>
                <CardBottom />
              </template>
            </CardContainer>
          </div>
        </template>

        <!-- Right Panel - Only when hasRightPanel is true -->
        <template #rightPanel>
          <div class="size-full bg-modal-panel-background pr-6 pb-8 pl-4"></div>
        </template>
      </BaseModalLayout>
    </div>
  `
})

export const Default: Story = {
  render: (args: StoryArgs) => createStoryTemplate(args),
  args: {
    contentTitle: 'Content Title',
    hasLeftPanel: true,
    hasRightPanel: true,
    hasHeader: true,
    hasContentFilter: true,
    hasHeaderRightArea: true,
    cardCount: 12
  }
}

export const BothPanels: Story = {
  render: (args: StoryArgs) => createStoryTemplate(args),
  args: {
    contentTitle: 'Content Title',
    hasLeftPanel: true,
    hasRightPanel: true,
    hasHeader: true,
    hasContentFilter: true,
    hasHeaderRightArea: true,
    cardCount: 12
  }
}

export const LeftPanelOnly: Story = {
  render: (args: StoryArgs) => createStoryTemplate(args),
  args: {
    contentTitle: 'Content Title',
    hasLeftPanel: true,
    hasRightPanel: false,
    hasHeader: true,
    hasContentFilter: true,
    hasHeaderRightArea: true,
    cardCount: 12
  }
}

export const RightPanelOnly: Story = {
  render: (args: StoryArgs) => createStoryTemplate(args),
  args: {
    contentTitle: 'Content Title',
    hasLeftPanel: false,
    hasRightPanel: true,
    hasHeader: true,
    hasContentFilter: true,
    hasHeaderRightArea: true,
    cardCount: 12
  }
}

export const NoPanels: Story = {
  render: (args: StoryArgs) => createStoryTemplate(args),
  args: {
    contentTitle: 'Content Title',
    hasLeftPanel: false,
    hasRightPanel: false,
    hasHeader: true,
    hasContentFilter: true,
    hasHeaderRightArea: true,
    cardCount: 12
  }
}

export const MinimalLayout: Story = {
  render: (args: StoryArgs) => createStoryTemplate(args),
  args: {
    contentTitle: 'Simple Content',
    hasLeftPanel: false,
    hasRightPanel: false,
    hasHeader: false,
    hasContentFilter: false,
    hasHeaderRightArea: false,
    cardCount: 6
  }
}

export const NoContent: Story = {
  render: (args: StoryArgs) => createStoryTemplate(args),
  args: {
    contentTitle: 'Empty State',
    hasLeftPanel: true,
    hasRightPanel: true,
    hasHeader: true,
    hasContentFilter: true,
    hasHeaderRightArea: true,
    cardCount: 0
  }
}

export const HeaderOnly: Story = {
  render: (args: StoryArgs) => createStoryTemplate(args),
  args: {
    contentTitle: 'Header Layout',
    hasLeftPanel: false,
    hasRightPanel: false,
    hasHeader: true,
    hasContentFilter: false,
    hasHeaderRightArea: true,
    cardCount: 8
  }
}

export const FilterOnly: Story = {
  render: (args: StoryArgs) => createStoryTemplate(args),
  args: {
    contentTitle: 'Filter Layout',
    hasLeftPanel: false,
    hasRightPanel: false,
    hasHeader: false,
    hasContentFilter: true,
    hasHeaderRightArea: false,
    cardCount: 8
  }
}

export const MaxContent: Story = {
  render: (args: StoryArgs) => createStoryTemplate(args),
  args: {
    contentTitle: 'Full Content',
    hasLeftPanel: true,
    hasRightPanel: true,
    hasHeader: true,
    hasContentFilter: true,
    hasHeaderRightArea: true,
    cardCount: 50
  }
}
