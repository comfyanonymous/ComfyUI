<template>
  <BaseModalLayout :content-title="$t('assetBrowser.checkpoints')">
    <template #leftPanel>
      <LeftSidePanel v-model="selectedNavItem" :nav-items="tempNavigation">
        <template #header-icon>
          <i class="text-neutral icon-[lucide--puzzle]" />
        </template>
        <template #header-title>
          <span class="text-neutral text-base">{{ $t('g.title') }}</span>
        </template>
      </LeftSidePanel>
    </template>

    <template #header>
      <SearchBox v-model="searchQuery" size="lg" class="max-w-[384px]" />
    </template>

    <template #header-right-area>
      <div class="flex gap-2">
        <Button variant="primary" @click="() => {}">
          <i class="icon-[lucide--upload]" />
          <span>{{ $t('g.upload') }}</span>
        </Button>
        <MoreButton>
          <template #default="{ close }">
            <Button
              variant="secondary"
              @click="
                () => {
                  close()
                }
              "
            >
              <i class="icon-[lucide--download]" />
              <span>{{ $t('g.settings') }}</span>
            </Button>
            <Button
              variant="primary"
              @click="
                () => {
                  close()
                }
              "
            >
              <i class="icon-[lucide--scroll]" />
              <span>{{ $t('g.profile') }}</span>
            </Button>
          </template>
        </MoreButton>
      </div>
    </template>

    <template #contentFilter>
      <div class="relative flex gap-2 px-6 pb-4">
        <MultiSelect
          v-model="selectedFrameworks"
          v-model:search-query="searchText"
          class="w-[250px]"
          :label="$t('assetBrowser.selectFrameworks')"
          :options="frameworkOptions"
          :show-search-box="true"
          :show-selected-count="true"
          :show-clear-button="true"
        />
        <MultiSelect
          v-model="selectedProjects"
          :label="$t('assetBrowser.selectProjects')"
          :options="projectOptions"
        />
        <SingleSelect
          v-model="selectedSort"
          :label="$t('assetBrowser.sortingType')"
          :options="sortOptions"
          class="w-[135px]"
        >
          <template #icon>
            <i class="icon-[lucide--filter]" />
          </template>
        </SingleSelect>
      </div>
    </template>

    <template #content>
      <!-- Card Examples -->
      <div :style="gridStyle">
        <CardContainer v-for="i in 100" :key="i" size="regular">
          <template #top>
            <CardTop ratio="landscape">
              <template #default>
                <div class="h-full w-full bg-blue-500"></div>
              </template>
              <template #top-right>
                <Button
                  size="icon"
                  class="!bg-white !text-neutral-900"
                  @click="() => {}"
                >
                  <i class="icon-[lucide--info]" />
                </Button>
              </template>
              <template #bottom-right>
                <SquareChip label="png" />
                <SquareChip label="1.2 MB" />
                <SquareChip label="LoRA">
                  <template #icon>
                    <i class="icon-[lucide--folder]" />
                  </template>
                </SquareChip>
              </template>
            </CardTop>
          </template>
          <template #bottom>
            <CardBottom></CardBottom>
          </template>
        </CardContainer>
      </div>
    </template>

    <template #rightPanel>
      <div class="size-full bg-modal-panel-background pr-6 pb-8 pl-4"></div>
    </template>
  </BaseModalLayout>
</template>

<script setup lang="ts">
import { computed, provide, ref } from 'vue'

import MoreButton from '@/components/button/MoreButton.vue'
import CardBottom from '@/components/card/CardBottom.vue'
import CardContainer from '@/components/card/CardContainer.vue'
import CardTop from '@/components/card/CardTop.vue'
import SquareChip from '@/components/chip/SquareChip.vue'
import SearchBox from '@/components/common/SearchBox.vue'
import MultiSelect from '@/components/input/MultiSelect.vue'
import SingleSelect from '@/components/input/SingleSelect.vue'
import Button from '@/components/ui/button/Button.vue'
import BaseModalLayout from '@/components/widget/layout/BaseModalLayout.vue'
import LeftSidePanel from '@/components/widget/panel/LeftSidePanel.vue'
import type { NavGroupData, NavItemData } from '@/types/navTypes'
import { OnCloseKey } from '@/types/widgetTypes'
import { createGridStyle } from '@/utils/gridUtil'

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

const tempNavigation = ref<(NavItemData | NavGroupData)[]>([
  { id: 'installed', label: 'Installed', icon: 'icon-[lucide--download]' },
  {
    title: 'TAGS',
    items: [
      { id: 'tag-sd15', label: 'SD 1.5', icon: 'icon-[lucide--tag]' },
      { id: 'tag-sdxl', label: 'SDXL', icon: 'icon-[lucide--tag]' },
      { id: 'tag-utility', label: 'Utility', icon: 'icon-[lucide--tag]' }
    ]
  },
  {
    title: 'CATEGORIES',
    items: [
      { id: 'cat-models', label: 'Models', icon: 'icon-[lucide--layers]' },
      { id: 'cat-nodes', label: 'Nodes', icon: 'icon-[lucide--grid-3x3]' }
    ]
  }
])

const { onClose } = defineProps<{
  onClose: () => void
}>()

provide(OnCloseKey, onClose)

const searchQuery = ref<string>('')
const searchText = ref<string>('')
const selectedFrameworks = ref([])
const selectedProjects = ref([])
const selectedSort = ref<string>('popular')

const selectedNavItem = ref<string | null>('installed')

const gridStyle = computed(() => createGridStyle())
</script>
