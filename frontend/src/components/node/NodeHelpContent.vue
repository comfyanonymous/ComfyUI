<template>
  <div class="node-help-content mx-auto w-full">
    <ProgressSpinner
      v-if="isLoading"
      class="m-auto"
      :aria-label="$t('g.loading')"
    />
    <!-- Markdown fetched successfully -->
    <div
      v-else-if="!error"
      class="markdown-content overflow-visible text-sm leading-(--text-sm--line-height)"
      v-html="renderedHelpHtml"
    />
    <!-- Fallback: markdown not found or fetch error -->
    <div
      v-else
      class="fallback-content space-y-6 text-sm leading-(--text-sm--line-height)"
    >
      <p v-if="node.description">
        <strong>{{ $t('g.description') }}:</strong> {{ node.description }}
      </p>

      <div v-if="inputList.length">
        <p>
          <strong>{{ $t('nodeHelpPage.inputs') }}:</strong>
        </p>
        <!-- Using plain HTML table instead of DataTable for consistent styling with markdown content -->
        <div class="overflow-x-auto">
          <table>
            <thead>
              <tr>
                <th>{{ $t('g.name') }}</th>
                <th>{{ $t('nodeHelpPage.type') }}</th>
                <th>{{ $t('g.description') }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="input in inputList" :key="input.name">
                <td>
                  <code>{{ input.name }}</code>
                </td>
                <td>{{ input.type }}</td>
                <td>{{ input.tooltip || '-' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div v-if="outputList.length">
        <p>
          <strong>{{ $t('nodeHelpPage.outputs') }}:</strong>
        </p>
        <div class="overflow-x-auto">
          <table>
            <thead>
              <tr>
                <th>{{ $t('g.name') }}</th>
                <th>{{ $t('nodeHelpPage.type') }}</th>
                <th>{{ $t('g.description') }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="output in outputList" :key="output.name">
                <td>
                  <code>{{ output.name }}</code>
                </td>
                <td>{{ output.type }}</td>
                <td>{{ output.tooltip || '-' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import ProgressSpinner from 'primevue/progressspinner'
import { computed } from 'vue'

import { useNodeHelpContent } from '@/composables/useNodeHelpContent'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'

const { node } = defineProps<{
  node: ComfyNodeDefImpl
}>()

const { renderedHelpHtml, isLoading, error } = useNodeHelpContent(() => node)

const inputList = computed(() =>
  Object.values(node.inputs).map((spec) => ({
    name: spec.name,
    type: spec.type,
    tooltip: spec.tooltip || ''
  }))
)

const outputList = computed(() =>
  node.outputs.map((spec) => ({
    name: spec.name,
    type: spec.type,
    tooltip: spec.tooltip || ''
  }))
)
</script>

<style scoped>
.node-help-content :deep(:is(img, video)) {
  display: block;
  max-width: 100%;
  height: auto;
  margin-bottom: calc(var(--spacing) * 4);
}

.markdown-content :deep(h1),
.fallback-content h1 {
  margin-top: calc(var(--spacing) * 8);
  margin-bottom: calc(var(--spacing) * 4);
  font-size: var(--text-2xl);
  font-weight: var(--font-weight-bold);
}

.markdown-content :deep(h2),
.fallback-content h2 {
  margin-top: calc(var(--spacing) * 8);
  margin-bottom: calc(var(--spacing) * 4);
  font-size: var(--text-lg);
  font-weight: var(--font-weight-bold);
}

.markdown-content :deep(h3),
.fallback-content h3 {
  margin-top: calc(var(--spacing) * 8);
  margin-bottom: calc(var(--spacing) * 4);
  font-size: var(--text-base);
  font-weight: var(--font-weight-bold);
}

.markdown-content :deep(h4),
.fallback-content h4 {
  margin-top: calc(var(--spacing) * 8);
  margin-bottom: calc(var(--spacing) * 4);
  font-size: var(--text-sm);
  font-weight: var(--font-weight-bold);
}

.markdown-content :deep(h5),
.fallback-content h5 {
  margin-top: calc(var(--spacing) * 8);
  margin-bottom: calc(var(--spacing) * 4);
  font-size: var(--text-sm);
  font-weight: var(--font-weight-bold);
}

.markdown-content :deep(h6),
.fallback-content h6 {
  margin-top: calc(var(--spacing) * 8);
  margin-bottom: calc(var(--spacing) * 4);
  font-size: var(--text-xs);
  font-weight: var(--font-weight-bold);
}

.markdown-content :deep(td),
.fallback-content td {
  color: var(--drag-text);
}

.markdown-content :deep(a),
.fallback-content a {
  color: var(--drag-text);
  text-decoration: underline;
}

.markdown-content :deep(th),
.fallback-content th {
  color: var(--fg-color);
}

.markdown-content :deep(ul),
.markdown-content :deep(ol),
.fallback-content ul,
.fallback-content ol {
  margin-block: calc(var(--spacing) * 2);
  padding-left: calc(var(--spacing) * 8);
}

.markdown-content :deep(ul ul),
.markdown-content :deep(ol ol),
.markdown-content :deep(ul ol),
.markdown-content :deep(ol ul),
.fallback-content ul ul,
.fallback-content ol ol,
.fallback-content ul ol,
.fallback-content ol ul {
  margin-block: calc(var(--spacing) * 2);
  padding-left: calc(var(--spacing) * 6);
}

.markdown-content :deep(li),
.fallback-content li {
  margin-block: calc(var(--spacing) * 2);
}

.markdown-content :deep(*:first-child),
.fallback-content > *:first-child {
  margin-top: 0;
}

.markdown-content :deep(code),
.fallback-content code {
  color: var(--code-text-color);
  background-color: var(--code-bg-color);
  border-radius: var(--radius);
  padding: calc(var(--spacing) * 0.5) calc(var(--spacing) * 1.5);
}

.markdown-content :deep(table),
.fallback-content table {
  border-collapse: collapse;
}

.fallback-content table {
  width: 100%;
}

.markdown-content :deep(th),
.markdown-content :deep(td),
.fallback-content th,
.fallback-content td {
  padding: calc(var(--spacing) * 2);
}

.markdown-content :deep(tr),
.fallback-content tr {
  border-bottom: 1px solid var(--content-bg);
}

.markdown-content :deep(tr:last-child),
.fallback-content tr:last-child {
  border-bottom: none;
}

.markdown-content :deep(thead),
.fallback-content thead {
  border-bottom: 1px solid var(--p-text-color);
}

.markdown-content :deep(pre),
.fallback-content pre {
  margin-block: calc(var(--spacing) * 4);
  overflow-x: auto;
  border-radius: var(--radius);
  padding: calc(var(--spacing) * 4);
  background-color: var(--code-block-bg-color);

  code {
    background-color: transparent;
    padding: 0;
    color: var(--p-text-color);
  }
}

.markdown-content :deep(table) {
  display: block;
  width: 100%;
  overflow-x: auto;
}
</style>
