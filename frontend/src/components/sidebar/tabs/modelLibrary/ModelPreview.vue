<template>
  <div class="model_preview">
    <div class="model_preview_title">
      {{ modelDef.title }}
    </div>
    <div class="model_preview_top_container">
      <div class="model_preview_filename">
        {{ modelDef.file_name }}
      </div>
      <div v-if="modelDef.architecture_id" class="model_preview_architecture">
        <span class="model_preview_prefix">{{ $t('g.architecture') }}: </span>
        {{ modelDef.architecture_id }}
      </div>
      <div v-if="modelDef.author" class="model_preview_author">
        <span class="model_preview_prefix">{{ $t('g.author') }}: </span>
        {{ modelDef.author }}
      </div>
    </div>
    <div v-if="modelDef.image" class="model_preview_image">
      <img :src="modelDef.image" />
    </div>
    <div v-if="modelDef.usage_hint" class="model_preview_usage_hint">
      <span class="model_preview_prefix">{{ $t('g.usageHint') }}: </span>
      {{ modelDef.usage_hint }}
    </div>
    <div v-if="modelDef.trigger_phrase" class="model_preview_trigger_phrase">
      <span class="model_preview_prefix">{{ $t('g.triggerPhrase') }}: </span>
      {{ modelDef.trigger_phrase }}
    </div>
    <div v-if="modelDef.description" class="model_preview_description">
      <span class="model_preview_prefix">{{ $t('g.description') }}: </span>
      {{ modelDef.description }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ComfyModelDef } from '@/stores/modelStore'

const props = defineProps({
  modelDef: {
    type: ComfyModelDef,
    required: true
  }
})

const modelDef = props.modelDef
</script>
<style scoped>
.model_preview {
  background-color: var(--comfy-menu-bg);
  color: var(--descrip-text);
  border: 1px solid var(--descrip-text);
  min-width: 300px;
  max-width: 500px;
  width: fit-content;
  height: fit-content;
  z-index: 9999;
  border-radius: 12px;
  overflow: hidden;
  font-size: 12px;
  padding: 10px;
}
.model_preview_image {
  margin: auto;
  width: fit-content;
}
.model_preview_image img {
  max-width: 100%;
  max-height: 150px;
  object-fit: contain;
}
.model_preview_title {
  font-weight: 700;
  text-align: center;
  font-size: 14px;
}
.model_preview_top_container {
  text-align: center;
  line-height: 0.5;
}
.model_preview_filename,
.model_preview_author,
.model_preview_architecture {
  display: inline-block;
  text-align: center;
  margin: 5px;
  font-size: 10px;
}
.model_preview_prefix {
  font-weight: 700;
}
</style>
