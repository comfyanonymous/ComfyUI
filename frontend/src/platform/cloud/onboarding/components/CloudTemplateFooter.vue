<template>
  <footer class="mx-auto flex h-[5%] max-h-[60px] w-5/6 items-start gap-2.5">
    <a
      href="https://www.comfy.org/terms-of-service"
      target="_blank"
      class="cursor-pointer text-sm text-gray-600 no-underline"
    >
      {{ t('auth.login.termsLink') }}
    </a>
    <a
      href="https://www.comfy.org/privacy-policy"
      target="_blank"
      class="cursor-pointer text-sm text-gray-600 no-underline"
    >
      {{ t('auth.login.privacyLink') }}
    </a>
    <a
      href="https://support.comfy.org"
      class="cursor-pointer text-sm text-gray-600 no-underline"
      target="_blank"
      rel="noopener noreferrer"
    >
      {{ t('cloudFooter_needHelp') }}
    </a>
  </footer>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
</script>
