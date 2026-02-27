import { computed, ref } from 'vue'

export function useZoomControls() {
  const isModalVisible = ref(false)

  const showModal = () => {
    isModalVisible.value = true
  }

  const hideModal = () => {
    isModalVisible.value = false
  }

  const toggleModal = () => {
    isModalVisible.value = !isModalVisible.value
  }

  const hasActivePopup = computed(() => isModalVisible.value)

  return {
    isModalVisible,
    showModal,
    hideModal,
    toggleModal,
    hasActivePopup
  }
}
