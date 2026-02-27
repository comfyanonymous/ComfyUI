<!-- Auto complete with extra event "focused-option-changed" -->
<script>
import AutoComplete from 'primevue/autocomplete'

export default {
  name: 'AutoCompletePlus',
  extends: AutoComplete,
  emits: ['focused-option-changed'],
  data() {
    return {
      // Flag to determine if IME is active
      isComposing: false
    }
  },
  mounted() {
    if (typeof AutoComplete.mounted === 'function') {
      AutoComplete.mounted.call(this)
    }

    // Retrieve the actual <input> element and attach IME events
    const inputEl = this.$el.querySelector('input')
    if (inputEl) {
      inputEl.addEventListener('compositionstart', () => {
        this.isComposing = true
      })
      inputEl.addEventListener('compositionend', () => {
        this.isComposing = false
      })
    }
    // Add a watcher on the focusedOptionIndex property
    this.$watch(
      () => this.focusedOptionIndex,
      (newVal, oldVal) => {
        // Emit a custom event when focusedOptionIndex changes
        this.$emit('focused-option-changed', newVal)
      }
    )
  },
  methods: {
    // Override onKeyDown to block Enter when IME is active
    onKeyDown(event) {
      if (event.key === 'Enter' && this.isComposing) {
        event.preventDefault()
        event.stopPropagation()
        return
      }

      AutoComplete.methods.onKeyDown.call(this, event)
    }
  }
}
</script>
