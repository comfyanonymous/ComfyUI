<script>
import Select from 'primevue/select'

export default {
  name: 'SelectPlus',
  extends: Select,
  emits: ['hide'],
  methods: {
    onOverlayLeave() {
      this.unbindOutsideClickListener()
      this.unbindScrollListener()
      this.unbindResizeListener()

      this.$emit('hide')
      this.overlay = null
    }
  }
}
</script>
