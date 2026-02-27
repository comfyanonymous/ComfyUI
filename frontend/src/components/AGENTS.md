# Component Guidelines

## Component Communication

- Prefer `emit/@event-name` for state changes
- Use `defineExpose` only for imperative operations (`form.validate()`, `modal.open()`)
