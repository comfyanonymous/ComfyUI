export class NullGraphError extends Error {
  constructor(
    message: string = 'Attempted to access LGraph reference that was null or undefined.',
    cause?: Error
  ) {
    super(message, { cause })
    this.name = 'NullGraphError'
  }
}
