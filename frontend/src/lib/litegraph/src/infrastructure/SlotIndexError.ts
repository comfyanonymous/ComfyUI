export class SlotIndexError extends Error {
  constructor(
    message: string = 'Attempted to access a slot that was out of bounds.',
    cause?: Error
  ) {
    super(message, { cause })
    this.name = 'SlotIndexError'
  }
}
