/**
 * Error thrown when infinite recursion is detected.
 */
export class RecursionError extends Error {
  constructor(subject: string) {
    super(subject)
    this.name = 'RecursionError'
  }
}
