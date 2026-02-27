/**
 * Polyfill for disposable type; symbol already registered in modern browsers.
 */

interface SymbolConstructor {
  /**
   * A method that is used to release resources held by an object. Called by the semantics of the `using` statement.
   */
  readonly dispose: unique symbol

  /**
   * A method that is used to asynchronously release resources held by an object. Called by the semantics of the `await using` statement.
   */
  readonly asyncDispose: unique symbol
}

interface Disposable {
  [Symbol.dispose](): void
}

interface AsyncDisposable {
  [Symbol.asyncDispose](): PromiseLike<void>
}
