/**
 * General-purpose, TypeScript utility types.
 */

/** {@link Pick} only properties that evaluate to `never`. */
export type PickNevers<T> = {
  [K in keyof T as T[K] extends never ? K : never]: T[K]
}

/** {@link Omit} all properties that evaluate to `never`. */
export type NeverNever<T> = {
  [K in keyof T as T[K] extends never ? never : K]: T[K]
}
