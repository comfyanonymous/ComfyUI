declare module 'algoliasearch/dist/lite/builds/browser' {
  import type {
    LiteClient,
    ClientOptions
  } from 'algoliasearch/dist/lite/browser'

  /**
   * Creates a new Algolia Search client that uses the Lite API Client (Browser version)
   *
   * @param appId - Your Algolia Application ID
   * @param apiKey - Your Algolia API Key
   * @param options - Options for the client
   * @returns An Algolia Search client instance
   */
  export function liteClient(
    appId: string,
    apiKey: string,
    options?: ClientOptions
  ): LiteClient

  export const apiClientVersion: string
}
