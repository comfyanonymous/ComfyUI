type LoggedInAuthHeader = {
  Authorization: `Bearer ${string}`
}

export type ApiKeyAuthHeader = {
  'X-API-KEY': string
}

export type AuthHeader = LoggedInAuthHeader | ApiKeyAuthHeader

export interface AuthUserInfo {
  id: string
}
