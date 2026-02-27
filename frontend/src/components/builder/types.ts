export interface BuilderToolbarStep<T extends string = string> {
  id: T
  title: string
  subtitle: string
  icon: string
}
