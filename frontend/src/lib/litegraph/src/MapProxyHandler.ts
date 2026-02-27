/**
 * Temporary workaround until downstream consumers migrate to Map.
 * A brittle wrapper with many flaws, but should be fine for simple maps using int indexes.
 */
export class MapProxyHandler<V> implements ProxyHandler<
  Map<number | string, V>
> {
  getOwnPropertyDescriptor(
    target: Map<number | string, V>,
    p: string | symbol
  ): PropertyDescriptor | undefined {
    const value = this.get(target, p)
    if (value) {
      return {
        configurable: true,
        enumerable: true,
        value
      }
    }
  }

  has(target: Map<number | string, V>, p: string | symbol): boolean {
    if (typeof p === 'symbol') return false

    const int = parseInt(p, 10)
    return target.has(!isNaN(int) ? int : p)
  }

  ownKeys(target: Map<number | string, V>): ArrayLike<string | symbol> {
    return [...target.keys()].map(String)
  }

  get(target: Map<number | string, V>, p: string | symbol): V | undefined {
    // Workaround does not support link IDs of "values", "entries", "constructor", etc.
    if (p in target) return Reflect.get(target, p, target)
    if (typeof p === 'symbol') return

    const int = parseInt(p, 10)
    return target.get(!isNaN(int) ? int : p)
  }

  set(
    target: Map<number | string, V>,
    p: string | symbol,
    newValue: V
  ): boolean {
    if (typeof p === 'symbol') return false

    const int = parseInt(p, 10)
    target.set(!isNaN(int) ? int : p, newValue)
    return true
  }

  deleteProperty(target: Map<number | string, V>, p: string | symbol): boolean {
    return target.delete(p as number | string)
  }

  static bindAllMethods(map: Map<unknown, unknown>): void {
    map.clear = map.clear.bind(map)
    map.delete = map.delete.bind(map)
    map.forEach = map.forEach.bind(map)
    map.get = map.get.bind(map)
    map.has = map.has.bind(map)
    map.set = map.set.bind(map)
    map.entries = map.entries.bind(map)
    map.keys = map.keys.bind(map)
    map.values = map.values.bind(map)

    map[Symbol.iterator] = map[Symbol.iterator].bind(map)
  }
}
