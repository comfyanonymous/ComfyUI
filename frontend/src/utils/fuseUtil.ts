import type { FuseOptionKey, FuseSearchOptions, IFuseOptions } from 'fuse.js'
import Fuse from 'fuse.js'

export type SearchAuxScore = number[]

export interface FuseFilterWithValue<T, O = string> {
  filterDef: FuseFilter<T, O>
  value: O
}

export class FuseFilter<T, O = string> {
  public readonly fuseSearch: FuseSearch<O>
  /** The unique identifier for the filter. */
  public readonly id: string
  /** The name of the filter for display purposes. */
  public readonly name: string
  /** The sequence of characters to invoke the filter. */
  public readonly invokeSequence: string
  /** A function that returns the options for the filter. */
  public readonly getItemOptions: (item: T) => O[]

  constructor(
    data: T[],
    options: {
      id: string
      name: string
      invokeSequence: string
      getItemOptions: (item: T) => O[]
      fuseOptions?: IFuseOptions<O>
    }
  ) {
    this.id = options.id
    this.name = options.name
    this.invokeSequence = options.invokeSequence
    this.getItemOptions = options.getItemOptions

    this.fuseSearch = new FuseSearch(this.getAllNodeOptions(data), {
      fuseOptions: options.fuseOptions
    })
  }

  public getAllNodeOptions(data: T[]): O[] {
    const options = new Set<O>()
    for (const item of data) {
      for (const option of this.getItemOptions(item)) {
        options.add(option)
      }
    }
    return Array.from(options)
  }

  public matches(
    item: T,
    value: O,
    extraOptions: {
      wildcard?: O
    } = {}
  ): boolean {
    const { wildcard } = extraOptions

    if (wildcard && value === wildcard) {
      return true
    }
    const options = this.getItemOptions(item)
    if (wildcard) return options.some((option) => option === wildcard)
    if (typeof value !== 'string' || !value.includes(','))
      return options.includes(value)
    const values = value.split(',')
    //Alas, typescript doesn't understand string satisfies O
    return values.some((v) => options.includes(v as O))
  }
}

export interface FuseSearchable {
  postProcessSearchScores: (scores: SearchAuxScore) => SearchAuxScore
}

function isFuseSearchable(item: unknown): item is FuseSearchable {
  return (
    typeof item === 'object' &&
    item !== null &&
    'postProcessSearchScores' in item
  )
}

/**
 * A wrapper around Fuse.js that provides a more type-safe API.
 */
export class FuseSearch<T> {
  public readonly fuse: Fuse<T>
  public readonly keys: FuseOptionKey<T>[]
  public readonly data: T[]
  public readonly advancedScoring: boolean

  constructor(
    data: T[],
    options: {
      fuseOptions?: IFuseOptions<T>
      createIndex?: boolean
      advancedScoring?: boolean
    }
  ) {
    const { fuseOptions, createIndex = true, advancedScoring = false } = options

    this.data = data
    this.keys = fuseOptions?.keys ?? []
    this.advancedScoring = advancedScoring
    const index =
      createIndex && this.keys.length
        ? Fuse.createIndex(this.keys, data)
        : undefined
    this.fuse = new Fuse(data, fuseOptions, index)
  }

  public search(query: string, options?: FuseSearchOptions): T[] {
    const fuseResult = !query
      ? this.data.map((x) => ({ item: x, score: 0 }))
      : this.fuse.search(query, options)

    if (!this.advancedScoring) {
      return fuseResult.map((x) => x.item)
    }

    const aux = fuseResult
      .map((x) => ({
        item: x.item,
        scores: this.calcAuxScores(
          query.toLocaleLowerCase(),
          x.item,
          x.score ?? 0
        )
      }))
      .sort((a, b) => this.compareAux(a.scores, b.scores))

    return aux.map((x) => x.item)
  }

  public calcAuxScores(query: string, entry: T, score: number): SearchAuxScore {
    let values: string[] = []
    if (typeof entry === 'string') {
      values = [entry]
    } else if (typeof entry === 'object' && entry !== null) {
      values = this.keys
        .map((x) => entry[x as keyof T])
        .filter((x) => typeof x === 'string') as string[]
    }
    const scores = values.map((x) => this.calcAuxSingle(query, x, score))
    let result = scores.sort(this.compareAux)[0]

    const deprecated = values.some((x) =>
      x.toLocaleLowerCase().includes('deprecated')
    )
    result[0] += deprecated && result[0] !== 0 ? 5 : 0
    if (isFuseSearchable(entry)) {
      result = entry.postProcessSearchScores(result)
    }
    return result
  }

  public calcAuxSingle(
    query: string,
    item: string,
    score: number
  ): SearchAuxScore {
    const itemWords = item
      .split(/ |\b|(?<=[a-z])(?=[A-Z])|(?=[A-Z][a-z])/)
      .map((x) => x.toLocaleLowerCase())
    const queryParts = query.split(' ')
    item = item.toLocaleLowerCase()

    let main = 9
    let aux1 = 0
    let aux2 = 0

    if (item == query) {
      main = 0
    } else if (item.startsWith(query)) {
      main = 1
      aux2 = item.length
    } else if (itemWords.includes(query)) {
      main = 2
      aux1 = item.indexOf(query) + item.length * 0.5
      aux2 = item.length
    } else if (item.includes(query)) {
      main = 3
      aux1 = item.indexOf(query) + item.length * 0.5
      aux2 = item.length
    } else if (queryParts.every((x) => itemWords.includes(x))) {
      const indexes = queryParts.map((x) => itemWords.indexOf(x))
      const min = Math.min(...indexes)
      const max = Math.max(...indexes)
      main = 4
      aux1 = max - min + max * 0.5 + item.length * 0.5
      aux2 = item.length
    } else if (queryParts.every((x) => item.includes(x))) {
      const min = Math.min(...queryParts.map((x) => item.indexOf(x)))
      const max = Math.max(...queryParts.map((x) => item.indexOf(x) + x.length))
      main = 5
      aux1 = max - min + max * 0.5 + item.length * 0.5
      aux2 = item.length
    }

    const lengthPenalty =
      0.2 *
      (1 -
        Math.min(item.length, query.length) /
          Math.max(item.length, query.length))
    return [main, aux1, aux2, score + lengthPenalty]
  }

  public compareAux(a: SearchAuxScore, b: SearchAuxScore) {
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      if (a[i] !== b[i]) {
        return a[i] - b[i]
      }
    }
    return a.length - b.length
  }
}
