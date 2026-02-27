// TODO: Fix these tests after migration
import { describe } from 'vitest'

import { LGraph } from '@/lib/litegraph/src/litegraph'

import { dirtyTest } from './__fixtures__/testExtensions'

describe.skip('LGraph (constructor only)', () => {
  dirtyTest(
    'Matches previous snapshot',
    ({ expect, minimalSerialisableGraph, basicSerialisableGraph }) => {
      const minLGraph = new LGraph(minimalSerialisableGraph)
      expect(minLGraph).toMatchSnapshot('minLGraph')

      const basicLGraph = new LGraph(basicSerialisableGraph)
      expect(basicLGraph).toMatchSnapshot('basicLGraph')
    }
  )
})
