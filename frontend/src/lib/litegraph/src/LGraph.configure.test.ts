// TODO: Fix these tests after migration
import { describe } from 'vitest'

import { LGraph } from '@/lib/litegraph/src/litegraph'

import { dirtyTest } from './__fixtures__/testExtensions'

describe.skip('LGraph configure()', () => {
  dirtyTest(
    'LGraph matches previous snapshot (normal configure() usage)',
    ({ expect, minimalSerialisableGraph, basicSerialisableGraph }) => {
      const configuredMinGraph = new LGraph()
      configuredMinGraph.configure(minimalSerialisableGraph)
      expect(configuredMinGraph).toMatchSnapshot('configuredMinGraph')

      const configuredBasicGraph = new LGraph()
      configuredBasicGraph.configure(basicSerialisableGraph)
      expect(configuredBasicGraph).toMatchSnapshot('configuredBasicGraph')
    }
  )
})
