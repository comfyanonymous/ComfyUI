import * as THREE from 'three'

import { parseASCIIPLY } from '@/scripts/metadata/ply'

/**
 * Fast ASCII PLY Loader
 * Optimized for simple ASCII PLY files with position and color data
 * 4-5x faster than Three.js PLYLoader for ASCII files
 */
export class FastPLYLoader {
  parse(arrayBuffer: ArrayBuffer): THREE.BufferGeometry {
    const plyData = parseASCIIPLY(arrayBuffer)

    if (!plyData) {
      throw new Error('Failed to parse PLY data')
    }

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute(
      'position',
      new THREE.BufferAttribute(plyData.positions, 3)
    )

    if (plyData.colors) {
      geometry.setAttribute(
        'color',
        new THREE.BufferAttribute(plyData.colors, 3)
      )
    }

    return geometry
  }
}
