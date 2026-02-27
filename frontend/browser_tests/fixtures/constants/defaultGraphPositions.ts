import type { Position } from './types'

/**
 * Hardcoded positions for the default graph loaded in tests.
 * These coordinates are specific to the default workflow viewport.
 */
export const DefaultGraphPositions = {
  // Node click positions
  textEncodeNode1: { x: 618, y: 191 },
  textEncodeNode2: { x: 622, y: 400 },
  textEncodeNodeToggler: { x: 430, y: 171 },
  emptySpaceClick: { x: 35, y: 31 },

  // Slot positions
  clipTextEncodeNode1InputSlot: { x: 427, y: 198 },
  clipTextEncodeNode2InputSlot: { x: 422, y: 402 },
  clipTextEncodeNode2InputLinkPath: { x: 395, y: 422 },
  loadCheckpointNodeClipOutputSlot: { x: 332, y: 509 },
  emptySpace: { x: 427, y: 98 },

  // Widget positions
  emptyLatentWidgetClick: { x: 724, y: 645 },

  // Node positions and sizes for resize operations
  ksampler: {
    pos: { x: 863, y: 156 },
    size: { width: 315, height: 292 }
  },
  loadCheckpoint: {
    pos: { x: 26, y: 444 },
    size: { width: 315, height: 127 }
  },
  emptyLatent: {
    pos: { x: 473, y: 579 },
    size: { width: 315, height: 136 }
  }
} as const satisfies {
  textEncodeNode1: Position
  textEncodeNode2: Position
  textEncodeNodeToggler: Position
  emptySpaceClick: Position
  clipTextEncodeNode1InputSlot: Position
  clipTextEncodeNode2InputSlot: Position
  clipTextEncodeNode2InputLinkPath: Position
  loadCheckpointNodeClipOutputSlot: Position
  emptySpace: Position
  emptyLatentWidgetClick: Position
  ksampler: { pos: Position; size: { width: number; height: number } }
  loadCheckpoint: { pos: Position; size: { width: number; height: number } }
  emptyLatent: { pos: Position; size: { width: number; height: number } }
}
