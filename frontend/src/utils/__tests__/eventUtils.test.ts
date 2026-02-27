import { extractFilesFromDragEvent } from '@/utils/eventUtils'
import { describe, expect, it } from 'vitest'

describe('eventUtils', () => {
  describe('extractFilesFromDragEvent', () => {
    it('should return empty array when no dataTransfer', async () => {
      const actual = await extractFilesFromDragEvent(new FakeDragEvent('drop'))
      expect(actual).toEqual([])
    })

    it('should return empty array when dataTransfer has no files', async () => {
      const actual = await extractFilesFromDragEvent(
        new FakeDragEvent('drop', { dataTransfer: new DataTransfer() })
      )
      expect(actual).toEqual([])
    })

    it('should return single file from dataTransfer', async () => {
      const file = new File([new Uint8Array()], 'workflow.json', {
        type: 'application/json'
      })
      const dataTransfer = new DataTransfer()
      dataTransfer.items.add(file)

      const actual = await extractFilesFromDragEvent(
        new FakeDragEvent('drop', { dataTransfer })
      )
      expect(actual).toEqual([file])
    })

    it('should return multiple files from dataTransfer', async () => {
      const file1 = new File([new Uint8Array()], 'workflow1.json', {
        type: 'application/json'
      })
      const file2 = new File([new Uint8Array()], 'workflow2.json', {
        type: 'application/json'
      })
      const dataTransfer = new DataTransfer()
      dataTransfer.items.add(file1)
      dataTransfer.items.add(file2)

      const actual = await extractFilesFromDragEvent(
        new FakeDragEvent('drop', { dataTransfer })
      )
      expect(actual).toEqual([file1, file2])
    })

    it('should filter out bmp files', async () => {
      const jsonFile = new File([new Uint8Array()], 'workflow.json', {
        type: 'application/json'
      })
      const bmpFile = new File([new Uint8Array()], 'image.bmp', {
        type: 'image/bmp'
      })
      const dataTransfer = new DataTransfer()
      dataTransfer.items.add(jsonFile)
      dataTransfer.items.add(bmpFile)

      const actual = await extractFilesFromDragEvent(
        new FakeDragEvent('drop', { dataTransfer })
      )
      expect(actual).toEqual([jsonFile])
    })

    it('should return multiple image files from dataTransfer', async () => {
      const imageFile1 = new File([new Uint8Array()], 'image1.png', {
        type: 'image/png'
      })
      const imageFile2 = new File([new Uint8Array()], 'image2.jpg', {
        type: 'image/jpeg'
      })

      const dataTransfer = new DataTransfer()
      dataTransfer.items.add(imageFile1)
      dataTransfer.items.add(imageFile2)

      const actual = await extractFilesFromDragEvent(
        new FakeDragEvent('drop', { dataTransfer })
      )
      expect(actual).toEqual([imageFile1, imageFile2])
    })

    it('should return multiple non-image files from dataTransfer', async () => {
      const file1 = new File([new Uint8Array()], 'file1.txt', {
        type: 'text/plain'
      })
      const file2 = new File([new Uint8Array()], 'file2.txt', {
        type: 'text/plain'
      })

      const dataTransfer = new DataTransfer()
      dataTransfer.items.add(file1)
      dataTransfer.items.add(file2)

      const actual = await extractFilesFromDragEvent(
        new FakeDragEvent('drop', { dataTransfer })
      )
      expect(actual).toEqual([file1, file2])
    })

    // Skip until we can setup MSW
    it.skip('should handle drops with URLs', async () => {
      const urlWithWorkflow = 'https://fakewebsite.notreal/fake_workflow.json'

      const dataTransfer = new DataTransfer()
      dataTransfer.setData('text/uri-list', urlWithWorkflow)
      dataTransfer.setData('text/x-moz-url', urlWithWorkflow)

      const actual = await extractFilesFromDragEvent(
        new FakeDragEvent('drop', { dataTransfer })
      )
      expect(actual.length).toBe(1)
      expect(actual[0]).toBeInstanceOf(File)
    })
  })
})

// Needed to keep the dataTransfer defined
class FakeDragEvent extends DragEvent {
  override dataTransfer: DataTransfer | null
  override clientX: number
  override clientY: number

  constructor(
    type: string,
    { dataTransfer, clientX, clientY }: DragEventInit = {}
  ) {
    super(type)
    this.dataTransfer = dataTransfer ?? null
    this.clientX = clientX ?? 0
    this.clientY = clientY ?? 0
  }
}
