import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { api } from '@/scripts/api'
import { UserFile, useUserFileStore } from '@/stores/userFileStore'

// Mock the api
vi.mock('@/scripts/api', () => ({
  api: {
    listUserDataFullInfo: vi.fn(),
    getUserData: vi.fn(),
    storeUserData: vi.fn(),
    deleteUserData: vi.fn(),
    moveUserData: vi.fn()
  }
}))

describe('useUserFileStore', () => {
  let store: ReturnType<typeof useUserFileStore>

  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    store = useUserFileStore()
    vi.resetAllMocks()
  })

  it('should initialize with empty files', () => {
    expect(store.userFiles).toHaveLength(0)
    expect(store.modifiedFiles).toHaveLength(0)
    expect(store.loadedFiles).toHaveLength(0)
  })

  describe('syncFiles', () => {
    it('should add new files', async () => {
      const mockFiles = [
        { path: 'file1.txt', modified: 123, size: 100 },
        { path: 'file2.txt', modified: 456, size: 200 }
      ]
      vi.mocked(api.listUserDataFullInfo).mockResolvedValue(mockFiles)

      await store.syncFiles('dir')

      expect(store.userFiles).toHaveLength(2)
      expect(store.userFiles[0].path).toBe('dir/file1.txt')
      expect(store.userFiles[1].path).toBe('dir/file2.txt')
    })

    it('should update existing files', async () => {
      const initialFile = { path: 'file1.txt', modified: 123, size: 100 }
      vi.mocked(api.listUserDataFullInfo).mockResolvedValue([initialFile])
      await store.syncFiles('dir')

      const updatedFile = { path: 'file1.txt', modified: 456, size: 200 }
      vi.mocked(api.listUserDataFullInfo).mockResolvedValue([updatedFile])
      await store.syncFiles('dir')

      expect(store.userFiles).toHaveLength(1)
      expect(store.userFiles[0].lastModified).toBe(456)
      expect(store.userFiles[0].size).toBe(200)
    })

    it('should remove non-existent files', async () => {
      const initialFiles = [
        { path: 'file1.txt', modified: 123, size: 100 },
        { path: 'file2.txt', modified: 456, size: 200 }
      ]
      vi.mocked(api.listUserDataFullInfo).mockResolvedValue(initialFiles)
      await store.syncFiles('dir')

      const updatedFiles = [{ path: 'file1.txt', modified: 123, size: 100 }]
      vi.mocked(api.listUserDataFullInfo).mockResolvedValue(updatedFiles)
      await store.syncFiles('dir')

      expect(store.userFiles).toHaveLength(1)
      expect(store.userFiles[0].path).toBe('dir/file1.txt')
    })

    it('should sync root directory when no directory is specified', async () => {
      const mockFiles = [{ path: 'file1.txt', modified: 123, size: 100 }]
      vi.mocked(api.listUserDataFullInfo).mockResolvedValue(mockFiles)

      await store.syncFiles()

      expect(api.listUserDataFullInfo).toHaveBeenCalledWith('')
      expect(store.userFiles).toHaveLength(1)
      expect(store.userFiles[0].path).toBe('file1.txt')
    })
  })

  describe('UserFile', () => {
    describe('load', () => {
      it('should load file content', async () => {
        const file = new UserFile('file1.txt', 123, 100)
        vi.mocked(api.getUserData).mockResolvedValue({
          status: 200,
          text: () => Promise.resolve('file content')
        } as Response)

        await file.load()

        expect(file.content).toBe('file content')
        expect(file.originalContent).toBe('file content')
        expect(file.isLoading).toBe(false)
        expect(file.isLoaded).toBe(true)
      })

      it('should throw error on failed load', async () => {
        const file = new UserFile('file1.txt', 123, 100)
        vi.mocked(api.getUserData).mockResolvedValue({
          status: 404,
          statusText: 'Not Found'
        } as Response)

        await expect(file.load()).rejects.toThrow(
          "Failed to load file 'file1.txt': 404 Not Found"
        )
      })
    })

    describe('save', () => {
      it('should save modified file', async () => {
        const file = new UserFile('file1.txt', 123, 100)
        file.content = 'modified content'
        file.originalContent = 'original content'
        vi.mocked(api.storeUserData).mockResolvedValue({
          status: 200,
          json: () => Promise.resolve({ modified: 456, size: 200 })
        } as Response)

        await file.save()

        expect(api.storeUserData).toHaveBeenCalledWith(
          'file1.txt',
          'modified content',
          { throwOnError: true, full_info: true, overwrite: true }
        )
        expect(file.lastModified).toBe(456)
        expect(file.size).toBe(200)
      })

      it('should not save unmodified file', async () => {
        const file = new UserFile('file1.txt', 123, 100)
        file.content = 'content'
        file.originalContent = 'content'

        await file.save()

        expect(api.storeUserData).not.toHaveBeenCalled()
      })
    })

    describe('delete', () => {
      it('should delete file', async () => {
        const file = new UserFile('file1.txt', 123, 100)
        vi.mocked(api.deleteUserData).mockResolvedValue({
          status: 204
        } as Response)

        await file.delete()

        expect(api.deleteUserData).toHaveBeenCalledWith('file1.txt')
      })
    })

    describe('rename', () => {
      it('should rename file', async () => {
        const file = new UserFile('file1.txt', 123, 100)
        vi.mocked(api.moveUserData).mockResolvedValue({
          status: 200,
          json: () => Promise.resolve({ modified: 456, size: 200 })
        } as Response)

        await file.rename('newfile.txt')

        expect(api.moveUserData).toHaveBeenCalledWith(
          'file1.txt',
          'newfile.txt'
        )
        expect(file.path).toBe('newfile.txt')
        expect(file.lastModified).toBe(456)
        expect(file.size).toBe(200)
      })
    })

    describe('saveAs', () => {
      it('should save file with new path', async () => {
        const file = new UserFile('file1.txt', 123, 100)
        file.content = 'file content'
        vi.mocked(api.storeUserData).mockResolvedValue({
          status: 200,
          json: () => Promise.resolve({ modified: 456, size: 200 })
        } as Response)

        const newFile = await file.saveAs('newfile.txt')

        expect(api.storeUserData).toHaveBeenCalledWith(
          'newfile.txt',
          'file content',
          // SaveAs should create a new temporary file, which will mean
          // overwrite is false
          { throwOnError: true, full_info: true, overwrite: false }
        )
        expect(newFile).toBeInstanceOf(UserFile)
        expect(newFile.path).toBe('newfile.txt')
        expect(newFile.lastModified).toBe(456)
        expect(newFile.size).toBe(200)
        expect(newFile.content).toBe('file content')
      })
    })
  })
})
