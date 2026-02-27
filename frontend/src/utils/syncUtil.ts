import { api } from '@/scripts/api'
import type { UserDataFullInfo } from '@/schemas/apiSchema'

/**
 * Sync entities from the API to the entityByPath map.
 * @param dir The directory to sync from
 * @param entityByPath The map to sync to
 * @param createEntity A function to create an entity from a file
 * @param updateEntity A function to update an entity from a file
 * @param exclude A function to exclude an entity
 */
export async function syncEntities<T>(
  dir: string,
  entityByPath: Record<string, T>,
  createEntity: (file: UserDataFullInfo & { path: string }) => T,
  updateEntity: (entity: T, file: UserDataFullInfo & { path: string }) => void,
  exclude: (file: T) => boolean = () => false
) {
  const files = (await api.listUserDataFullInfo(dir)).map((file) => ({
    ...file,
    path: dir ? `${dir}/${file.path}` : file.path
  }))

  for (const file of files) {
    const existingEntity = entityByPath[file.path]

    if (!existingEntity) {
      // New entity, add it to the map
      entityByPath[file.path] = createEntity(file)
    } else if (exclude(existingEntity)) {
      // Entity has been excluded, skip it
      continue
    } else {
      // Entity has been modified, update its properties
      updateEntity(existingEntity, file)
    }
  }

  // Remove entities that no longer exist
  for (const [path, entity] of Object.entries(entityByPath)) {
    if (exclude(entity)) continue
    if (!files.some((file) => file.path === path)) {
      delete entityByPath[path]
    }
  }
}
