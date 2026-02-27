export interface WorkspaceWithRole {
  id: string
  name: string
  type: 'personal' | 'team'
  role: 'owner' | 'member'
}
