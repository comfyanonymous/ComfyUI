import { useToast } from 'primevue/usetoast'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'

import {
  clearPreservedQuery,
  hydratePreservedQuery,
  mergePreservedQueryIntoQuery
} from '@/platform/navigation/preservedQueryManager'
import { PRESERVED_QUERY_NAMESPACES } from '@/platform/navigation/preservedQueryNamespaces'

import { useTeamWorkspaceStore } from '../stores/teamWorkspaceStore'

/**
 * Composable for loading workspace invites from URL query parameters
 *
 * Supports URLs like:
 * - /?invite=TOKEN (accepts workspace invite)
 *
 * The invite token is preserved through login redirects via the
 * preserved query system (sessionStorage), following the same pattern
 * as the template URL loader.
 */
export function useInviteUrlLoader() {
  const route = useRoute()
  const router = useRouter()
  const { t } = useI18n()
  const toast = useToast()
  const workspaceStore = useTeamWorkspaceStore()
  const INVITE_NAMESPACE = PRESERVED_QUERY_NAMESPACES.INVITE

  /**
   * Hydrates preserved query from sessionStorage and merges into route.
   * This restores the invite token after login redirects.
   */
  const ensureInviteQueryFromIntent = async () => {
    hydratePreservedQuery(INVITE_NAMESPACE)
    const mergedQuery = mergePreservedQueryIntoQuery(
      INVITE_NAMESPACE,
      route.query
    )

    if (mergedQuery) {
      await router.replace({ query: mergedQuery })
    }

    return mergedQuery ?? route.query
  }

  /**
   * Removes invite parameter from URL using Vue Router
   */
  const cleanupUrlParams = () => {
    const newQuery = { ...route.query }
    delete newQuery.invite
    void router.replace({ query: newQuery })
  }

  /**
   * Loads and accepts workspace invite from URL query parameters if present.
   * Handles errors internally and shows appropriate user feedback.
   *
   * Flow:
   * 1. Restore preserved query (for post-login redirect)
   * 2. Check for invite token in route.query
   * 3. Accept the invite via API (backend validates token)
   * 4. Show toast notification
   * 5. Clean up URL and preserved query
   */
  const loadInviteFromUrl = async () => {
    // Restore preserved query from sessionStorage (handles login redirect case)
    const query = await ensureInviteQueryFromIntent()
    const inviteParam = query.invite
    if (!inviteParam || typeof inviteParam !== 'string') {
      return
    }

    try {
      const result = await workspaceStore.acceptInvite(inviteParam)

      toast.add({
        severity: 'success',
        summary: t('workspace.inviteAccepted'),
        detail: {
          text: t(
            'workspace.addedToWorkspace',
            { workspaceName: result.workspaceName },
            { escapeParameter: false }
          ),
          workspaceName: result.workspaceName,
          workspaceId: result.workspaceId
        },
        group: 'invite-accepted',
        closable: true
      })
    } catch (error) {
      toast.add({
        severity: 'error',
        summary: t('workspace.inviteFailed'),
        detail: error instanceof Error ? error.message : t('g.unknownError'),
        life: 5000
      })
    } finally {
      cleanupUrlParams()
      clearPreservedQuery(INVITE_NAMESPACE)
    }
  }

  return {
    loadInviteFromUrl
  }
}
