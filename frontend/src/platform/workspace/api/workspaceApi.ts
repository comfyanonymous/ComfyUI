import axios from 'axios'

import { t } from '@/i18n'
import { api } from '@/scripts/api'
import { useFirebaseAuthStore } from '@/stores/firebaseAuthStore'

export type WorkspaceType = 'personal' | 'team'
export type WorkspaceRole = 'owner' | 'member'

interface Workspace {
  id: string
  name: string
  type: WorkspaceType
  created_at: string
  joined_at: string
}

export interface WorkspaceWithRole extends Workspace {
  role: WorkspaceRole
  subscription_tier?: SubscriptionTier
}

export interface Member {
  id: string
  name: string
  email: string
  joined_at: string
  role: WorkspaceRole
}

interface PaginationInfo {
  offset: number
  limit: number
  total: number
}

interface ListMembersResponse {
  members: Member[]
  pagination: PaginationInfo
}

export interface ListMembersParams {
  offset?: number
  limit?: number
}

export interface PendingInvite {
  id: string
  email: string
  token: string
  invited_at: string
  expires_at: string
}

interface ListInvitesResponse {
  invites: PendingInvite[]
}

interface CreateInviteRequest {
  email: string
}

interface AcceptInviteResponse {
  workspace_id: string
  workspace_name: string
}

interface CreateWorkspacePayload {
  name: string
}

interface UpdateWorkspacePayload {
  name: string
}

interface ListWorkspacesResponse {
  workspaces: WorkspaceWithRole[]
}

export type SubscriptionTier =
  | 'FREE'
  | 'STANDARD'
  | 'CREATOR'
  | 'PRO'
  | 'FOUNDERS_EDITION'
export type SubscriptionDuration = 'MONTHLY' | 'ANNUAL'
type PlanAvailabilityReason =
  | 'same_plan'
  | 'incompatible_transition'
  | 'requires_team'
  | 'requires_personal'
  | 'exceeds_max_seats'

interface PlanAvailability {
  available: boolean
  reason?: PlanAvailabilityReason
}

interface PlanSeatSummary {
  seat_count: number
  total_cost_cents: number
  total_credits_cents: number
}

export interface Plan {
  slug: string
  tier: SubscriptionTier
  duration: SubscriptionDuration
  price_cents: number
  credits_cents: number
  max_seats: number
  availability: PlanAvailability
  seat_summary: PlanSeatSummary
}

interface BillingPlansResponse {
  current_plan_slug?: string
  plans: Plan[]
}

type SubscriptionTransitionType =
  | 'new_subscription'
  | 'upgrade'
  | 'downgrade'
  | 'duration_change'

interface PreviewSubscribeRequest {
  plan_slug: string
}

interface SubscribeRequest {
  plan_slug: string
  idempotency_key?: string
  return_url?: string
  cancel_url?: string
}

type SubscribeStatus = 'subscribed' | 'needs_payment_method' | 'pending_payment'

export interface SubscribeResponse {
  billing_op_id: string
  status: SubscribeStatus
  effective_at?: string
  payment_method_url?: string
}

interface CancelSubscriptionRequest {
  idempotency_key?: string
}

interface CancelSubscriptionResponse {
  billing_op_id: string
  cancel_at: string
}

interface ResubscribeRequest {
  idempotency_key?: string
}

interface ResubscribeResponse {
  billing_op_id: string
  status: 'active'
  message?: string
}

interface PaymentPortalRequest {
  return_url?: string
}

interface PaymentPortalResponse {
  url: string
}

interface PreviewPlanInfo {
  slug: string
  tier: SubscriptionTier
  duration: SubscriptionDuration
  price_cents: number
  credits_cents: number
  seat_summary: PlanSeatSummary
  period_start?: string
  period_end?: string
}

export interface PreviewSubscribeResponse {
  allowed: boolean
  reason?: string
  transition_type: SubscriptionTransitionType
  effective_at: string
  is_immediate: boolean
  cost_today_cents: number
  cost_next_period_cents: number
  credits_today_cents: number
  credits_next_period_cents: number
  current_plan?: PreviewPlanInfo
  new_plan: PreviewPlanInfo
}

type BillingSubscriptionStatus = 'active' | 'scheduled' | 'ended' | 'canceled'

type BillingStatus =
  | 'awaiting_payment_method'
  | 'pending_payment'
  | 'paid'
  | 'payment_failed'
  | 'inactive'

export interface BillingStatusResponse {
  is_active: boolean
  subscription_status?: BillingSubscriptionStatus
  subscription_tier?: SubscriptionTier
  subscription_duration?: SubscriptionDuration
  plan_slug?: string
  billing_status?: BillingStatus
  has_funds: boolean
  cancel_at?: string
  renewal_date?: string
}

export interface BillingBalanceResponse {
  amount_micros: number
  prepaid_balance_micros?: number
  cloud_credit_balance_micros?: number
  pending_charges_micros?: number
  effective_balance_micros?: number
  currency: string
}

interface CreateTopupRequest {
  amount_cents: number
  idempotency_key?: string
}

type TopupStatus = 'pending' | 'completed' | 'failed'

interface CreateTopupResponse {
  billing_op_id: string
  topup_id: string
  status: TopupStatus
  amount_cents: number
}

type BillingOpStatus = 'pending' | 'succeeded' | 'failed'

export interface BillingOpStatusResponse {
  id: string
  status: BillingOpStatus
  error_message?: string
  started_at: string
  completed_at?: string
}

interface BillingEvent {
  event_type: string
  event_id: string
  params?: Record<string, unknown>
  createdAt: string
}

interface BillingEventsResponse {
  total: number
  events: BillingEvent[]
  page: number
  limit: number
  totalPages: number
}

interface GetBillingEventsParams {
  page?: number
  limit?: number
}

class WorkspaceApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
    public readonly code?: string
  ) {
    super(message)
    this.name = 'WorkspaceApiError'
  }
}

const workspaceApiClient = axios.create({
  headers: {
    'Content-Type': 'application/json'
  }
})

async function getAuthHeaderOrThrow() {
  const authHeader = await useFirebaseAuthStore().getAuthHeader()
  if (!authHeader) {
    throw new WorkspaceApiError(
      t('toastMessages.userNotAuthenticated'),
      401,
      'NOT_AUTHENTICATED'
    )
  }
  return authHeader
}

async function getFirebaseHeaderOrThrow() {
  const authHeader = await useFirebaseAuthStore().getFirebaseAuthHeader()
  if (!authHeader) {
    throw new WorkspaceApiError(
      t('toastMessages.userNotAuthenticated'),
      401,
      'NOT_AUTHENTICATED'
    )
  }
  return authHeader
}

function handleAxiosError(err: unknown): never {
  if (axios.isAxiosError(err)) {
    const status = err.response?.status
    const message = err.response?.data?.message ?? err.message
    throw new WorkspaceApiError(message, status)
  }
  throw err
}

export const workspaceApi = {
  /**
   * List all workspaces the user has access to
   * GET /api/workspaces
   */
  async list(): Promise<ListWorkspacesResponse> {
    const headers = await getAuthHeaderOrThrow()
    try {
      const response = await workspaceApiClient.get<ListWorkspacesResponse>(
        api.apiURL('/workspaces'),
        { headers }
      )
      return response.data
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Create a new workspace
   * POST /api/workspaces
   */
  async create(payload: CreateWorkspacePayload): Promise<WorkspaceWithRole> {
    const headers = await getAuthHeaderOrThrow()
    try {
      const response = await workspaceApiClient.post<WorkspaceWithRole>(
        api.apiURL('/workspaces'),
        payload,
        { headers }
      )
      return response.data
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Update workspace name
   * PATCH /api/workspaces/:id
   */
  async update(
    workspaceId: string,
    payload: UpdateWorkspacePayload
  ): Promise<WorkspaceWithRole> {
    const headers = await getAuthHeaderOrThrow()
    try {
      const response = await workspaceApiClient.patch<WorkspaceWithRole>(
        api.apiURL(`/workspaces/${workspaceId}`),
        payload,
        { headers }
      )
      return response.data
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Delete a workspace (owner only)
   * DELETE /api/workspaces/:id
   */
  async delete(workspaceId: string): Promise<void> {
    const headers = await getAuthHeaderOrThrow()
    try {
      await workspaceApiClient.delete(
        api.apiURL(`/workspaces/${workspaceId}`),
        {
          headers
        }
      )
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Leave the current workspace.
   * POST /api/workspace/leave
   */
  async leave(): Promise<void> {
    const headers = await getAuthHeaderOrThrow()
    try {
      await workspaceApiClient.post(api.apiURL('/workspace/leave'), null, {
        headers
      })
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * List workspace members (paginated).
   * GET /api/workspace/members
   */
  async listMembers(params?: ListMembersParams): Promise<ListMembersResponse> {
    const headers = await getAuthHeaderOrThrow()
    try {
      const response = await workspaceApiClient.get<ListMembersResponse>(
        api.apiURL('/workspace/members'),
        { headers, params }
      )
      return response.data
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Remove a member from the workspace.
   * DELETE /api/workspace/members/:userId
   */
  async removeMember(userId: string): Promise<void> {
    const headers = await getAuthHeaderOrThrow()
    try {
      await workspaceApiClient.delete(
        api.apiURL(`/workspace/members/${userId}`),
        { headers }
      )
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * List pending invites for the workspace.
   * GET /api/workspace/invites
   */
  async listInvites(): Promise<ListInvitesResponse> {
    const headers = await getAuthHeaderOrThrow()
    try {
      const response = await workspaceApiClient.get<ListInvitesResponse>(
        api.apiURL('/workspace/invites'),
        { headers }
      )
      return response.data
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Create an invite for the workspace.
   * POST /api/workspace/invites
   */
  async createInvite(payload: CreateInviteRequest): Promise<PendingInvite> {
    const headers = await getAuthHeaderOrThrow()
    try {
      const response = await workspaceApiClient.post<PendingInvite>(
        api.apiURL('/workspace/invites'),
        payload,
        { headers }
      )
      return response.data
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Revoke a pending invite.
   * DELETE /api/workspace/invites/:inviteId
   */
  async revokeInvite(inviteId: string): Promise<void> {
    const headers = await getAuthHeaderOrThrow()
    try {
      await workspaceApiClient.delete(
        api.apiURL(`/workspace/invites/${inviteId}`),
        { headers }
      )
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Accept a workspace invite.
   * POST /api/invites/:token/accept
   * Uses Firebase auth (user identity) since the user isn't yet a workspace member.
   */
  async acceptInvite(token: string): Promise<AcceptInviteResponse> {
    const headers = await getFirebaseHeaderOrThrow()
    try {
      const response = await workspaceApiClient.post<AcceptInviteResponse>(
        api.apiURL(`/invites/${token}/accept`),
        null,
        { headers }
      )
      return response.data
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Get billing status for the current workspace
   * GET /api/billing/status
   */
  async getBillingStatus(): Promise<BillingStatusResponse> {
    const headers = await getAuthHeaderOrThrow()
    try {
      const response = await workspaceApiClient.get<BillingStatusResponse>(
        api.apiURL('/billing/status'),
        { headers }
      )
      return response.data
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Get credit balance for the current workspace
   * GET /api/billing/balance
   */
  async getBillingBalance(): Promise<BillingBalanceResponse> {
    const headers = await getAuthHeaderOrThrow()
    try {
      const response = await workspaceApiClient.get<BillingBalanceResponse>(
        api.apiURL('/billing/balance'),
        { headers }
      )
      return response.data
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Get available subscription plans
   * GET /api/billing/plans
   */
  async getBillingPlans(): Promise<BillingPlansResponse> {
    const headers = await getAuthHeaderOrThrow()
    try {
      const response = await workspaceApiClient.get<BillingPlansResponse>(
        api.apiURL('/billing/plans'),
        { headers }
      )
      return response.data
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Preview subscription change
   * POST /api/billing/preview-subscribe
   */
  async previewSubscribe(planSlug: string): Promise<PreviewSubscribeResponse> {
    const headers = await getAuthHeaderOrThrow()
    try {
      const response = await workspaceApiClient.post<PreviewSubscribeResponse>(
        api.apiURL('/billing/preview-subscribe'),
        { plan_slug: planSlug } satisfies PreviewSubscribeRequest,
        { headers }
      )
      return response.data
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Subscribe to a billing plan
   * POST /api/billing/subscribe
   */
  async subscribe(
    planSlug: string,
    returnUrl?: string,
    cancelUrl?: string
  ): Promise<SubscribeResponse> {
    const headers = await getAuthHeaderOrThrow()
    try {
      const response = await workspaceApiClient.post<SubscribeResponse>(
        api.apiURL('/billing/subscribe'),
        {
          plan_slug: planSlug,
          return_url: returnUrl,
          cancel_url: cancelUrl
        } satisfies SubscribeRequest,
        { headers }
      )
      return response.data
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Cancel current subscription
   * POST /api/billing/subscription/cancel
   */
  async cancelSubscription(
    idempotencyKey?: string
  ): Promise<CancelSubscriptionResponse> {
    const headers = await getAuthHeaderOrThrow()
    try {
      const response =
        await workspaceApiClient.post<CancelSubscriptionResponse>(
          api.apiURL('/billing/subscription/cancel'),
          {
            idempotency_key: idempotencyKey
          } satisfies CancelSubscriptionRequest,
          { headers }
        )
      return response.data
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Resubscribe (undo cancel) before period ends
   * POST /api/billing/subscription/resubscribe
   */
  async resubscribe(idempotencyKey?: string): Promise<ResubscribeResponse> {
    const headers = await getAuthHeaderOrThrow()
    try {
      const response = await workspaceApiClient.post<ResubscribeResponse>(
        api.apiURL('/billing/subscription/resubscribe'),
        { idempotency_key: idempotencyKey } satisfies ResubscribeRequest,
        { headers }
      )
      return response.data
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Get Stripe payment portal URL for managing payment methods
   * POST /api/billing/payment-portal
   */
  async getPaymentPortalUrl(
    returnUrl?: string
  ): Promise<PaymentPortalResponse> {
    const headers = await getAuthHeaderOrThrow()
    try {
      const response = await workspaceApiClient.post<PaymentPortalResponse>(
        api.apiURL('/billing/payment-portal'),
        { return_url: returnUrl } satisfies PaymentPortalRequest,
        { headers }
      )
      return response.data
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Create a credit top-up
   * POST /api/billing/topup
   */
  async createTopup(
    amountCents: number,
    idempotencyKey?: string
  ): Promise<CreateTopupResponse> {
    const headers = await getAuthHeaderOrThrow()
    try {
      const response = await workspaceApiClient.post<CreateTopupResponse>(
        api.apiURL('/billing/topup'),
        {
          amount_cents: amountCents,
          idempotency_key: idempotencyKey
        } satisfies CreateTopupRequest,
        { headers }
      )
      return response.data
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Get billing events
   * GET /api/billing/events
   */
  async getBillingEvents(
    params?: GetBillingEventsParams
  ): Promise<BillingEventsResponse> {
    const headers = await getAuthHeaderOrThrow()
    try {
      const response = await workspaceApiClient.get<BillingEventsResponse>(
        api.apiURL('/billing/events'),
        { headers, params }
      )
      return response.data
    } catch (err) {
      handleAxiosError(err)
    }
  },

  /**
   * Get billing operation status
   * GET /api/billing/ops/:id
   */
  async getBillingOpStatus(opId: string): Promise<BillingOpStatusResponse> {
    const headers = await getAuthHeaderOrThrow()
    try {
      const response = await workspaceApiClient.get<BillingOpStatusResponse>(
        api.apiURL(`/billing/ops/${opId}`),
        { headers }
      )
      return response.data
    } catch (err) {
      handleAxiosError(err)
    }
  }
}
