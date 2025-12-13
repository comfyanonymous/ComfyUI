# Nano Banana Environment Configuration

## Issues Identified
1. **ComfyUI-Manager Security Alert**: ComfyUI version outdated; Manager in frozen mode (installations blocked). Defer to separate branch.
2. **API Key Missing**: `GOOGLE_API_KEY` not configured; Nano Banana fails with "No valid credentials found."

## Design Decisions
- **Local Docker**: Use `.env` file loaded via `docker-compose.yml` `env_file` directive
- **RunPod Serverless**: Environment variables set in template interface (per [RunPod docs](https://docs.runpod.io/serverless/development/environment-variables)); code reads `os.environ`
- **Nano Banana Auth**: Supports two methods:
  - API approach: `GOOGLE_API_KEY` env var (simpler, primary)
  - Vertex AI: `PROJECT_ID` + `LOCATION` + ADC (optional, document only)

## Implementation Steps

### Step 1: Create PLAN.md and .env.example
**IMPLEMENTATION:**
- Create `PLAN.md` at repo root with this plan (living document)
- Create `.env.example` with `GOOGLE_API_KEY` placeholder and commented Vertex AI vars
- Expected outcome: Planning artifact tracked; env template ready
- Success criteria: Both files committed; `.env.example` has clear comments

**GIT TRACKING:**
- Commit after both files created
- Message: `[ENV-1] Add plan and env example for Nano Banana API keys`
- Branch: `feature/comfy-nano-banana-setup` (existing)
- Push: After commit for visibility
- PR: Update draft PR description with checklist

**USER FEEDBACK TOUCHPOINT:**
- Who: {users}
- What to show: PR diff showing `.env.example` content
- Feedback needed: "Does `.env.example` cover your needs? Any additional vars?"
- Blocking: Non-blocking (proceed if no response)

**CHECKPOINT:**
- Natural stopping point: After commit
- Rollback: Delete files and revert commit

---

### Step 2: Wire .env into docker-compose.yml
**IMPLEMENTATION:**
- Add `env_file: .env` to `comfyui` service
- Add `GOOGLE_API_KEY` to `environment` section (passes through from `.env`)
- Expected outcome: Container receives env vars from `.env` file
- Success criteria: `docker-compose config` shows env_file and environment vars

**GIT TRACKING:**
- Commit after docker-compose.yml change
- Message: `[ENV-2] Wire .env file into docker-compose for Nano Banana`
- Branch: Same feature branch
- Push: After commit
- PR: Update checklist, add compose snippet to PR description

**USER FEEDBACK TOUCHPOINT:**
- Who: {users}
- What to show: Diff of docker-compose.yml changes
- Feedback needed: "Confirm env_file approach works for your local setup"
- Blocking: Non-blocking

**CHECKPOINT:**
- Natural stopping point: After commit
- Rollback: Revert docker-compose.yml change

---

### Step 3: Protect secrets in Git
**IMPLEMENTATION:**
- Add `.env` to `.gitignore`
- Ensure `.env.example` remains tracked (not ignored)
- Expected outcome: Secrets never committed
- Success criteria: `git status` shows `.env` ignored; `.env.example` tracked

**GIT TRACKING:**
- Commit after .gitignore update
- Message: `[ENV-3] Ignore .env file, keep example tracked`
- Branch: Same feature branch
- Push: After commit
- PR: Update checklist

**USER FEEDBACK TOUCHPOINT:**
- Who: {users}
- What to show: .gitignore diff
- Feedback needed: "Confirm .env should be ignored"
- Blocking: Non-blocking

**CHECKPOINT:**
- Natural stopping point: After commit
- Rollback: Revert .gitignore change

---

### Step 4: Local validation with real API key
**IMPLEMENTATION:**
- Create local `.env` file (not committed) with user's `GOOGLE_API_KEY`
- Restart container: `docker restart comfy`
- Verify in logs: No "No valid credentials found" error
- Test in UI: Nano Banana nodes visible and functional
- Expected outcome: Node authenticates successfully
- Success criteria: Logs show successful auth; nodes work in ComfyUI UI

**GIT TRACKING:**
- Commit (empty or documentation) after validation
- Message: `[ENV-4] Validate Nano Banana with env-based API key - validates auth`
- Branch: Same feature branch
- Push: After commit
- PR: Update checklist, attach log snippet showing successful auth

**USER FEEDBACK TOUCHPOINT:**
- Who: {users}
- What to show: Log snippet showing successful auth; screenshot of nodes in UI
- Feedback needed: "Confirm API key authentication works; nodes functional?"
- Blocking: Non-blocking (preferred before merge)

**CHECKPOINT:**
- Natural stopping point: After validation commit
- Rollback: Remove `.env`, restart container, verify error returns

---

### Step 5: Document RunPod configuration
**IMPLEMENTATION:**
- Update PR description with RunPod env var setup instructions
- Add note: Set `GOOGLE_API_KEY` in RunPod template env vars (no `.env` file needed)
- RunPod specific configuration instructions:
  - Go to RunPod console -> Templates -> [Your Template] -> Edit
  - Scroll to Environment Variables
  - Key: `GOOGLE_API_KEY`, Value: [Your API Key]
  - No need for `.env` file in the container image
  - Code reads `os.environ` which works for both methods
- Expected outcome: Clear instructions for RunPod deployment
- Success criteria: PR description has RunPod section with env var guidance

**GIT TRACKING:**
- Commit (empty or documentation)
- Message: `[ENV-5] Document RunPod env configuration for Nano Banana`
- Branch: Same feature branch
- Push: After commit
- PR: Update PR description with RunPod section

**USER FEEDBACK TOUCHPOINT:**
- Who: {users}
- What to show: PR description RunPod section
- Feedback needed: "Does RunPod env var guidance match your setup?"
- Blocking: Non-blocking

**CHECKPOINT:**
- Natural stopping point: After commit
- Rollback: Edit PR description

---

### Step 6: Cleanup planning artifact (pre-merge)
**IMPLEMENTATION:**
- Copy final `PLAN.md` content to PR description
- Delete `PLAN.md` file
- Expected outcome: Clean main branch; plan preserved in PR
- Success criteria: `PLAN.md` removed; PR description has complete plan

**GIT TRACKING:**
- Final commit before merge
- Message: `[ENV-6] Cleanup planning artifact - plan moved to PR description`
- Branch: Same feature branch
- Push: After commit
- PR: Final PR description update

**USER FEEDBACK TOUCHPOINT:**
- Who: {users}
- What to show: Final PR ready for review
- Feedback needed: "Ready for final review and merge?"
- Blocking: Non-blocking

**CHECKPOINT:**
- Natural stopping point: Before merge
- Rollback: Restore `PLAN.md` if needed

---

## Communication Templates

### After Step 1 (Initial Setup)
**Notify {users} via PR comment:**
```
✅ Step 1 Complete: Plan and .env.example created

What's done:
- Added PLAN.md (living plan document)
- Created .env.example template with GOOGLE_API_KEY

What you can try:
- Review .env.example: [link to file in PR]

Specific feedback needed:
- Does .env.example cover your needs?
- Any additional environment variables needed?

What's next:
- Will proceed with docker-compose.yml integration while waiting for feedback
```

### After Step 4 (Validation)
**Notify {users} via PR comment:**
```
✅ Step 4 Complete: Local validation successful

What's done:
- Wired .env into docker-compose.yml
- Validated with real API key
- Nano Banana nodes authenticating successfully

What you can try:
- Test at http://localhost:8188
- Log snippet: [attach log showing successful auth]
- Screenshot: [attach UI showing nodes]

Specific feedback needed:
- Confirm API key authentication works for you?
- Nodes functional in UI?

What's next:
- Will document RunPod configuration while waiting for feedback
```

---

## Rollback Strategy

**If env config breaks container:**
```bash
# Remove .env
rm .env
# Restart without env file
docker restart comfy
# Revert docker-compose.yml if needed
git revert [commit-hash]
```

**If validation fails:**
- Keep error logs in commit message
- Document in PR description
- Branch from last good commit if major changes needed

---

## Backlog Items (Separate Branch)
- [ ] ComfyUI version update to resolve Manager frozen mode
- [ ] OpenTelemetry tracing for Nano Banana errors

