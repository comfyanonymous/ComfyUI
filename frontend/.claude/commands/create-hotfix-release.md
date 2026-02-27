# Create Hotfix Release

This command creates patch/hotfix releases for ComfyUI Frontend by backporting fixes to stable core branches. It handles both automated backports (preferred) and manual cherry-picking (fallback).

**Process Overview:**

1. **Check automated backports first** (via labels)
2. **Skip to version bump** if backports already merged
3. **Manual cherry-picking** if automation failed
4. **Create patch release** with version bump
5. **Publish GitHub release** (manually uncheck "latest")
6. **Update ComfyUI requirements.txt** via PR

<task>
Create a hotfix release by backporting commits/PRs from main to a core branch: $ARGUMENTS

Expected format: Comma-separated list of commits or PR numbers
Examples:

- `#1234,#5678` (PRs - preferred)
- `abc123,def456` (commit hashes)
- `#1234,abc123` (mixed)

If no arguments provided, the command will guide you through identifying commits/PRs to backport.
</task>

## Prerequisites

- Push access to repository
- GitHub CLI (`gh`) authenticated
- Clean working tree
- Understanding of what fixes need backporting

## Hotfix Release Process

### Step 1: Try Automated Backports First

**Check if automated backports were attempted:**

1. **For each PR, check existing backport labels:**

   ```bash
   gh pr view #1234 --json labels | jq -r '.labels[].name'
   ```

2. **If no backport labels exist, add them now:**

   ```bash
   # Add backport labels (this triggers automated backports)
   gh pr edit #1234 --add-label "needs-backport"
   gh pr edit #1234 --add-label "1.24"  # Replace with target version
   ```

3. **Check for existing backport PRs:**

   ```bash
   # Check for backport PRs created by automation
   PR_NUMBER=${ARGUMENTS%%,*}  # Extract first PR number from arguments
   PR_NUMBER=${PR_NUMBER#\#}   # Remove # prefix
   gh pr list --search "backport-${PR_NUMBER}-to" --json number,title,state,baseRefName
   ```

4. **Handle existing backport scenarios:**

   **Scenario A: Automated backports already merged**

   ```bash
   # Check if backport PRs were merged to core branches
   gh pr list --search "backport-${PR_NUMBER}-to" --state merged
   ```

   - If backport PRs are merged → Skip to Step 10 (Version Bump)
   - **CONFIRMATION**: Automated backports completed, proceeding to version bump?

   **Scenario B: Automated backport PRs exist but not merged**

   ```bash
   # Show open backport PRs that need merging
   gh pr list --search "backport-${PR_NUMBER}-to" --state open
   ```

   - **ACTION REQUIRED**: Merge the existing backport PRs first
   - Use: `gh pr merge [PR_NUMBER] --merge` for each backport PR
   - After merging, return to this command and skip to Step 10 (Version Bump)
   - **CONFIRMATION**: Have you merged all backport PRs? Ready to proceed to version bump?

   **Scenario C: No automated backports or they failed**
   - Continue to Step 2 for manual cherry-picking
   - **CONFIRMATION**: Proceeding with manual cherry-picking because automation failed?

### Step 2: Identify Target Core Branch

1. Fetch the current ComfyUI requirements.txt from master branch:
   ```bash
   curl -s https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/requirements.txt | grep "comfyui-frontend-package"
   ```
2. Extract the `comfyui-frontend-package` version (e.g., `comfyui-frontend-package==1.23.4`)
3. Parse version to get major.minor (e.g., `1.23.4` → `1.23`)
4. Determine core branch: `core/<major>.<minor>` (e.g., `core/1.23`)
5. Verify the core branch exists: `git ls-remote origin refs/heads/core/*`
6. **CONFIRMATION REQUIRED**: Is `core/X.Y` the correct target branch?

### Step 3: Parse and Validate Arguments

1. Parse the comma-separated list of commits/PRs
2. For each item:
   - If starts with `#`: Treat as PR number
   - Otherwise: Treat as commit hash
3. For PR numbers:
   - Fetch PR details using `gh pr view <number>`
   - Extract the merge commit if PR is merged
   - If PR has multiple commits, list them all
   - **CONFIRMATION REQUIRED**: Use merge commit or cherry-pick individual commits?
4. Validate all commit hashes exist in the repository

### Step 4: Analyze Target Changes

1. For each commit/PR to cherry-pick:
   - Display commit hash, author, date
   - Show PR title and number (if applicable)
   - Display commit message
   - Show files changed and diff statistics
   - Check if already in core branch: `git branch --contains <commit>`
2. Identify potential conflicts by checking changed files
3. **CONFIRMATION REQUIRED**: Proceed with these commits?

### Step 5: Create Hotfix Branch

1. Checkout the core branch (e.g., `core/1.23`)
2. Pull latest changes: `git pull origin core/X.Y`
3. Display current version from package.json
4. Create hotfix branch: `hotfix/<version>-<timestamp>`
   - Example: `hotfix/1.23.4-20241120`
5. **CONFIRMATION REQUIRED**: Created branch correctly?

### Step 6: Cherry-pick Changes

For each commit:

1. Attempt cherry-pick: `git cherry-pick <commit>`
2. If conflicts occur:
   - Display conflict details
   - Show conflicting sections
   - Provide resolution guidance
   - **CONFIRMATION REQUIRED**: Conflicts resolved correctly?
3. After successful cherry-pick:
   - Show the changes: `git show HEAD`
   - Run validation: `pnpm typecheck && pnpm lint`
4. **CONFIRMATION REQUIRED**: Cherry-pick successful and valid?

### Step 7: Create PR to Core Branch

1. Push the hotfix branch: `git push origin hotfix/<version>-<timestamp>`
2. Create PR using gh CLI:
   ```bash
   gh pr create --base core/X.Y --head hotfix/<version>-<timestamp> \
     --title "[Hotfix] Cherry-pick fixes to core/X.Y" \
     --body "Cherry-picked commits: ..."
   ```
3. Add appropriate labels (but NOT "Release" yet)
4. PR body should include:
   - List of cherry-picked commits/PRs
   - Original issue references
   - Testing instructions
   - Impact assessment
5. **CONFIRMATION REQUIRED**: PR created correctly?

### Step 8: Wait for Tests

1. Monitor PR checks: `gh pr checks`
2. Display test results as they complete
3. If any tests fail:
   - Show failure details
   - Analyze if related to cherry-picks
   - **DECISION REQUIRED**: Fix and continue, or abort?
4. Wait for all required checks to pass
5. **CONFIRMATION REQUIRED**: All tests passing?

### Step 9: Merge Hotfix PR

1. Verify all checks have passed
2. Check for required approvals
3. Merge the PR: `gh pr merge --merge`
4. Delete the hotfix branch
5. **CONFIRMATION REQUIRED**: PR merged successfully?

### Step 10: Create Version Bump

1. Checkout the core branch: `git checkout core/X.Y`
2. Pull latest changes: `git pull origin core/X.Y`
3. Read current version from package.json
4. Determine patch version increment:
   - Current: `1.23.4` → New: `1.23.5`
5. Create release branch named with new version: `release/1.23.5`
6. Update version in package.json to `1.23.5`
7. Commit: `git commit -m "[release] Bump version to 1.23.5"`
8. **CONFIRMATION REQUIRED**: Version bump correct?

### Step 11: Create Release PR

1. Push release branch: `git push origin release/1.23.5`
2. Create PR with Release label:
   ```bash
   gh pr create --base core/X.Y --head release/1.23.5 \
     --title "[Release] v1.23.5" \
     --body "Release notes will be added shortly..." \
     --label "Release"
   ```
3. **CRITICAL**: Verify "Release" label is added
4. Create standardized release notes:

   ```bash
   cat > release-notes-${NEW_VERSION}.md << 'EOF'
   ## ⚠️ Breaking Changes
   <!-- List breaking changes if any, otherwise remove this entire section -->
   - Breaking change description (#PR_NUMBER)

   ---

   ## What's Changed

   ### 🚀 Features
   <!-- List features here, one per line with PR reference -->
   - Feature description (#PR_NUMBER)

   ### 🐛 Bug Fixes
   <!-- List bug fixes here, one per line with PR reference -->
   - Bug fix description (#PR_NUMBER)

   ### 🔧 Maintenance
   <!-- List refactoring, chore, and other maintenance items -->
   - Maintenance item description (#PR_NUMBER)

   ### 📚 Documentation
   <!-- List documentation changes if any, remove section if empty -->
   - Documentation update description (#PR_NUMBER)

   ### ⬆️ Dependencies
   <!-- List dependency updates -->
   - Updated dependency from vX.X.X to vY.Y.Y (#PR_NUMBER)

   **Full Changelog**: https://github.com/Comfy-Org/ComfyUI_frontend/compare/v${CURRENT_VERSION}...v${NEW_VERSION}
   EOF
   ```

   - For hotfixes, typically only populate the "Bug Fixes" section
   - Include links to the cherry-picked PRs/commits
   - Update the PR body with the release notes:
     ```bash
     gh pr edit ${PR_NUMBER} --body-file release-notes-${NEW_VERSION}.md
     ```

5. **CONFIRMATION REQUIRED**: Release PR has "Release" label?

### Step 12: Monitor Release Process

1. Wait for PR checks to pass
2. **FINAL CONFIRMATION**: Ready to trigger release by merging?
3. Merge the PR: `gh pr merge --merge`
4. Monitor release workflow:
   ```bash
   gh run list --workflow=release-draft-create.yaml --limit=1
   gh run watch
   ```
5. Track progress:
   - GitHub release draft/publication
   - PyPI upload
   - pnpm types publication

### Step 13: Manually Publish Draft Release

**CRITICAL**: The release workflow creates a DRAFT release. You must manually publish it:

1. **Go to GitHub Releases:** https://github.com/Comfy-Org/ComfyUI_frontend/releases
2. **Find the DRAFT release** (e.g., "v1.23.5 Draft")
3. **Click "Edit release"**
4. **UNCHECK "Set as the latest release"** ⚠️ **CRITICAL**
   - This prevents the hotfix from showing as "latest"
   - Main branch should always be "latest release"
5. **Click "Publish release"**
6. **CONFIRMATION REQUIRED**: Draft release published with "latest" unchecked?

### Step 14: Create ComfyUI Requirements.txt Update PR

**IMPORTANT**: Create PR to update ComfyUI's requirements.txt via fork:

1. **Setup fork (if needed):**

   ```bash
   # Check if fork already exists
   if gh repo view ComfyUI --json owner | jq -r '.owner.login' | grep -q "$(gh api user --jq .login)"; then
     echo "Fork already exists"
   else
     # Fork the ComfyUI repository
     gh repo fork comfyanonymous/ComfyUI --clone=false
     echo "Created fork of ComfyUI"
   fi
   ```

2. **Clone fork and create branch:**

   ```bash
   # Clone your fork (or use existing clone)
   GITHUB_USER=$(gh api user --jq .login)
   if [ ! -d "ComfyUI-fork" ]; then
     gh repo clone ${GITHUB_USER}/ComfyUI ComfyUI-fork
   fi

   cd ComfyUI-fork
   git checkout master
   git pull origin master

   # Create update branch
   BRANCH_NAME="update-frontend-${NEW_VERSION}"
   git checkout -b ${BRANCH_NAME}
   ```

3. **Update requirements.txt:**

   ```bash
   # Update the version in requirements.txt
   sed -i "s/comfyui-frontend-package==[0-9].*$/comfyui-frontend-package==${NEW_VERSION}/" requirements.txt

   # Verify the change
   grep "comfyui-frontend-package" requirements.txt

   # Commit the change
   git add requirements.txt
   git commit -m "Bump frontend to ${NEW_VERSION}"
   git push origin ${BRANCH_NAME}
   ```

4. **Create PR from fork:**
   ```bash
   # Create PR using gh CLI from fork
   gh pr create \
     --repo comfyanonymous/ComfyUI \
     --title "Bump frontend to ${NEW_VERSION}" \
     --body "$(cat <<EOF
   Bump frontend to ${NEW_VERSION}
   ```

\`\`\`
python main.py --front-end-version Comfy-Org/ComfyUI_frontend@${NEW_VERSION}
\`\`\`

- Diff: [Comfy-Org/ComfyUI_frontend: v${OLD_VERSION}...v${NEW_VERSION}](https://github.com/Comfy-Org/ComfyUI_frontend/compare/v${OLD_VERSION}...v${NEW_VERSION})
- PyPI Package: https://pypi.org/project/comfyui-frontend-package/${NEW_VERSION}/
- npm Types: https://www.npmjs.com/package/@comfyorg/comfyui-frontend-types/v/${NEW_VERSION}

## Changes

- Fix: [Brief description of hotfixes included]
  EOF
  )"

  ```

  ```

5. **Clean up:**

   ```bash
   # Return to original directory
   cd ..

   # Keep fork directory for future updates
   echo "Fork directory 'ComfyUI-fork' kept for future use"
   ```

6. **CONFIRMATION REQUIRED**: ComfyUI requirements.txt PR created from fork?

### Step 15: Post-Release Verification

1. Verify GitHub release:
   ```bash
   gh release view v1.23.5
   ```
2. Check PyPI package:
   ```bash
   pip index versions comfyui-frontend-package | grep 1.23.5
   ```
3. Verify npm package:
   ```bash
   pnpm view @comfyorg/comfyui-frontend-types@1.23.5
   ```
4. Monitor ComfyUI requirements.txt PR for approval/merge
5. Generate release summary with:
   - Version released
   - Commits included
   - Issues fixed
   - Distribution status
   - ComfyUI integration status
6. **CONFIRMATION REQUIRED**: Hotfix release fully completed?

## Safety Checks

Throughout the process:

- Always verify core branch matches ComfyUI's requirements.txt
- For PRs: Ensure using correct commits (merge vs individual)
- Check version numbers follow semantic versioning
- **Critical**: "Release" label must be on version bump PR
- Validate cherry-picks don't break core branch stability
- Keep audit trail of all operations

## Rollback Procedures

If something goes wrong:

- Before push: `git reset --hard origin/core/X.Y`
- After PR creation: Close PR and start over
- After failed release: Create new patch version with fixes
- Document any issues for future reference

## Important Notes

- **Always try automated backports first** - This command is for when automation fails
- Core branch version will be behind main - this is expected
- The "Release" label triggers the PyPI/npm publication
- **CRITICAL**: Always uncheck "Set as latest release" for hotfix releases
- **Must create ComfyUI requirements.txt PR** - Hotfix isn't complete without it
- PR numbers must include the `#` prefix
- Mixed commits/PRs are supported but review carefully
- Always wait for full test suite before proceeding

## Modern Workflow Context

**Primary Backport Method:** Automated via `needs-backport` + `X.YY` labels
**This Command Usage:**

- Smart path detection - skip to version bump if backports already merged
- Fallback to manual cherry-picking only when automation fails/has conflicts
  **Complete Hotfix:** Includes GitHub release publishing + ComfyUI requirements.txt integration

## Workflow Paths

- **Path A:** Backports already merged → Skip to Step 10 (Version Bump)
- **Path B:** Backport PRs need merging → Merge them → Skip to Step 10 (Version Bump)
- **Path C:** No/failed backports → Manual cherry-picking (Steps 2-9) → Version Bump (Step 10)

This process ensures a complete hotfix release with proper GitHub publishing, ComfyUI integration, and multiple safety checkpoints.
