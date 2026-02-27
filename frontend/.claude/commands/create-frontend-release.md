# Create Frontend Release

This command guides you through creating a comprehensive frontend release with semantic versioning analysis, automated change detection, security scanning, and multi-stage human verification.

<task>
Create a frontend release with version type: $ARGUMENTS

Expected format: Version increment type and optional description
Examples:

- `patch` - Bug fixes only
- `minor` - New features, backward compatible
- `major` - Breaking changes
- `prerelease` - Alpha/beta/rc releases
- `patch "Critical security fixes"` - With custom description
- `minor --skip-changelog` - Skip automated changelog generation
- `minor --dry-run` - Simulate release without executing

If no arguments provided, the command will always perform prerelease if the current version is prerelease, or patch in other cases. This command will never perform minor or major releases without explicit direction.
</task>

## Prerequisites

Before starting, ensure:

- You have push access to the repository
- GitHub CLI (`gh`) is authenticated
- You're on a clean main branch working tree
- All intended changes are merged to main
- You understand the scope of changes being released

## Critical Checks Before Starting

### 1. Check Current Version Status

```bash
# Get current version and check if it's a pre-release
CURRENT_VERSION=$(node -p "require('./package.json').version")
if [[ "$CURRENT_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+- ]]; then
  echo "⚠️  Current version $CURRENT_VERSION is a pre-release"
  echo "Consider releasing stable (e.g., 1.24.0-1 → 1.24.0) first"
fi
```

### 2. Find Last Stable Release

```bash
# Get last stable release tag (no pre-release suffix)
LAST_STABLE=$(git tag -l "v*" | grep -v "\-" | sort -V | tail -1)
echo "Last stable release: $LAST_STABLE"
```

## Configuration Options

**Environment Variables:**

- `RELEASE_SKIP_SECURITY_SCAN=true` - Skip security audit
- `RELEASE_AUTO_APPROVE=true` - Skip some confirmation prompts
- `RELEASE_DRY_RUN=true` - Simulate release without executing

## Release Process

### Step 1: Environment Safety Check

1. Verify clean working directory:
   ```bash
   git status --porcelain
   ```
2. Confirm on main branch:
   ```bash
   git branch --show-current
   ```
3. Pull latest changes:
   ```bash
   git pull origin main
   ```
4. Check GitHub CLI authentication:
   ```bash
   gh auth status
   ```
5. Verify npm/PyPI publishing access (dry run)
6. **CONFIRMATION REQUIRED**: Environment ready for release?

### Step 2: Analyze Recent Changes

1. Get current version from package.json
2. **IMPORTANT**: Determine correct base for comparison:
   ```bash
   # If current version is pre-release, use last stable release
   if [[ "$CURRENT_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+- ]]; then
     BASE_TAG=$LAST_STABLE
   else
     BASE_TAG=$(git describe --tags --abbrev=0)
   fi
   ```
3. Find commits since base release (CRITICAL: use --first-parent):
   ```bash
   git log ${BASE_TAG}..HEAD --oneline --no-merges --first-parent
   ```
4. Count total commits:
   ```bash
   COMMIT_COUNT=$(git log ${BASE_TAG}..HEAD --oneline --no-merges --first-parent | wc -l)
   echo "Found $COMMIT_COUNT commits since $BASE_TAG"
   ```
5. Analyze commits for:
   - Breaking changes (BREAKING CHANGE, !, feat())
   - New features (feat:, feature:)
   - Bug fixes (fix:, bugfix:)
   - Documentation changes (docs:)
   - Dependency updates
6. **VERIFY PR TARGET BRANCHES**:
   ```bash
   # Get merged PRs and verify they were merged to main
   gh pr list --state merged --limit 50 --json number,title,baseRefName,mergedAt | \
     jq -r '.[] | select(.baseRefName == "main") | "\(.number): \(.title)"'
   ```
7. **HUMAN ANALYSIS**: Review change summary and verify scope

### Step 3: Breaking Change Analysis

1. Analyze API changes in:
   - Public TypeScript interfaces
   - Extension APIs
   - Component props
   - CLAUDE.md guidelines
2. Check for:
   - Removed public functions/classes
   - Changed function signatures
   - Deprecated feature removals
   - Configuration changes
3. Generate breaking change summary
4. **COMPATIBILITY REVIEW**: Breaking changes documented and justified?

### Step 4: Analyze Dependency Updates

1. **Use pnpm's built-in dependency analysis:**

   ```bash
   # Get outdated dependencies with pnpm
   pnpm outdated --format table > outdated-deps-${NEW_VERSION}.txt

   # Check for license compliance
   pnpm licenses ls --json > licenses-${NEW_VERSION}.json

   # Analyze why specific dependencies exist
   echo "Dependency analysis:" > dep-analysis-${NEW_VERSION}.md
   MAJOR_DEPS=("vue" "vite" "@vitejs/plugin-vue" "typescript" "pinia")
   for dep in "${MAJOR_DEPS[@]}"; do
     echo -e "\n## $dep\n\`\`\`" >> dep-analysis-${NEW_VERSION}.md
     pnpm why "$dep" >> dep-analysis-${NEW_VERSION}.md || echo "Not found" >> dep-analysis-${NEW_VERSION}.md
     echo "\`\`\`" >> dep-analysis-${NEW_VERSION}.md
   done
   ```

2. **Check for significant dependency updates:**

   ```bash
   # Extract all dependency changes for major version bumps
   OTHER_DEP_CHANGES=""

   # Compare major dependency versions (you can extend this list)
   MAJOR_DEPS=("vue" "vite" "@vitejs/plugin-vue" "typescript" "pinia")

   for dep in "${MAJOR_DEPS[@]}"; do
     PREV_VER=$(echo "$PREV_PACKAGE_JSON" | grep -o "\"$dep\": \"[^\"]*\"" | grep -o '[0-9][^"]*' | head -1 || echo "")
     CURR_VER=$(echo "$CURRENT_PACKAGE_JSON" | grep -o "\"$dep\": \"[^\"]*\"" | grep -o '[0-9][^"]*' | head -1 || echo "")

     if [ "$PREV_VER" != "$CURR_VER" ] && [ -n "$PREV_VER" ] && [ -n "$CURR_VER" ]; then
       # Check if it's a major version change
       PREV_MAJOR=$(echo "$PREV_VER" | cut -d. -f1 | sed 's/[^0-9]//g')
       CURR_MAJOR=$(echo "$CURR_VER" | cut -d. -f1 | sed 's/[^0-9]//g')

       if [ "$PREV_MAJOR" != "$CURR_MAJOR" ]; then
         OTHER_DEP_CHANGES="${OTHER_DEP_CHANGES}\n- **${dep}**: ${PREV_VER} → ${CURR_VER} (Major version change)"
       fi
     fi
   done
   ```

### Step 5: Generate GTM Feature Summary

1. **Collect PR data for analysis:**

   ```bash
   # Get list of PR numbers from commits
   PR_NUMBERS=$(git log ${BASE_TAG}..HEAD --oneline --no-merges --first-parent | \
     grep -oE "#[0-9]+" | tr -d '#' | sort -u)

   # Save PR data for each PR
   echo "[" > prs-${NEW_VERSION}.json
   first=true
   for PR in $PR_NUMBERS; do
     [[ "$first" == true ]] && first=false || echo "," >> prs-${NEW_VERSION}.json
     gh pr view $PR --json number,title,author,body,labels 2>/dev/null >> prs-${NEW_VERSION}.json || echo "{}" >> prs-${NEW_VERSION}.json
   done
   echo "]" >> prs-${NEW_VERSION}.json
   ```

2. **Analyze for GTM-worthy features:**

   ```
   <task>
   Review these PRs to identify features worthy of marketing attention.

   A feature is GTM-worthy if it meets ALL of these criteria:
   - Introduces a NEW capability users didn't have before (not just improvements)
   - Would be a compelling reason for users to upgrade to this version
   - Can be demonstrated visually or has clear before/after comparison
   - Affects a significant portion of the user base

   NOT GTM-worthy:
   - Bug fixes (even important ones)
   - Minor UI tweaks or color changes
   - Performance improvements without user-visible impact
   - Internal refactoring
   - Small convenience features
   - Features that only improve existing functionality marginally

   For each GTM-worthy feature, note:
   - PR number, title, and author
   - Media links from the PR description
   - One compelling sentence on why users should care

   If there are no GTM-worthy features, just say "No marketing-worthy features in this release."
   </task>

   PR data: [contents of prs-${NEW_VERSION}.json]
   ```

3. **Generate GTM notification using this EXACT Slack-compatible format:**

   ```bash
   # Only create file if GTM-worthy features exist:
   if [ "$GTM_FEATURES_FOUND" = "true" ]; then
     cat > gtm-summary-${NEW_VERSION}.md << 'EOF'
   *GTM Summary: ComfyUI Frontend v${NEW_VERSION}*

   _Disclaimer: the below is AI-generated_

   1. *[Feature Title]* (#[PR_NUMBER])
       * *Author:* @[username]
       * *Demo:* [Media Link or "No demo available"]
       * *Why users should care:* [One compelling sentence]
       * *Key Features:*
           * [Feature detail 1]
           * [Feature detail 2]

   2. *[Feature Title]* (#[PR_NUMBER])
       * *Author:* @[username]
       * *Demo:* [Media Link]
       * *Why users should care:* [One compelling sentence]
       * *Key Features:*
           * [Feature detail 1]
           * [Feature detail 2]
   EOF
     echo "📋 GTM summary saved to: gtm-summary-${NEW_VERSION}.md"
     echo "📤 Share this file in #gtm channel to notify the team"
   else
     echo "✅ No GTM notification needed for this release"
     echo "📄 No gtm-summary file created - no marketing-worthy features"
   fi
   ```

   **CRITICAL Formatting Requirements:**
   - Use single asterisk (\*) for emphasis, NOT double (\*\*)
   - Use underscore (\_) for italics
   - Use 4 spaces for indentation (not tabs)
   - Convert author names to @username format (e.g., "John Smith" → "@john")
   - No section headers (#), no code language specifications
   - Always include "Disclaimer: the below is AI-generated"
   - Keep content minimal - no testing instructions, additional sections, etc.

### Step 6: Version Preview

**Version Preview:**

- Current: `${CURRENT_VERSION}`
- Proposed: Show exact version number based on analysis:
  - Major version if breaking changes detected
  - Minor version if new features added
  - Patch version if only bug fixes
- **CONFIRMATION REQUIRED**: Proceed with version `X.Y.Z`?

### Step 7: Security and Dependency Audit

1. Run pnpm security audit:
   ```bash
   pnpm audit --audit-level moderate
   pnpm licenses ls --summary
   ```
2. Check for known vulnerabilities in dependencies
3. Run comprehensive dependency health check:
   ```bash
   pnpm doctor
   ```
4. Scan for hardcoded secrets or credentials:
   ```bash
   git log -p ${BASE_TAG}..HEAD | grep -iE "(password|key|secret|token)" || echo "No sensitive data found"
   ```
5. Verify no sensitive data in recent commits
6. **SECURITY REVIEW**: Address any critical findings before proceeding?

### Step 8: Pre-Release Testing

1. Run complete test suite:
   ```bash
   pnpm test:unit
   ```
2. Run type checking:
   ```bash
   pnpm typecheck
   ```
3. Run linting (may have issues with missing packages):
   ```bash
   pnpm lint || echo "Lint issues - verify if critical"
   ```
4. Test build process:
   ```bash
   pnpm build
   pnpm build:types
   ```
5. **QUALITY GATE**: All tests and builds passing?

### Step 9: Generate Comprehensive Release Notes

1. Extract commit messages since base release:
   ```bash
   git log ${BASE_TAG}..HEAD --oneline --no-merges --first-parent > commits.txt
   ```
2. **CRITICAL**: Verify PR inclusion by checking merge location:
   ```bash
   # For each significant PR mentioned, verify it's on main
   for PR in ${SIGNIFICANT_PRS}; do
     COMMIT=$(gh pr view $PR --json mergeCommit -q .mergeCommit.oid)
     git branch -r --contains $COMMIT | grep -q "origin/main" || \
       echo "WARNING: PR #$PR not on main branch!"
   done
   ```
3. Create standardized release notes using this exact template:

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

   **Full Changelog**: https://github.com/Comfy-Org/ComfyUI_frontend/compare/${BASE_TAG}...v${NEW_VERSION}
   EOF
   ```

4. **Parse commits and populate template:**
   - Group commits by conventional commit type (feat:, fix:, chore:, etc.)
   - Extract PR numbers from commit messages
   - For breaking changes, analyze if changes affect:
     - Public APIs (app object, api module)
     - Extension/workspace manager APIs
     - Node schema, workflow schema, or other public schemas
     - Any other public-facing interfaces
   - For dependency updates, list version changes with PR numbers
   - Remove empty sections (e.g., if no documentation changes)
   - Ensure consistent bullet format: `- Description (#PR_NUMBER)`
5. **CONTENT REVIEW**: Release notes follow standard format?

### Step 10: Create Version Bump PR

**For standard version bumps (patch/minor/major):**

```bash
# Trigger the workflow
gh workflow run release-version-bump.yaml -f version_type=${VERSION_TYPE}

# Workflow runs quickly - usually creates PR within 30 seconds
echo "Workflow triggered. Waiting for PR creation..."
```

**For releasing a stable version:**

1. Must manually create branch and update version:

   ```bash
   git checkout -b version-bump-${NEW_VERSION}
   # Edit package.json to remove pre-release suffix
   git add package.json
   git commit -m "${NEW_VERSION}"
   git push origin version-bump-${NEW_VERSION}
   ```

2. Wait for PR creation (if using workflow) or create manually:

   ```bash
   # For workflow-created PRs - wait and find it
   sleep 30
   # Look for PR from comfy-pr-bot (not github-actions)
   PR_NUMBER=$(gh pr list --author comfy-pr-bot --limit 1 --json number --jq '.[0].number')

   # Verify we got the PR
   if [ -z "$PR_NUMBER" ]; then
     echo "PR not found yet. Checking recent PRs..."
     gh pr list --limit 5 --json number,title,author
   fi

   # For manual PRs
   gh pr create --title "${NEW_VERSION}" \
     --body-file release-notes-${NEW_VERSION}.md \
     --label "Release"
   ```

3. **Update PR with release notes:**
   ```bash
   # For workflow-created PRs, update the body with our release notes
   gh pr edit ${PR_NUMBER} --body-file release-notes-${NEW_VERSION}.md
   ```
4. **PR REVIEW**: Version bump PR created with standardized release notes?

### Step 11: Critical Release PR Verification

1. **CRITICAL**: Verify PR has "Release" label:
   ```bash
   gh pr view ${PR_NUMBER} --json labels | jq -r '.labels[].name' | grep -q "Release" || \
     echo "ERROR: Release label missing! Add it immediately!"
   ```
2. Verify version number in package.json
3. Review all changed files
4. Ensure no unintended changes included
5. Wait for required PR checks:
   ```bash
   gh pr checks ${PR_NUMBER} --watch
   ```
6. **FINAL CODE REVIEW**: Release label present and no [skip ci]?

### Step 12: Pre-Merge Validation

1. **Review Requirements**: Release PRs require approval
2. Monitor CI checks
3. Check no new commits to main since PR creation
4. **DEPLOYMENT READINESS**: Ready to merge?

### Step 13: Execute Release

1. **FINAL CONFIRMATION**: Merge PR to trigger release?
2. Merge the Release PR:
   ```bash
   gh pr merge ${PR_NUMBER} --merge
   ```
3. **IMMEDIATELY CHECK**: Did release workflow trigger?
   ```bash
   sleep 10
   gh run list --workflow=release-draft-create.yaml --limit=1
   ```
4. **For Minor/Major Version Releases**: The release-branch-create workflow will automatically:
   - Create a `core/x.yy` branch for the PREVIOUS minor version
   - Apply branch protection rules
   - Document the feature freeze policy
   ```bash
   # Monitor branch creation (for minor/major releases)
   gh run list --workflow=release-branch-create.yaml --limit=1
   ```
5. If workflow didn't trigger due to [skip ci]:
   ```bash
   echo "ERROR: Release workflow didn't trigger!"
   echo "Options:"
   echo "1. Create patch release (e.g., 1.24.1) to trigger workflow"
   echo "2. Investigate manual release options"
   ```
6. If workflow triggered, monitor execution:
   ```bash
   WORKFLOW_RUN_ID=$(gh run list --workflow=release-draft-create.yaml --limit=1 --json databaseId --jq '.[0].databaseId')
   gh run watch ${WORKFLOW_RUN_ID}
   ```

### Step 14: Enhance GitHub Release

1. Wait for automatic release creation:

   ```bash
   # Wait for release to be created
   while ! gh release view v${NEW_VERSION} >/dev/null 2>&1; do
     echo "Waiting for release creation..."
     sleep 10
   done
   ```

2. **Enhance the GitHub release:**

   ```bash
   # Update release with our release notes
   gh release edit v${NEW_VERSION} \
     --title "🚀 ComfyUI Frontend v${NEW_VERSION}" \
     --notes-file release-notes-${NEW_VERSION}.md \
     --latest

   # Add any additional assets if needed
   # gh release upload v${NEW_VERSION} additional-assets.zip
   ```

3. **Verify release details:**
   ```bash
   gh release view v${NEW_VERSION}
   ```

### Step 15: Verify Multi-Channel Distribution

1. **GitHub Release:**

   ```bash
   gh release view v${NEW_VERSION} --json assets,body,createdAt,tagName
   ```

   - ✅ Check release notes
   - ✅ Verify dist.zip attachment
   - ✅ Confirm release marked as latest (for main branch)

2. **PyPI Package:**

   ```bash
   # Check PyPI availability (may take a few minutes)
   for i in {1..10}; do
     if curl -s https://pypi.org/pypi/comfyui-frontend-package/json | jq -r '.releases | keys[]' | grep -q ${NEW_VERSION}; then
       echo "✅ PyPI package available"
       break
     fi
     echo "⏳ Waiting for PyPI package... (attempt $i/10)"
     sleep 30
   done
   ```

3. **npm Package:**

   ```bash
   # Check npm availability
   for i in {1..10}; do
     if pnpm view @comfyorg/comfyui-frontend-types@${NEW_VERSION} version >/dev/null 2>&1; then
       echo "✅ npm package available"
       break
     fi
     echo "⏳ Waiting for npm package... (attempt $i/10)"
     sleep 30
   done
   ```

4. **DISTRIBUTION VERIFICATION**: All channels published successfully?

### Step 16: Post-Release Monitoring Setup

1. **Monitor immediate release health:**

   ```bash
   # Check for immediate issues
   gh issue list --label "bug" --state open --limit 5 --json title,number,createdAt

   # Monitor download metrics (if accessible)
   gh release view v${NEW_VERSION} --json assets --jq '.assets[].downloadCount'
   ```

2. **Update documentation tracking:**

   ```bash
   cat > post-release-checklist.md << EOF
   # Post-Release Checklist for v${NEW_VERSION}

   ## Immediate Tasks (Next 24 hours)
   - [ ] Monitor error rates and user feedback
   - [ ] Watch for critical issues
   - [ ] Verify documentation is up to date
   - [ ] Check community channels for questions

   ## Short-term Tasks (Next week)
   - [ ] Update external integration guides
   - [ ] Monitor adoption metrics
   - [ ] Gather user feedback
   - [ ] Plan next release cycle

   ## Long-term Tasks
   - [ ] Analyze release process improvements
   - [ ] Update release templates based on learnings
   - [ ] Document any new patterns discovered

   ## Key Metrics to Track
   - Download counts: GitHub, PyPI, npm
   - Issue reports related to v${NEW_VERSION}
   - Community feedback and adoption
   - Performance impact measurements
   EOF
   ```

3. **Create release summary:**

   ```bash
   cat > release-summary-${NEW_VERSION}.md << EOF
   # Release Summary: ComfyUI Frontend v${NEW_VERSION}

   **Released:** $(date)
   **Type:** ${VERSION_TYPE}
   **Duration:** ~${RELEASE_DURATION} minutes
   **Release Commit:** ${RELEASE_COMMIT}

   ## Metrics
   - **Commits Included:** ${COMMITS_COUNT}
   - **Contributors:** ${CONTRIBUTORS_COUNT}
   - **Files Changed:** ${FILES_CHANGED}
   - **Lines Added/Removed:** +${LINES_ADDED}/-${LINES_REMOVED}

   ## Distribution Status
   - ✅ GitHub Release: Published
   - ✅ PyPI Package: Available
   - ✅ npm Types: Available

   ## Next Steps
   - Monitor for 24-48 hours
   - Address any critical issues immediately
   - Plan next release cycle

   ## Files Generated
   - \`release-notes-${NEW_VERSION}.md\` - Comprehensive release notes
   - \`post-release-checklist.md\` - Follow-up tasks
   - \`gtm-summary-${NEW_VERSION}.md\` - Marketing team notification
   EOF
   ```

4. **RELEASE COMPLETION**: All post-release setup completed?

### Step 17: Create Release Summary

1. **Create comprehensive release summary:**

   ```bash
   cat > release-summary-${NEW_VERSION}.md << EOF
   # Release Summary: ComfyUI Frontend v${NEW_VERSION}

   **Released:** $(date)
   **Type:** ${VERSION_TYPE}
   **Duration:** ~${RELEASE_DURATION} minutes
   **Release Commit:** ${RELEASE_COMMIT}

   ## Metrics
   - **Commits Included:** ${COMMITS_COUNT}
   - **Contributors:** ${CONTRIBUTORS_COUNT}
   - **Files Changed:** ${FILES_CHANGED}
   - **Lines Added/Removed:** +${LINES_ADDED}/-${LINES_REMOVED}

   ## Distribution Status
   - ✅ GitHub Release: Published
   - ✅ PyPI Package: Available
   - ✅ npm Types: Available

   ## Next Steps
   - Monitor for 24-48 hours
   - Address any critical issues immediately
   - Plan next release cycle

   ## Files Generated
   - \`release-notes-${NEW_VERSION}.md\` - Comprehensive release notes
   - \`post-release-checklist.md\` - Follow-up tasks
   - \`gtm-summary-${NEW_VERSION}.md\` - Marketing team notification
   EOF
   ```

2. **RELEASE COMPLETION**: All steps completed successfully?

## Advanced Safety Features

### Rollback Procedures

**Pre-Merge Rollback:**

```bash
# Close version bump PR and reset
gh pr close ${PR_NUMBER}
git reset --hard origin/main
git clean -fd
```

**Post-Merge Rollback:**

```bash
# Create immediate patch release with reverts
git revert ${RELEASE_COMMIT}
# Follow this command again with patch version
```

**Emergency Procedures:**

```bash
# Document incident
cat > release-incident-${NEW_VERSION}.md << EOF
# Release Incident Report

**Version:** ${NEW_VERSION}
**Issue:** [Describe the problem]
**Impact:** [Severity and scope]
**Resolution:** [Steps taken]
**Prevention:** [Future improvements]
EOF

# Contact package registries for critical issues
echo "For critical security issues, consider:"
echo "- PyPI: Contact support for package yanking"
echo "- npm: Use 'npm unpublish' within 72 hours"
echo "- GitHub: Update release with warning notes"
```

### Quality Gates Summary

The command implements multiple quality gates:

1. **🔒 Security Gate**: Vulnerability scanning, secret detection
2. **🧪 Quality Gate**: Unit and component tests, linting, type checking
3. **📋 Content Gate**: Changelog accuracy, release notes quality
4. **🔄 Process Gate**: Release timing verification
5. **✅ Verification Gate**: Multi-channel publishing confirmation
6. **📊 Monitoring Gate**: Post-release health tracking

## Common Scenarios

### Scenario 1: Regular Feature Release

```bash
/project:create-frontend-release minor
```

- Analyzes features since last release
- Generates changelog automatically
- Creates comprehensive release notes

### Scenario 2: Critical Security Patch

```bash
/project:create-frontend-release patch "Security fixes for CVE-2024-XXXX"
```

- Expedited security scanning
- Enhanced monitoring setup

### Scenario 3: Major Version with Breaking Changes

```bash
/project:create-frontend-release major
```

- Comprehensive breaking change analysis
- Migration guide generation

### Scenario 4: Pre-release Testing

```bash
/project:create-frontend-release prerelease
```

- Creates alpha/beta/rc versions
- Draft release status
- Python package specs require that prereleases use alpha/beta/rc as the preid

## Critical Implementation Notes

When executing this release process, pay attention to these key aspects:

### Version Handling

- For pre-release versions (e.g., 1.24.0-rc.1), the next stable release should be the same version without the suffix (1.24.0)
- Never skip version numbers - follow semantic versioning strictly

### Commit History Analysis

- **ALWAYS** use `--first-parent` flag with git log to avoid including commits from merged feature branches
- Verify PR merge targets before including them in changelogs:
  ```bash
  gh pr view ${PR_NUMBER} --json baseRefName
  ```

### Release Workflow Triggers

- The "Release" label on the PR is **CRITICAL** - without it, PyPI/npm publishing won't occur
- Check for `[skip ci]` in commit messages before merging - this blocks the release workflow
- If you encounter `[skip ci]`, push an empty commit to override it:
  ```bash
  git commit --allow-empty -m "Trigger release workflow"
  ```

### PR Creation Details

- Version bump PRs come from `comfy-pr-bot`, not `github-actions`
- The workflow typically completes in 20-30 seconds
- Always wait for the PR to be created before trying to edit it

### Breaking Changes Detection

- Analyze changes to public-facing APIs:
  - The `app` object and its methods
  - The `api` module exports
  - Extension and workspace manager interfaces
  - Node schema, workflow schema, and other public schemas
- Any modifications to these require marking as breaking changes

### Recovery Procedures

If the release workflow fails to trigger:

1. Create a revert PR to restore the previous version
2. Merge the revert
3. Re-run the version bump workflow
4. This approach is cleaner than creating extra version numbers
