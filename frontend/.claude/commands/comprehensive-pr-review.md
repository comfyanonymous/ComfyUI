# Comprehensive PR Review for ComfyUI Frontend

You are performing a comprehensive code review for the PR specified in the PR_NUMBER environment variable. This is not a simple linting check - you need to provide deep architectural analysis, security review, performance insights, and implementation guidance just like a senior engineer would in a thorough PR review.

## CRITICAL INSTRUCTIONS

**You MUST post individual inline comments on specific lines of code. DO NOT create a single summary comment until the very end.**

**IMPORTANT: You have full permission to execute gh api commands. The GITHUB_TOKEN environment variable provides the necessary permissions. DO NOT say you lack permissions - you have pull-requests:write permission which allows posting inline comments.**

To post inline comments, you will use the GitHub API via the `gh` command. Here's how:

1. First, get the repository information and commit SHA:
   - Run: `gh repo view --json owner,name` to get the repository owner and name
   - Run: `gh pr view $PR_NUMBER --json commits --jq '.commits[-1].oid'` to get the latest commit SHA

2. For each issue you find, post an inline comment using this exact command structure (as a single line):

   ```
   gh api --method POST -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" /repos/OWNER/REPO/pulls/$PR_NUMBER/comments -f body="YOUR_COMMENT_BODY" -f commit_id="COMMIT_SHA" -f path="FILE_PATH" -F line=LINE_NUMBER -f side="RIGHT"
   ```

3. Format your comment body using actual newlines in the command. Use a heredoc or construct the body with proper line breaks:
   ```
   COMMENT_BODY="**[category] severity Priority**
   ```

**Issue**: Brief description of the problem
**Context**: Why this matters  
**Suggestion**: How to fix it"

```

Then use: `-f body="$COMMENT_BODY"`

## Phase 1: Environment Setup and PR Context

### Step 1.1: Initialize Review Tracking

First, create variables to track your review metrics. Keep these in memory throughout the review:
- CRITICAL_COUNT = 0
- HIGH_COUNT = 0
- MEDIUM_COUNT = 0
- LOW_COUNT = 0
- ARCHITECTURE_ISSUES = 0
- SECURITY_ISSUES = 0
- PERFORMANCE_ISSUES = 0
- QUALITY_ISSUES = 0

### Step 1.2: Validate Environment

1. Check that PR_NUMBER environment variable is set. If not, exit with error.
2. Run `gh pr view $PR_NUMBER --json state` to verify the PR exists and is open.
3. Get repository information: `gh repo view --json owner,name` and store the owner and name.
4. Get the latest commit SHA: `gh pr view $PR_NUMBER --json commits --jq '.commits[-1].oid'` and store it.

### Step 1.3: Checkout PR Branch Locally

This is critical for better file inspection:

1. Get PR metadata: `gh pr view $PR_NUMBER --json files,title,body,additions,deletions,baseRefName,headRefName > pr_info.json`
2. Extract branch names from pr_info.json using jq
3. Fetch and checkout the PR branch:
```

git fetch origin "pull/$PR_NUMBER/head:pr-$PR_NUMBER"
git checkout "pr-$PR_NUMBER"

```

### Step 1.4: Get Changed Files and Diffs

Use git locally for much faster analysis:

1. Get list of changed files: `git diff --name-only "$BASE_SHA" > changed_files.txt`
2. Get the full diff: `git diff "$BASE_SHA" > pr_diff.txt`
3. Get detailed file changes with status: `git diff --name-status "$BASE_SHA" > file_changes.txt`

### Step 1.5: Create Analysis Cache

Set up caching to avoid re-analyzing unchanged files:

1. Create directory: `.claude-review-cache`
2. Clean old cache entries: Find and delete any .cache files older than 7 days
3. For each file you analyze, store the analysis result with the file's git hash as the cache key

## Phase 2: Load Comprehensive Knowledge Base

### Step 2.1: Set Up Knowledge Directories

1. Create `.claude-knowledge-cache` directory for caching downloaded knowledge
2. Check if `../comfy-claude-prompt-library` exists locally. If it does, use it for faster access.

### Step 2.2: Load Repository Guide

This is critical for understanding the architecture:

1. Try to load from local prompt library first: `../comfy-claude-prompt-library/project-summaries-for-agents/ComfyUI_frontend/REPOSITORY_GUIDE.md`
2. If not available locally, download from: `https://raw.githubusercontent.com/Comfy-Org/comfy-claude-prompt-library/master/project-summaries-for-agents/ComfyUI_frontend/REPOSITORY_GUIDE.md`
3. Cache the file for future use

### Step 2.3: Load Relevant Knowledge Folders

Intelligently load only relevant knowledge:

1. Use GitHub API to discover available knowledge folders: `https://api.github.com/repos/Comfy-Org/comfy-claude-prompt-library/contents/.claude/knowledge`
2. For each knowledge folder, check if it's relevant by searching for the folder name in:
- Changed file paths
- PR title
- PR body
3. If relevant, download all files from that knowledge folder

### Step 2.4: Load Validation Rules

Load specific validation rules:

1. Use GitHub API: `https://api.github.com/repos/Comfy-Org/comfy-claude-prompt-library/contents/.claude/commands/validation`
2. Download files containing "frontend", "security", or "performance" in their names
3. Cache all downloaded files

### Step 2.5: Load Local Guidelines

Check for and load:
1. `CLAUDE.md` in the repository root
2. `.github/CLAUDE.md`

## Phase 3: Deep Analysis Instructions

Perform comprehensive analysis on each changed file:

### 3.1 Architectural Analysis

Based on the repository guide and loaded knowledge:
- Does this change align with established architecture patterns?
- Are domain boundaries respected?
- Is the extension system used appropriately?
- Are components properly organized by feature?
- Does it follow the established service/composable/store patterns?

### 3.2 Code Quality Beyond Linting

Look for:
- Cyclomatic complexity and cognitive load
- SOLID principles adherence
- DRY violations not caught by simple duplication checks
- Proper abstraction levels
- Interface design and API clarity
- Leftover debug code (console.log, commented code, TODO comments)

### 3.3 Library Usage Enforcement

CRITICAL: Flag any re-implementation of existing functionality:
- **Tailwind CSS**: Custom CSS instead of utility classes
- **PrimeVue**: Re-implementing buttons, modals, dropdowns, etc.
- **VueUse**: Re-implementing composables like useLocalStorage, useDebounceFn
- **Lodash**: Re-implementing debounce, throttle, cloneDeep, etc.
- **Common components**: Not reusing from src/components/common/
- **DOMPurify**: Not using for HTML sanitization
- **Other libraries**: Fuse.js, Marked, Pinia, Zod, Tiptap, Xterm.js, Axios

### 3.4 Security Deep Dive

Check for:
- SQL injection vulnerabilities
- XSS vulnerabilities (v-html without sanitization)
- Hardcoded secrets or API keys
- Missing input validation
- Authentication/authorization issues
- State management security
- Cross-origin concerns
- Extension security boundaries

### 3.5 Performance Analysis

Look for:
- O(n²) or worse algorithms
- Missing memoization in expensive operations
- Unnecessary re-renders in Vue components
- Memory leak patterns (missing cleanup)
- Large bundle imports that should be lazy loaded
- N+1 query patterns
- Render performance issues
- Layout thrashing
- Network request optimization

### 3.6 Integration Concerns

Consider:
- Breaking changes to internal APIs
- Extension compatibility
- Backward compatibility
- Migration requirements

## Phase 4: Posting Inline Comments

### Step 4.1: Comment Format

For each issue found, create a concise inline comment with this structure:

```

**[category] severity Priority**

**Issue**: Brief description of the problem
**Context**: Why this matters
**Suggestion**: How to fix it

````

Categories: architecture/security/performance/quality
Severities: critical/high/medium/low

### Step 4.2: Posting Comments

For EACH issue:

1. Identify the exact file path and line number
2. Update your tracking counters (CRITICAL_COUNT, etc.)
3. Construct the comment body with proper newlines
4. Execute the gh api command as a SINGLE LINE:

```bash
gh api --method POST -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" /repos/OWNER/REPO/pulls/$PR_NUMBER/comments -f body="$COMMENT_BODY" -f commit_id="COMMIT_SHA" -f path="FILE_PATH" -F line=LINE_NUMBER -f side="RIGHT"
````

CRITICAL: The entire command must be on one line. Use actual values, not placeholders.

### Example Workflow

Here's an example of how to review a file with a security issue:

1. First, get the repository info:

   ```bash
   gh repo view --json owner,name
   # Output: {"owner":{"login":"Comfy-Org"},"name":"ComfyUI_frontend"}
   ```

2. Get the commit SHA:

   ```bash
   gh pr view $PR_NUMBER --json commits --jq '.commits[-1].oid'
   # Output: abc123def456
   ```

3. Find an issue (e.g., SQL injection on line 42 of src/db/queries.js)

4. Post the inline comment:
   ```bash
   # First, create the comment body with proper newlines
   COMMENT_BODY="**[security] critical Priority**
   ```

**Issue**: SQL injection vulnerability - user input directly concatenated into query
**Context**: Allows attackers to execute arbitrary SQL commands
**Suggestion**: Use parameterized queries or prepared statements"

# Then post the comment (as a single line)

gh api --method POST -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" /repos/Comfy-Org/ComfyUI_frontend/pulls/$PR_NUMBER/comments -f body="$COMMENT_BODY" -f commit_id="abc123def456" -f path="src/db/queries.js" -F line=42 -f side="RIGHT"

```

Repeat this process for every issue you find in the PR.

## Phase 5: Validation Rules Application

Apply ALL validation rules from the loaded knowledge files:

### Frontend Standards
- Vue 3 Composition API patterns
- Component communication patterns
- Proper use of composables
- TypeScript strict mode compliance
- Bundle optimization

### Security Audit
- Input validation
- XSS prevention
- CSRF protection
- Secure state management
- API security

### Performance Check
- Render optimization
- Memory management
- Network efficiency
- Bundle size impact

## Phase 6: Contextual Review Based on PR Type

Analyze the PR to determine its type:

1. Extract PR title and body from pr_info.json
2. Count files, additions, and deletions
3. Determine PR type:
- Feature: Check for tests, documentation, backward compatibility
- Bug fix: Verify root cause addressed, includes regression tests
- Refactor: Ensure behavior preservation, tests still pass

## Phase 7: Generate Comprehensive Summary

After ALL inline comments are posted, create a summary:

1. Calculate total issues by category and severity
2. Use `gh pr review $PR_NUMBER --comment` to post a summary with:
- Review disclaimer
- Issue distribution (counts by severity)
- Category breakdown
- Key findings for each category
- Positive observations
- References to guidelines
- Next steps

Include in the summary:
```

# Comprehensive PR Review

This review is generated by Claude. It may not always be accurate, as with human reviewers. If you believe that any of the comments are invalid or incorrect, please state why for each. For others, please implement the changes in one way or another.

## Review Summary

**PR**: [PR TITLE] (#$PR_NUMBER)
**Impact**: [X] additions, [Y] deletions across [Z] files

### Issue Distribution

- Critical: [CRITICAL_COUNT]
- High: [HIGH_COUNT]
- Medium: [MEDIUM_COUNT]
- Low: [LOW_COUNT]

### Category Breakdown

- Architecture: [ARCHITECTURE_ISSUES] issues
- Security: [SECURITY_ISSUES] issues
- Performance: [PERFORMANCE_ISSUES] issues
- Code Quality: [QUALITY_ISSUES] issues

## Key Findings

### Architecture & Design

[Detailed architectural analysis based on repository patterns]

### Security Considerations

[Security implications beyond basic vulnerabilities]

### Performance Impact

[Performance analysis including bundle size, render impact]

### Integration Points

[How this affects other systems, extensions, etc.]

## Positive Observations

[What was done well, good patterns followed]

## References

- [Repository Architecture Guide](https://github.com/Comfy-Org/comfy-claude-prompt-library/blob/master/project-summaries-for-agents/ComfyUI_frontend/REPOSITORY_GUIDE.md)
- [Frontend Standards](https://github.com/Comfy-Org/comfy-claude-prompt-library/blob/master/.claude/commands/validation/frontend-code-standards.md)
- [Security Guidelines](https://github.com/Comfy-Org/comfy-claude-prompt-library/blob/master/.claude/commands/validation/security-audit.md)

## Next Steps

1. Address critical issues before merge
2. Consider architectural feedback for long-term maintainability
3. Add tests for uncovered scenarios
4. Update documentation if needed

---

_This is a comprehensive automated review. For architectural decisions requiring human judgment, please request additional manual review._

```

## Important Guidelines

1. **Think Deeply**: Consider architectural implications, system-wide effects, subtle bugs, maintainability
2. **Be Specific**: Point to exact lines with concrete suggestions
3. **Be Constructive**: Focus on improvements, not just problems
4. **Be Concise**: Keep comments brief and actionable
5. **No Formatting**: Don't use markdown headers in inline comments
6. **No Emojis**: Keep comments professional

This is a COMPREHENSIVE review, not a linting pass. Provide the same quality feedback a senior engineer would give after careful consideration.

## Execution Order

1. Phase 1: Setup and checkout PR
2. Phase 2: Load all relevant knowledge
3. Phase 3-5: Analyze each changed file thoroughly
4. Phase 4: Post inline comments as you find issues
5. Phase 6: Consider PR type for additional checks
6. Phase 7: Post comprehensive summary ONLY after all inline comments

Remember: Individual inline comments for each issue, then one final summary. Never batch issues into a single comment.
```
