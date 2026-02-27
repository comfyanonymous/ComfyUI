#!/usr/bin/env tsx
import { execSync } from 'child_process'
import fs from 'fs'
import path from 'path'

interface ReleaseInfo {
  current_version: string
  target_minor: number
  target_version: string
  target_branch: string
  needs_release: boolean
  latest_patch_tag: string | null
  branch_head_sha: string | null
  tag_commit_sha: string | null
  diff_url: string
  release_pr_url: string | null
}

/**
 * Execute a command and return stdout
 */
function exec(command: string, cwd?: string): string {
  try {
    return execSync(command, {
      cwd,
      encoding: 'utf-8',
      stdio: ['pipe', 'pipe', 'pipe']
    }).trim()
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error)
    const cwdInfo = cwd ? ` in directory: ${cwd}` : ''
    console.error(
      `Command failed: ${command}${cwdInfo}\nError: ${errorMessage}`
    )
    return ''
  }
}

/**
 * Parse version from requirements.txt
 * Handles formats: comfyui-frontend-package==1.2.3, comfyui-frontend-package>=1.2.3, etc.
 */
function parseRequirementsVersion(requirementsPath: string): string | null {
  if (!fs.existsSync(requirementsPath)) {
    console.error(`Requirements file not found: ${requirementsPath}`)
    return null
  }

  const content = fs.readFileSync(requirementsPath, 'utf-8')
  const match = content.match(
    /comfyui-frontend-package\s*(?:==|>=|<=|~=|>|<)\s*([0-9]+\.[0-9]+\.[0-9]+)/
  )

  if (!match) {
    console.error(
      'Could not find comfyui-frontend-package version in requirements.txt'
    )
    return null
  }

  return match[1]
}

/**
 * Validate semantic version string
 */
function isValidSemver(version: string): boolean {
  if (!version || typeof version !== 'string') {
    return false
  }

  const parts = version.split('.')
  if (parts.length !== 3) {
    return false
  }

  return parts.every((part) => {
    const num = Number(part)
    return Number.isFinite(num) && num >= 0 && String(num) === part
  })
}

/**
 * Get the latest patch tag for a given minor version
 */
function getLatestPatchTag(repoPath: string, minor: number): string | null {
  // Fetch all tags
  exec('git fetch --tags', repoPath)

  // Use git's native version sorting to get the latest tag
  const latestTag = exec(
    `git tag -l 'v1.${minor}.*' --sort=-version:refname | head -n 1`,
    repoPath
  )

  if (!latestTag) {
    return null
  }

  // Validate the tag is a valid semver (vX.Y.Z format)
  const validTagRegex = /^v\d+\.\d+\.\d+$/
  if (!validTagRegex.test(latestTag)) {
    console.error(
      `Latest tag for minor version ${minor} is not valid semver: ${latestTag}`
    )
    return null
  }

  return latestTag
}

/**
 * Resolve the ComfyUI release information
 */
function resolveRelease(
  comfyuiRepoPath: string,
  frontendRepoPath: string
): ReleaseInfo | null {
  // Parse current version from ComfyUI requirements.txt
  const requirementsPath = path.join(comfyuiRepoPath, 'requirements.txt')
  const currentVersion = parseRequirementsVersion(requirementsPath)

  if (!currentVersion) {
    return null
  }

  // Validate version format
  if (!isValidSemver(currentVersion)) {
    console.error(
      `Invalid semantic version format: ${currentVersion}. Expected format: X.Y.Z`
    )
    return null
  }

  const [major, currentMinor, patch] = currentVersion.split('.').map(Number)

  // Fetch all branches
  exec('git fetch origin', frontendRepoPath)

  // Try next minor first, fall back to current minor if not available
  let targetMinor = currentMinor + 1
  let targetBranch = `core/1.${targetMinor}`

  const nextMinorExists = exec(
    `git rev-parse --verify origin/${targetBranch}`,
    frontendRepoPath
  )

  if (!nextMinorExists) {
    // Fall back to current minor for patch releases
    targetMinor = currentMinor
    targetBranch = `core/1.${targetMinor}`

    const currentMinorExists = exec(
      `git rev-parse --verify origin/${targetBranch}`,
      frontendRepoPath
    )

    if (!currentMinorExists) {
      console.error(
        `Neither core/1.${currentMinor + 1} nor core/1.${currentMinor} branches exist in frontend repo`
      )
      return null
    }

    console.error(
      `Next minor branch core/1.${currentMinor + 1} not found, falling back to core/1.${currentMinor} for patch release`
    )
  }

  // Get latest patch tag for target minor
  const latestPatchTag = getLatestPatchTag(frontendRepoPath, targetMinor)

  let needsRelease = false
  let branchHeadSha: string | null = null
  let tagCommitSha: string | null = null
  let targetVersion = currentVersion

  if (latestPatchTag) {
    // Get commit SHA for the tag
    tagCommitSha = exec(`git rev-list -n 1 ${latestPatchTag}`, frontendRepoPath)

    // Get commit SHA for branch head
    branchHeadSha = exec(
      `git rev-parse origin/${targetBranch}`,
      frontendRepoPath
    )

    // Check if there are commits between tag and branch head
    const commitsBetween = exec(
      `git rev-list ${latestPatchTag}..origin/${targetBranch} --count`,
      frontendRepoPath
    )

    const commitCount = parseInt(commitsBetween, 10)
    needsRelease = !isNaN(commitCount) && commitCount > 0

    // Parse existing patch number and increment if needed
    const tagVersion = latestPatchTag.replace('v', '')

    // Validate tag version format
    if (!isValidSemver(tagVersion)) {
      console.error(
        `Invalid tag version format: ${tagVersion}. Expected format: X.Y.Z`
      )
      return null
    }

    const [, , existingPatch] = tagVersion.split('.').map(Number)

    // Validate existingPatch is a valid number
    if (!Number.isFinite(existingPatch) || existingPatch < 0) {
      console.error(`Invalid patch number in tag: ${existingPatch}`)
      return null
    }

    if (needsRelease) {
      targetVersion = `1.${targetMinor}.${existingPatch + 1}`
    } else {
      targetVersion = tagVersion
    }
  } else {
    // No tags exist for this minor version, need to create v1.{targetMinor}.0
    needsRelease = true
    targetVersion = `1.${targetMinor}.0`
    branchHeadSha = exec(
      `git rev-parse origin/${targetBranch}`,
      frontendRepoPath
    )
  }

  const diffUrl = `https://github.com/Comfy-Org/ComfyUI_frontend/compare/v${currentVersion}...v${targetVersion}`

  return {
    current_version: currentVersion,
    target_minor: targetMinor,
    target_version: targetVersion,
    target_branch: targetBranch,
    needs_release: needsRelease,
    latest_patch_tag: latestPatchTag,
    branch_head_sha: branchHeadSha,
    tag_commit_sha: tagCommitSha,
    diff_url: diffUrl,
    release_pr_url: null // Will be populated by workflow if release is triggered
  }
}

// Main execution
const comfyuiRepoPath = process.argv[2]
const frontendRepoPath = process.argv[3] || process.cwd()

if (!comfyuiRepoPath) {
  console.error(
    'Usage: resolve-comfyui-release.ts <comfyui-repo-path> [frontend-repo-path]'
  )
  process.exit(1)
}

const releaseInfo = resolveRelease(comfyuiRepoPath, frontendRepoPath)

if (!releaseInfo) {
  console.error('Failed to resolve release information')
  process.exit(1)
}

// Output as JSON for GitHub Actions

console.log(JSON.stringify(releaseInfo, null, 2))

export { resolveRelease }
