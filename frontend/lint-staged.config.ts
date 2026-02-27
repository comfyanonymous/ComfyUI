import path from 'node:path'

export default {
  'tests-ui/**': () =>
    'echo "Files in tests-ui/ are deprecated. Colocate tests with source files." && exit 1',

  './**/*.{css,vue}': (stagedFiles: string[]) => {
    const joinedPaths = toJoinedRelativePaths(stagedFiles)
    return [`pnpm exec stylelint --allow-empty-input ${joinedPaths}`]
  },

  './**/*.js': (stagedFiles: string[]) => formatAndEslint(stagedFiles),

  './**/*.{ts,tsx,vue,mts}': (stagedFiles: string[]) => {
    const commands = [...formatAndEslint(stagedFiles), 'pnpm typecheck']

    const hasBrowserTestsChanges = stagedFiles
      .map((f) => path.relative(process.cwd(), f).replace(/\\/g, '/'))
      .some((f) => f.startsWith('browser_tests/'))

    if (hasBrowserTestsChanges) {
      commands.push('pnpm typecheck:browser')
    }

    return commands
  }
}

function formatAndEslint(fileNames: string[]) {
  const joinedPaths = toJoinedRelativePaths(fileNames)
  return [
    `pnpm exec oxfmt --write ${joinedPaths}`,
    `pnpm exec oxlint --fix ${joinedPaths}`,
    `pnpm exec eslint --cache --fix --no-warn-ignored ${joinedPaths}`
  ]
}

function toJoinedRelativePaths(fileNames: string[]) {
  const relativePaths = fileNames.map((f) =>
    path.relative(process.cwd(), f).replace(/\\/g, '/')
  )
  return relativePaths.map((p) => `"${p}"`).join(' ')
}
