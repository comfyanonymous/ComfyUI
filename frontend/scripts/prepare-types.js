import fs from 'fs'
import path from 'path'

const mainPackage = JSON.parse(fs.readFileSync('./package.json', 'utf8'))

// Create the types-only package.json
const typesPackage = {
  name: `${mainPackage.name}-types`,
  version: mainPackage.version,
  types: './index.d.ts',
  files: ['index.d.ts'],
  publishConfig: {
    access: 'public'
  },
  repository: mainPackage.repository,
  homepage: mainPackage.homepage,
  description: `TypeScript definitions for ${mainPackage.name}`,
  license: mainPackage.license,
  dependencies: {},
  peerDependencies: {
    vue: mainPackage.dependencies.vue,
    zod: mainPackage.dependencies.zod
  }
}

// Ensure dist directory exists
const distDir = './dist'
if (!fs.existsSync(distDir)) {
  fs.mkdirSync(distDir, { recursive: true })
}

// Write the new package.json to the dist directory
fs.writeFileSync(
  path.join(distDir, 'package.json'),
  JSON.stringify(typesPackage, null, 2)
)

console.log('Types package.json have been prepared in the dist directory')
