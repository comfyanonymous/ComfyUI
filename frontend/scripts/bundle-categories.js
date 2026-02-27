// @ts-check
/**
 * Bundle categorization configuration
 *
 * This file defines how bundles are categorized in size reports.
 * Categories help identify which parts of the application are growing.
 */

/**
 * @typedef {Object} BundleCategory
 * @property {string} name - Display name of the category
 * @property {string} description - Description of what this category includes
 * @property {RegExp[]} patterns - Regex patterns to match bundle files
 * @property {number} order - Sort order for display (lower = first)
 */

/** @type {BundleCategory[]} */
export const BUNDLE_CATEGORIES = [
  {
    name: 'App Entry Points',
    description: 'Main entry bundles and manifests',
    patterns: [/^index-.*\.js$/i, /^manifest-.*\.js$/i],
    order: 1
  },
  {
    name: 'Graph Workspace',
    description: 'Graph editor runtime, canvas, workflow orchestration',
    patterns: [
      /Graph(View|State)?-.*\.js$/i,
      /(Canvas|Workflow|History|NodeGraph|Compositor)-.*\.js$/i
    ],
    order: 2
  },
  {
    name: 'Views & Navigation',
    description: 'Top-level views, pages, and routed surfaces',
    patterns: [/.*(View|Page|Layout|Screen|Route)-.*\.js$/i],
    order: 3
  },
  {
    name: 'Panels & Settings',
    description: 'Configuration panels, inspectors, and settings screens',
    patterns: [/.*(Panel|Settings|Config|Preferences|Manager)-.*\.js$/i],
    order: 4
  },
  {
    name: 'User & Accounts',
    description: 'Authentication, profile, and account management bundles',
    patterns: [
      /.*((User(Panel|Select|Auth|Account|Profile|Settings|Preferences|Manager|List|Menu|Modal))|Account|Auth|Profile|Login|Signup|Password).*-.+\.js$/i
    ],
    order: 5
  },
  {
    name: 'Editors & Dialogs',
    description: 'Modals, dialogs, drawers, and in-app editors',
    patterns: [/.*(Modal|Dialog|Drawer|Editor)-.*\.js$/i],
    order: 6
  },
  {
    name: 'UI Components',
    description: 'Reusable component library chunks',
    patterns: [
      /.*(Button|Avatar|Badge|Dropdown|Tabs|Table|List|Card|Form|Input|Toggle|Menu|Toolbar|Sidebar)-.*\.js$/i,
      /.*\.vue_vue_type_script_setup_true_lang-.*\.js$/i
    ],
    order: 7
  },
  {
    name: 'Data & Services',
    description: 'Stores, services, APIs, and repositories',
    patterns: [/.*(Service|Store|Api|Repository)-.*\.js$/i],
    order: 8
  },
  {
    name: 'Utilities & Hooks',
    description: 'Helpers, composables, and utility bundles',
    patterns: [
      /.*(Util|Utils|Helper|Composable|Hook)-.*\.js$/i,
      /use[A-Z].*\.js$/
    ],
    order: 9
  },
  {
    name: 'Vendor & Third-Party',
    description: 'External libraries and shared vendor chunks',
    patterns: [
      /^(chunk|vendor|prime|three|lodash|chart|firebase|yjs|axios|uuid)-.*\.js$/i
    ],
    order: 10
  },
  {
    name: 'Other',
    description: 'Bundles that do not match a named category',
    patterns: [/.*/],
    order: 99
  }
]

/**
 * Categorize a bundle file based on its name
 *
 * @param {string} fileName - The bundle file name (e.g., "assets/GraphView-BnV6iF9h.js")
 * @returns {string} - The category name
 */
export function categorizeBundle(fileName) {
  // Extract just the file name without path
  const baseName = fileName.split('/').pop() || fileName

  // Find the first matching category
  for (const category of BUNDLE_CATEGORIES) {
    for (const pattern of category.patterns) {
      if (pattern.test(baseName)) {
        return category.name
      }
    }
  }

  return 'Other'
}

/**
 * Get category metadata by name
 *
 * @param {string} categoryName - The category name
 * @returns {BundleCategory | undefined} - The category metadata
 */
export function getCategoryMetadata(categoryName) {
  return BUNDLE_CATEGORIES.find((cat) => cat.name === categoryName)
}
