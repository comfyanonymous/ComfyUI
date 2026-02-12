import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CONTENT_DIR = path.join(__dirname, '..', 'src', 'content', 'docs');
const OUTPUT_FILE = path.join(__dirname, '..', 'src', 'lora-stats.json');

const stats = {};

// Parse frontmatter from MDX file
function parseFrontmatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---/);
  if (!match) return null;

  const frontmatter = match[1];
  const data = {};

  // Extract fields
  const extractField = (name, isNumber = false) => {
    const regex = new RegExp(`^${name}:\\s*(.+)$`, 'm');
    const match = frontmatter.match(regex);
    if (match) {
      const value = match[1].trim().replace(/^["']|["']$/g, '');
      return isNumber ? parseInt(value) || 0 : value;
    }
    return isNumber ? 0 : '';
  };

  data.downloads = extractField('downloads', true);
  data.rating = extractField('rating', true);
  data.tips = extractField('tips', true);
  data.score = extractField('score');
  data.type = extractField('type');
  data.trigger = extractField('trigger');
  data.strength = extractField('strength');
  data.civitaiUrl = extractField('civitaiUrl');
  data.compatibility = extractField('compatibility');
  data.title = extractField('title');

  return data;
}

// Walk through all directories and process MDX files
function walkDir(dir, basePath = '') {
  const files = fs.readdirSync(dir);

  for (const file of files) {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      walkDir(filePath, path.join(basePath, file));
    } else if (file.endsWith('.mdx') && file !== 'index.mdx') {
      const content = fs.readFileSync(filePath, 'utf-8');
      const data = parseFrontmatter(content);

      if (data && data.title) {
        // Create slug from path
        const slug = path.join(basePath, file.replace('.mdx', '')).replace(/\\/g, '/');
        stats[slug] = data;
      }
    }
  }
}

console.log('📊 Exporting stats from MDX frontmatter...\n');
walkDir(CONTENT_DIR);

fs.writeFileSync(OUTPUT_FILE, JSON.stringify(stats, null, 2), 'utf-8');
console.log(`✅ Exported ${Object.keys(stats).length} entries to ${OUTPUT_FILE}`);
