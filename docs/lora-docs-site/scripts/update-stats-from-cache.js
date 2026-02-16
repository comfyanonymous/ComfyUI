import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CACHE_FILE = path.join(__dirname, '..', 'civitai-image-cache.json');
const CONTENT_DIR = path.join(__dirname, '..', 'src', 'content', 'docs');

// Read the cache
const cache = JSON.parse(fs.readFileSync(CACHE_FILE, 'utf-8'));

// Calculate score based on downloads
function calculateScore(downloads) {
  if (downloads >= 10000) return '⭐⭐⭐';
  if (downloads >= 2000) return '⭐⭐';
  if (downloads >= 500) return '⭐';
  return '-';
}

// Extract model ID from Civitai URL
function getModelIdFromUrl(url) {
  const match = url.match(/civitai\.com\/models\/(\d+)/);
  return match ? match[1] : null;
}

// Update a single MDX file
function updateMdxFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');

  // Extract civitaiUrl from frontmatter
  const civitaiMatch = content.match(/civitaiUrl:\s*"([^"]+)"/);
  if (!civitaiMatch) {
    console.log(`⚠️  No civitaiUrl in ${filePath}`);
    return;
  }

  const civitaiUrl = civitaiMatch[1];
  const modelId = getModelIdFromUrl(civitaiUrl);

  if (!modelId) {
    console.log(`⚠️  Cannot extract model ID from ${civitaiUrl}`);
    return;
  }

  // Look up stats in cache
  const cachedData = cache[modelId];
  if (!cachedData) {
    console.log(`⚠️  No cache entry for model ${modelId} (${path.basename(filePath)})`);
    return;
  }

  const downloads = cachedData.downloads || 0;
  const rating = cachedData.rating || 0;
  const tips = cachedData.tips || 0;
  const score = calculateScore(downloads);

  // Update frontmatter
  let newContent = content;

  // Update frontmatter fields
  newContent = newContent.replace(/downloads:\s*\d+/, `downloads: ${downloads}`);
  newContent = newContent.replace(/rating:\s*\d+/, `rating: ${rating}`);
  newContent = newContent.replace(/tips:\s*\d+/, `tips: ${tips}`);
  newContent = newContent.replace(/score:\s*"[^"]*"/, `score: "${score}"`);

  // Update LoraCard props
  newContent = newContent.replace(/downloads=\{[^}]+\}/, `downloads={${downloads}}`);
  newContent = newContent.replace(/rating=\{[^}]+\}/, `rating={${rating}}`);
  newContent = newContent.replace(/tips=\{[^}]+\}/, `tips={${tips}}`);
  newContent = newContent.replace(/score="[^"]*"/, `score="${score}"`);

  // Update sidebar badge
  newContent = newContent.replace(/(sidebar:\s*\n\s*badge:\s*\n\s*text:\s*)"[^"]*"/, `$1"${score}"`);

  fs.writeFileSync(filePath, newContent, 'utf-8');
  console.log(`✅ Updated ${path.basename(filePath)}: ${downloads.toLocaleString()} downloads, ${rating} 👍, ${tips} tips (${score})`);
}

// Walk through all directories and update MDX files
function walkDir(dir) {
  const files = fs.readdirSync(dir);

  for (const file of files) {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      walkDir(filePath);
    } else if (file.endsWith('.mdx') && file !== 'index.mdx') {
      updateMdxFile(filePath);
    }
  }
}

console.log('🔄 Updating statistics from cache...\n');
walkDir(CONTENT_DIR);
console.log('\n✨ Done!');
