import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const IMAGES_DIR = path.join(__dirname, '..', 'public', 'images', 'loras');
const CONTENT_DIR = path.join(__dirname, '..', 'src', 'content', 'docs');
const CACHE_FILE = path.join(__dirname, '..', 'civitai-image-cache.json');

const cache = JSON.parse(fs.readFileSync(CACHE_FILE, 'utf-8'));

// Files to fix (MP4s with .jpg extension)
const filesToFix = [
  'beautiful_girls_faces_prettynova',
  'bobbi_spektre',
  'corset_lingerie',
  'flux_pro_11_extreme_detailer',
  'flux_pro_ultra_detailer',
  'god_tier_tits',
  'isabella_flux_cfh',
  'livi_ssp_se_doll',
  'midjourney_style',
  'mystic_xxx',
  'normal_european_woman',
  'ona_instagram_model',
  'realistic_people_photograph_flux',
  'rich_girl_instagram_influencer',
  'russian_model_katerina',
  'secret_sauce_hero',
  'strange_brue',
  'tessa_sheen'
];

// Find MDX file and extract model ID
function findModelId(fileName) {
  try {
    const result = execSync(
      `find "${CONTENT_DIR}" -name "${fileName}.mdx" -type f`,
      { encoding: 'utf-8' }
    ).trim();

    if (!result) {
      console.log(`⚠️  No MDX file found for ${fileName}`);
      return null;
    }

    const content = fs.readFileSync(result, 'utf-8');
    const match = content.match(/civitaiUrl:\s*"https:\/\/civitai\.com\/models\/(\d+)/);

    if (match) {
      return match[1];
    } else {
      console.log(`⚠️  No civitaiUrl found in ${fileName}.mdx`);
      return null;
    }
  } catch (e) {
    console.log(`⚠️  Error finding ${fileName}: ${e.message}`);
    return null;
  }
}

console.log('🔧 Fixing MP4 files with .jpg extension...\n');

let fixed = 0;
let failed = 0;

for (const fileName of filesToFix) {
  const modelId = findModelId(fileName);

  if (!modelId) {
    failed++;
    continue;
  }

  const cachedData = cache[modelId];
  if (!cachedData || !cachedData.url) {
    console.log(`❌ ${fileName}: No image in cache for model ${modelId}`);
    failed++;
    continue;
  }

  const imageUrl = cachedData.url;
  const outputPath = path.join(IMAGES_DIR, `${fileName}.jpg`);

  try {
    console.log(`📥 Downloading ${fileName} from ${imageUrl.substring(0, 60)}...`);
    execSync(`curl -L "${imageUrl}" -o "${outputPath}"`, { stdio: 'ignore' });

    // Verify it's an image
    const fileType = execSync(`file "${outputPath}"`, { encoding: 'utf-8' });
    if (fileType.includes('image') || fileType.includes('PNG') || fileType.includes('JPEG')) {
      console.log(`✅ ${fileName}: Fixed successfully`);
      fixed++;
    } else {
      console.log(`⚠️  ${fileName}: Downloaded but not an image file`);
      failed++;
    }
  } catch (e) {
    console.log(`❌ ${fileName}: Download failed - ${e.message}`);
    failed++;
  }
}

console.log(`\n📊 Summary:`);
console.log(`✅ Fixed: ${fixed}`);
console.log(`❌ Failed: ${failed}`);
