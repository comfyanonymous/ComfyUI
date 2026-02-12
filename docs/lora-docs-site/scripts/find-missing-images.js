import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CONTENT_DIR = path.join(__dirname, '..', 'src', 'content', 'docs');
const IMAGES_DIR = path.join(__dirname, '..', 'public', 'images', 'loras');

const missingImages = [];
const existingImages = new Set(fs.readdirSync(IMAGES_DIR));

// Walk through all directories and check for missing images
function walkDir(dir, basePath = '') {
  const files = fs.readdirSync(dir);

  for (const file of files) {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory()) {
      walkDir(filePath, path.join(basePath, file));
    } else if (file.endsWith('.mdx') && file !== 'index.mdx') {
      const fileName = file.replace('.mdx', '');
      const imageName = `${fileName}.jpg`;

      if (!existingImages.has(imageName)) {
        const slug = path.join(basePath, fileName).replace(/\\/g, '/');
        missingImages.push({ slug, imageName, mdxPath: filePath });
      }
    }
  }
}

console.log('🔍 Checking for missing images...\n');
walkDir(CONTENT_DIR);

console.log(`📊 Found ${missingImages.length} missing images:\n`);
missingImages.forEach(item => {
  console.log(`  ❌ ${item.imageName} (${item.slug})`);
});

console.log(`\n✅ Total existing images: ${existingImages.size}`);
console.log(`❌ Total missing images: ${missingImages.length}`);
