#!/usr/bin/env node

/**
 * Fetch preview images from Civitai API for all LoRAs
 *
 * Usage: node scripts/fetch-civitai-images.js
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DOCS_DIR = path.join(__dirname, '..', '..');
const OUTPUT_DIR = path.join(__dirname, '..', 'public', 'images', 'loras');
const CACHE_FILE = path.join(__dirname, '..', 'civitai-image-cache.json');

// Ensure output directory exists
await fs.mkdir(OUTPUT_DIR, { recursive: true });

// Load cache
let imageCache = {};
try {
	const cacheContent = await fs.readFile(CACHE_FILE, 'utf-8');
	imageCache = JSON.parse(cacheContent);
	console.log('📦 Loaded image cache with', Object.keys(imageCache).length, 'entries');
} catch (err) {
	console.log('📦 No cache found, starting fresh');
}

/**
 * Extract model ID from Civitai URL
 */
function extractModelId(url) {
	const match = url.match(/models\/(\d+)/);
	return match ? match[1] : null;
}

/**
 * Fetch model data from Civitai API
 */
async function fetchModelData(modelId) {
	if (imageCache[modelId]) {
		console.log(`  ✓ Using cached data for model ${modelId}`);
		return imageCache[modelId];
	}

	try {
		const response = await fetch(`https://civitai.com/api/v1/models/${modelId}`);
		if (!response.ok) {
			console.error(`  ✗ Failed to fetch model ${modelId}: ${response.status}`);
			return null;
		}

		const data = await response.json();

		// Get the first model version's first image (skip videos)
		if (data.modelVersions && data.modelVersions.length > 0) {
			const version = data.modelVersions[0];
			if (version.images && version.images.length > 0) {
				// Filter out videos - only get actual images
				const actualImages = version.images.filter(img => img.type === 'image' || !img.type);

				if (actualImages.length > 0) {
					const image = actualImages[0];
					const imageData = {
						url: image.url,
						width: image.width,
						height: image.height,
						nsfw: image.nsfw || false,
						// Add stats
						downloads: data.stats?.downloadCount || 0,
						rating: data.stats?.thumbsUpCount || 0,
						tips: data.stats?.tippedAmountCount || 0,
					};

					// Cache it
					imageCache[modelId] = imageData;
					console.log(`  ✓ Fetched image and stats for model ${modelId} (${imageData.downloads} downloads)`);

					return imageData;
				}
			}
		}

		console.log(`  ⚠ No images found for model ${modelId}`);
		return null;
	} catch (error) {
		console.error(`  ✗ Error fetching model ${modelId}:`, error.message);
		return null;
	}
}

/**
 * Download image from URL
 */
async function downloadImage(url, outputPath) {
	try {
		const response = await fetch(url);
		if (!response.ok) {
			throw new Error(`HTTP ${response.status}`);
		}

		const buffer = await response.arrayBuffer();
		await fs.writeFile(outputPath, Buffer.from(buffer));
		console.log(`    ✓ Downloaded image to ${path.basename(outputPath)}`);
		return true;
	} catch (error) {
		console.error(`    ✗ Failed to download image:`, error.message);
		return false;
	}
}

/**
 * Parse markdown file to extract Civitai URL
 */
async function parseMdFile(filePath) {
	try {
		const content = await fs.readFile(filePath, 'utf-8');

		// Extract Civitai URL
		const civitaiMatch = content.match(/\*\*Civitai\*\*.*?(https:\/\/civitai\.com\/models\/\d+[^\s\)]*)/);
		if (!civitaiMatch) {
			return null;
		}

		const civitaiUrl = civitaiMatch[1];
		const modelId = extractModelId(civitaiUrl);

		if (!modelId) {
			return null;
		}

		return {
			modelId,
			civitaiUrl,
			fileName: path.basename(filePath, '.md'),
		};
	} catch (error) {
		console.error(`Error parsing ${filePath}:`, error.message);
		return null;
	}
}

/**
 * Process all markdown files in a directory
 */
async function processDirectory(dirPath, categoryName) {
	console.log(`\n📂 Processing ${categoryName}...`);

	try {
		const files = await fs.readdir(dirPath);
		const mdFiles = files.filter(f => f.endsWith('.md') && f !== 'INDEX.md');

		console.log(`   Found ${mdFiles.length} LoRA files`);

		let processed = 0;
		let failed = 0;

		for (const file of mdFiles) {
			const filePath = path.join(dirPath, file);
			const data = await parseMdFile(filePath);

			if (!data) {
				continue;
			}

			console.log(`\n  📄 Processing: ${file}`);

			// Fetch image data
			const imageData = await fetchModelData(data.modelId);

			if (imageData && imageData.url) {
				// Download image
				const outputPath = path.join(OUTPUT_DIR, `${data.fileName}.jpg`);

				// Check if image already exists
				try {
					await fs.access(outputPath);
					console.log(`    ✓ Image already exists, skipping download`);
					processed++;
					continue;
				} catch {
					// Image doesn't exist, download it
				}

				const success = await downloadImage(imageData.url, outputPath);
				if (success) {
					processed++;
				} else {
					failed++;
				}

				// Rate limiting - wait 500ms between requests
				await new Promise(resolve => setTimeout(resolve, 500));
			} else {
				failed++;
			}
		}

		console.log(`\n  ✅ ${categoryName}: ${processed} images processed, ${failed} failed`);
	} catch (error) {
		console.error(`Error processing directory ${dirPath}:`, error.message);
	}
}

/**
 * Main function
 */
async function main() {
	console.log('🚀 Starting Civitai image fetcher...\n');

	const categories = [
		{ dir: 'CHARACTERS_OTHER', name: 'Characters - Other' },
		{ dir: 'CHARACTERS_WATW', name: 'Characters - WATW' },
		{ dir: 'CHARACTERS_ACORN', name: 'Characters - Acorn' },
		{ dir: 'ANATOMY_BREASTS', name: 'Anatomy - Breasts' },
		{ dir: 'ANATOMY_PUSSY', name: 'Anatomy - Pussy' },
		{ dir: 'ANATOMY_ASS', name: 'Anatomy - Ass' },
		{ dir: 'POSES', name: 'Poses' },
		{ dir: 'CLOTHING', name: 'Clothing' },
		{ dir: 'STYLE_ENHANCEMENT', name: 'Style Enhancement' },
		{ dir: 'BODY_TYPES', name: 'Body Types' },
	];

	for (const category of categories) {
		const dirPath = path.join(DOCS_DIR, category.dir);

		try {
			await fs.access(dirPath);
			await processDirectory(dirPath, category.name);
		} catch {
			console.log(`⚠️  Directory not found: ${category.dir}`);
		}
	}

	// Save cache
	await fs.writeFile(CACHE_FILE, JSON.stringify(imageCache, null, 2));
	console.log(`\n💾 Saved image cache with ${Object.keys(imageCache).length} entries`);

	console.log('\n✨ Done!');
}

// Run main function
main().catch(console.error);
