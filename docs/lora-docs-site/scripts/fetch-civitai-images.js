#!/usr/bin/env node

/**
 * Fetch preview images and stats from Civitai API for all LoRAs
 * Reads .mdx files from Starlight content directory (src/content/docs/)
 *
 * Usage: node scripts/fetch-civitai-images.js
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CONTENT_DIR = path.join(__dirname, '..', 'src', 'content', 'docs');
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
 * Parse .mdx file to extract civitaiUrl from YAML frontmatter
 */
async function parseMdxFile(filePath) {
	try {
		const content = await fs.readFile(filePath, 'utf-8');

		// Extract civitaiUrl from frontmatter
		const civitaiMatch = content.match(/civitaiUrl:\s*"([^"]+)"/);
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
			fileName: path.basename(filePath, '.mdx'),
		};
	} catch (error) {
		console.error(`Error parsing ${filePath}:`, error.message);
		return null;
	}
}

/**
 * Recursively walk directory and collect all .mdx files (excluding index.mdx)
 */
async function collectMdxFiles(dir) {
	const results = [];
	const entries = await fs.readdir(dir, { withFileTypes: true });

	for (const entry of entries) {
		const fullPath = path.join(dir, entry.name);
		if (entry.isDirectory()) {
			const subResults = await collectMdxFiles(fullPath);
			results.push(...subResults);
		} else if (entry.name.endsWith('.mdx') && entry.name !== 'index.mdx') {
			results.push(fullPath);
		}
	}

	return results;
}

/**
 * Main function
 */
async function main() {
	console.log('🚀 Starting Civitai image fetcher...\n');
	console.log(`📂 Scanning: ${CONTENT_DIR}\n`);

	const mdxFiles = await collectMdxFiles(CONTENT_DIR);
	console.log(`Found ${mdxFiles.length} .mdx files to process\n`);

	let processed = 0;
	let skipped = 0;
	let failed = 0;

	for (const filePath of mdxFiles) {
		const relativePath = path.relative(CONTENT_DIR, filePath);
		const data = await parseMdxFile(filePath);

		if (!data) {
			skipped++;
			continue;
		}

		console.log(`📄 Processing: ${relativePath}`);

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

			// Rate limiting - wait 500ms between API requests
			await new Promise(resolve => setTimeout(resolve, 500));
		} else {
			failed++;
		}
	}

	// Save cache
	await fs.writeFile(CACHE_FILE, JSON.stringify(imageCache, null, 2));
	console.log(`\n💾 Saved image cache with ${Object.keys(imageCache).length} entries`);

	console.log(`\n✅ Summary: ${processed} processed, ${skipped} skipped (no civitaiUrl), ${failed} failed`);
	console.log('\n✨ Done!');
}

// Run main function
main().catch(console.error);
