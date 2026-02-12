#!/usr/bin/env node

/**
 * Generate Starlight content from existing markdown documentation
 *
 * Usage: node scripts/generate-content.js
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DOCS_DIR = path.join(__dirname, '..', '..');
const CONTENT_DIR = path.join(__dirname, '..', 'src', 'content', 'docs');

/**
 * Parse markdown frontmatter and content
 */
function parseMarkdown(content) {
	// Remove BOM (Byte Order Mark) if present
	content = content.replace(/^\uFEFF/, '');

	// Extract title from first heading
	const titleMatch = content.match(/^#\s+(.+)$/m);
	const title = titleMatch ? titleMatch[1] : 'Untitled';

	// Extract Civitai URL
	const civitaiMatch = content.match(/\*\*Civitai\*\*.*?(https:\/\/civitai\.com\/models\/\d+[^\s\)]*)/);
	const civitaiUrl = civitaiMatch ? civitaiMatch[1] : null;

	// Extract model ID
	const modelId = civitaiUrl ? civitaiUrl.match(/models\/(\d+)/)?.[1] : null;

	// Extract stats
	const downloadsMatch = content.match(/\*\*Downloads\*\*[^\d]*(\d[\d,]+)/);
	const ratingMatch = content.match(/\*\*👍\*\*[^\d]*(\d+)/);
	const tipsMatch = content.match(/\*\*Tips\*\*[^\d$]*\$?(\d+)/);
	const scoreMatch = content.match(/\*\*Score\*\*[^\⭐]*(⭐+|-)/);

	// Extract trigger word (supports both formats: - **Trigger:** and | **Trigger** |)
	const triggerMatch = content.match(/\|\s*\*\*Trigger\*\*\s*\|\s*`([^`]+)`/) ||
	                     content.match(/\*\*Trigger[^:]*:\*\*[^`]*`([^`]+)`/) ||
	                     content.match(/\*\*Trigger[^:]*:\*\*[^\n]*?([^\n]+)/);
	const trigger = triggerMatch ? triggerMatch[1].trim() : 'None';

	// Extract strength (supports both table and list format)
	const strengthMatch = content.match(/\|\s*\*\*Strength\*\*\s*\|\s*([^\|]+)\|/) ||
	                      content.match(/\*\*Strength\*\*[^\d\.]*([\d\.\-]+)/);
	const strength = strengthMatch ? strengthMatch[1].trim() : '1.0';

	// Extract type (supports both table and list format)
	const typeMatch = content.match(/\|\s*\*\*Type\*\*\s*\|\s*([^\|]+)\|/) ||
	                  content.match(/\*\*Type\*\*[^:]*:\s*([^\n]+)/);
	const type = typeMatch ? typeMatch[1].trim() : 'LoRA';

	return {
		title: title.replace(/🇧🇳|🌍|#WATW|-/g, '').trim(),
		civitaiUrl,
		modelId,
		downloads: downloadsMatch ? parseInt(downloadsMatch[1].replace(/,/g, '')) : 0,
		rating: ratingMatch ? parseInt(ratingMatch[1]) : 0,
		tips: tipsMatch ? parseInt(tipsMatch[1]) : 0,
		score: scoreMatch ? scoreMatch[1] : '-',
		trigger,
		strength,
		type,
		content,
	};
}

/**
 * Escape YAML string
 */
function escapeYaml(str) {
	if (!str) return '';
	// Replace quotes and special characters
	return str.replace(/"/g, '\\"').replace(/\n/g, ' ').trim();
}

/**
 * Generate Starlight frontmatter
 */
function generateFrontmatter(data, categorySlug, fileName) {
	const imageUrl = data.modelId ? `/images/loras/${fileName}.jpg` : null;
	const safeTitle = escapeYaml(data.title);
	const safeType = escapeYaml(data.type);

	// Calculate relative path based on category depth
	const depth = categorySlug.split('/').length;
	const upLevels = '../'.repeat(depth + 2); // +2 for docs and src folders
	const loraCardPath = `${upLevels}components/LoraCard.astro`;

	return `---
title: "${safeTitle}"
description: "${safeType} for FLUX"
downloads: ${data.downloads}
rating: ${data.rating}
tips: ${data.tips}
score: "${data.score}"
type: "${safeType}"
trigger: "${data.trigger}"
strength: "${data.strength}"
civitaiUrl: "${data.civitaiUrl || ''}"
compatibility: "FLUX"
sidebar:
  badge:
    text: "${data.score}"
    variant: ${data.score === '⭐⭐⭐' ? 'success' : data.score === '⭐⭐' ? 'tip' : 'note'}
---

import LoraCard from '${loraCardPath}';

<LoraCard
  title="${data.title}"
  slug="${categorySlug}/${fileName}"
  downloads={${data.downloads}}
  rating={${data.rating}}
  tips={${data.tips}}
  score="${data.score}"
  type="${data.type}"
  trigger="${data.trigger}"
  strength="${data.strength}"
  civitaiUrl="${data.civitaiUrl || ''}"
  imageUrl="${imageUrl || ''}"
  compatibility="FLUX"
/>

`;
}

/**
 * Process a markdown file
 */
async function processFile(filePath, categorySlug, outputDir) {
	try {
		const content = await fs.readFile(filePath, 'utf-8');
		const fileName = path.basename(filePath, '.md');

		if (fileName === 'INDEX') {
			// Skip INDEX files for now
			return;
		}

		const data = parseMarkdown(content);
		const frontmatter = generateFrontmatter(data, categorySlug, fileName);

		// Remove the first heading (we use frontmatter title)
		const contentWithoutTitle = content.replace(/^#\s+.+$/m, '').trim();

		const newContent = frontmatter + contentWithoutTitle;

		const outputPath = path.join(outputDir, `${fileName}.mdx`);
		await fs.writeFile(outputPath, newContent);

		console.log(`  ✓ Generated: ${fileName}.mdx`);
	} catch (error) {
		console.error(`  ✗ Error processing ${filePath}:`, error.message);
	}
}

/**
 * Process a category directory
 */
async function processCategory(categoryDir, categoryName, categorySlug) {
	console.log(`\n📂 Processing ${categoryName}...`);

	const inputDir = path.join(DOCS_DIR, categoryDir);
	const outputDir = path.join(CONTENT_DIR, categorySlug);

	// Create output directory
	await fs.mkdir(outputDir, { recursive: true });

	try {
		const files = await fs.readdir(inputDir);
		const mdFiles = files.filter(f => f.endsWith('.md'));

		for (const file of mdFiles) {
			const filePath = path.join(inputDir, file);
			await processFile(filePath, categorySlug, outputDir);
		}

		console.log(`✅ ${categoryName}: ${mdFiles.length} files processed`);
	} catch (error) {
		console.error(`Error processing category ${categoryName}:`, error.message);
	}
}

/**
 * Generate category index page
 */
async function generateCategoryIndex(categorySlug, categoryName, description) {
	const outputPath = path.join(CONTENT_DIR, categorySlug, 'index.md');

	const content = `---
title: "${categoryName}"
description: "${description}"
---

# ${categoryName}

${description}

Browse all LoRAs in this category below.
`;

	await fs.writeFile(outputPath, content);
	console.log(`  ✓ Generated category index: ${categorySlug}/index.md`);
}

/**
 * Main function
 */
async function main() {
	console.log('🚀 Generating Starlight content from markdown...\n');

	// Create content directories
	await fs.mkdir(CONTENT_DIR, { recursive: true });

	const categories = [
		{
			dir: 'CHARACTERS_OTHER',
			name: 'Characters - Other',
			slug: 'characters/other',
			description: 'Character LoRAs for consistent character generation',
		},
		{
			dir: 'CHARACTERS_WATW',
			name: 'Women Around The World',
			slug: 'characters/watw',
			description: 'WATW series - 208 countries as character LoRAs',
		},
		{
			dir: 'CHARACTERS_ACORN',
			name: 'Acorn Collection',
			slug: 'characters/acorn',
			description: 'Acorn series character LoRAs',
		},
		{
			dir: 'ANATOMY_BREASTS',
			name: 'Breasts & Nipples',
			slug: 'anatomy/breasts',
			description: 'LoRAs for breast and nipple generation',
		},
		{
			dir: 'ANATOMY_PUSSY',
			name: 'Pussy & Vagina',
			slug: 'anatomy/pussy',
			description: 'LoRAs for vaginal anatomy',
		},
		{
			dir: 'ANATOMY_ASS',
			name: 'Ass & Butt',
			slug: 'anatomy/ass',
			description: 'LoRAs for buttocks and rear anatomy',
		},
		{
			dir: 'POSES',
			name: 'Poses & Angles',
			slug: 'poses',
			description: 'LoRAs for specific poses and camera angles',
		},
		{
			dir: 'CLOTHING',
			name: 'Clothing & Fashion',
			slug: 'clothing',
			description: 'LoRAs for clothing effects and styles',
		},
		{
			dir: 'STYLE_ENHANCEMENT',
			name: 'Style & Enhancement',
			slug: 'style',
			description: 'Style, realism, and quality enhancers',
		},
		{
			dir: 'BODY_TYPES',
			name: 'Body Types',
			slug: 'body-types',
			description: 'LoRAs for body shapes and physiques',
		},
	];

	for (const category of categories) {
		// Create category directories
		const categoryPath = path.join(CONTENT_DIR, category.slug);
		await fs.mkdir(categoryPath, { recursive: true });

		// Generate category index
		await generateCategoryIndex(category.slug, category.name, category.description);

		// Process category files
		await processCategory(category.dir, category.name, category.slug);
	}

	console.log('\n✨ Content generation complete!');
}

// Run main function
main().catch(console.error);
