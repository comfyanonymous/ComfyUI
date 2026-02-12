import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DOCS_DIR = path.join(__dirname, '..', '..');
const CACHE_FILE = path.join(__dirname, '..', 'civitai-image-cache.json');

// Load cache
let cache = {};
if (fs.existsSync(CACHE_FILE)) {
	const cacheData = fs.readFileSync(CACHE_FILE, 'utf-8');
	cache = JSON.parse(cacheData);
	console.log(`📦 Loaded cache with ${Object.keys(cache).length} entries`);
}

// Categories to process
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

function getScoreFromDownloads(downloads) {
	if (downloads >= 3000) return '⭐⭐⭐';
	if (downloads >= 1000) return '⭐⭐';
	if (downloads >= 500) return '⭐';
	return '-';
}

function updateMarkdownFile(filePath, stats) {
	let content = fs.readFileSync(filePath, 'utf-8');

	// Check if already has stats table
	if (content.includes('| **Downloads** |')) {
		console.log(`  ⏭️  Already has stats table, skipping`);
		return false;
	}

	// Find the Info section or table
	const infoMatch = content.match(/## Info\n([\s\S]*?)(?=\n## |$)/);
	if (!infoMatch) {
		console.log(`  ⚠️  No Info section found`);
		return false;
	}

	const infoSection = infoMatch[1];

	// Extract existing data
	const fileMatch = infoSection.match(/\*\*File\*\*[:\s]*`?([^`\n]+)`?/);
	const origMatch = infoSection.match(/\*\*Original filename\*\*[:\s]*`?([^`\n]+)`?/);
	const civitaiMatch = infoSection.match(/\*\*Civitai\*\*[:\s]*(https:\/\/[^\s\n]+)/);
	const triggerMatch = infoSection.match(/\*\*Trigger[^:]*\*\*[:\s]*`?([^`\n]+)`?/);
	const strengthMatch = infoSection.match(/\*\*Strength\*\*[:\s]*([^\n]+)/);
	const typeMatch = infoSection.match(/\*\*Type\*\*[:\s]*([^\n]+)/);

	// Build new table
	let table = '| Parameter | Value |\n|-----------|-------|\n';

	if (fileMatch) table += `| **File** | \`${fileMatch[1].trim()}\` |\n`;
	if (origMatch) table += `| **Original filename** | \`${origMatch[1].trim()}\` |\n`;
	if (civitaiMatch) table += `| **Civitai** | ${civitaiMatch[1]} |\n`;

	// Add stats if available
	if (stats) {
		table += `| **Downloads** | ${stats.downloads.toLocaleString()} |\n`;
		table += `| **👍** | ${stats.rating} |\n`;
		table += `| **Tips** | $${stats.tips} |\n`;
		table += `| **Score** | ${stats.score} |\n`;
	}

	if (triggerMatch) table += `| **Trigger** | \`${triggerMatch[1].trim()}\` |\n`;
	if (strengthMatch) table += `| **Strength** | ${strengthMatch[1].trim()} |\n`;
	if (typeMatch) table += `| **Type** | ${typeMatch[1].trim()} |\n`;

	// Replace the Info section
	const newContent = content.replace(
		/## Info\n[\s\S]*?(?=\n## |$)/,
		`## Info\n${table}`
	);

	fs.writeFileSync(filePath, newContent, 'utf-8');
	return true;
}

async function processCategory(category) {
	const dirPath = path.join(DOCS_DIR, category.dir);

	if (!fs.existsSync(dirPath)) {
		console.log(`📂 Skipping ${category.name} (directory not found)`);
		return;
	}

	console.log(`\n📂 Processing ${category.name}...`);

	const files = fs.readdirSync(dirPath).filter(f =>
		f.endsWith('.md') && f !== 'INDEX.md'
	);

	console.log(`   Found ${files.length} files`);

	let updated = 0;
	let skipped = 0;

	for (const file of files) {
		const filePath = path.join(dirPath, file);
		console.log(`\n  📄 ${file}`);

		// Read file to get Civitai URL
		const content = fs.readFileSync(filePath, 'utf-8');
		const civitaiMatch = content.match(/civitai\.com\/models\/(\d+)/);

		if (!civitaiMatch) {
			console.log(`  ⚠️  No Civitai URL found`);
			skipped++;
			continue;
		}

		const modelId = civitaiMatch[1];
		const cachedData = cache[modelId];

		let stats = null;
		if (cachedData && cachedData.downloads) {
			stats = {
				downloads: cachedData.downloads,
				rating: cachedData.rating || 0,
				tips: cachedData.tips || 0,
				score: getScoreFromDownloads(cachedData.downloads)
			};
			console.log(`  ✓ Using cached stats: ${stats.downloads} downloads`);
		} else {
			console.log(`  ⚠️  No cached stats for model ${modelId}`);
		}

		if (updateMarkdownFile(filePath, stats)) {
			console.log(`  ✅ Updated`);
			updated++;
		} else {
			skipped++;
		}
	}

	console.log(`\n  ✅ ${category.name}: ${updated} updated, ${skipped} skipped\n`);
}

async function main() {
	console.log('🚀 Starting markdown format update...\n');

	for (const category of categories) {
		await processCategory(category);
	}

	console.log('\n✨ Done!\n');
}

main().catch(console.error);
