# Contributing Translations to ComfyUI

## Quick Start for New Languages

1. **Let us know** - Open an issue or reach out on Discord to request a new language
2. **Get technical setup help** - We'll help configure the initial files or you can follow the technical process below
3. **Automatic translation** - Our CI system will generate translations using OpenAI when you create a PR
4. **Review and refine** - You can improve the auto-generated translations and become a maintainer for that language

## Technical Process (Confirmed Working)

### Prerequisites

- Node.js installed
- Git/GitHub knowledge
- OpenAI API key (optional - CI will handle translations)

### Step 1: Update Configuration Files

**Time required: ~10 minutes**

#### 1.1 Update `.i18nrc.cjs`

Add your language code to the `outputLocales` array:

```javascript
module.exports = defineConfig({
  // ... existing config
  outputLocales: ['zh', 'zh-TW', 'ru', 'ja', 'ko', 'fr', 'es', 'tr'], // Add your language here
  reference: `Special names to keep untranslated: flux, photomaker, clip, vae, cfg, stable audio, stable cascade, stable zero, controlnet, lora, HiDream.
  'latent' is the short form of 'latent space'.
  'mask' is in the context of image processing.
  Note: For Traditional Chinese (Taiwan), use Taiwan-specific terminology and traditional characters.
  `
})
```

#### 1.2 Update `src/constants/coreSettings.ts`

Add your language to the dropdown options:

```typescript
{
  id: 'Comfy.Locale',
  name: 'Language',
  type: 'combo',
  options: [
    { value: 'en', text: 'English' },
    { value: 'zh', text: '中文' },
    { value: 'zh-TW', text: '繁體中文 (台灣)' }, // Add your language here
    { value: 'ru', text: 'Русский' },
    { value: 'ja', text: '日本語' },
    { value: 'ko', text: '한국어' },
    { value: 'fr', text: 'Français' },
    { value: 'es', text: 'Español' }
  ],
  defaultValue: () => navigator.language.split('-')[0] || 'en'
},
```

#### 1.3 Update `src/i18n.ts`

Add imports for your new language files:

```typescript
// Add these imports (replace zh-TW with your language code)
import zhTWCommands from './locales/zh-TW/commands.json'
import zhTW from './locales/zh-TW/main.json'
import zhTWNodes from './locales/zh-TW/nodeDefs.json'
import zhTWSettings from './locales/zh-TW/settings.json'

// Add to the messages object
const messages = {
  en: buildLocale(en, enNodes, enCommands, enSettings),
  zh: buildLocale(zh, zhNodes, zhCommands, zhSettings),
  'zh-TW': buildLocale(zhTW, zhTWNodes, zhTWCommands, zhTWSettings) // Add this line
  // ... other languages
}
```

### Step 2: Generate Translation Files

#### Option A: Local Generation (Optional)

```bash
# Only if you have OpenAI API key configured
pnpm locale
```

#### Option B: Let CI Handle It (Recommended)

- Create your PR with the configuration changes above
- **Important**: Translation files will be generated during release PRs, not feature PRs
- Empty JSON files are fine - they'll be populated during the next release workflow
- For urgent translation needs, maintainers can manually trigger the workflow

### Step 3: Test Your Changes

```bash
pnpm typecheck  # Check for TypeScript errors
pnpm dev        # Start development server
```

**Testing checklist:**

- [ ] Language appears in ComfyUI Settings > Locale dropdown
- [ ] Can select the new language without errors
- [ ] Partial translations display correctly
- [ ] UI falls back to English for untranslated strings
- [ ] No console errors when switching languages

### Step 4: Submit PR

1. **Create PR** with your configuration changes
2. **CI will run** and automatically populate translation files
3. **Request review** from language maintainers: @Yorha4D @KarryCharon @DorotaLuna @shinshin86
4. **Get added to CODEOWNERS** as a reviewer for your language

## What Happens in CI

Our automated translation workflow now runs on release PRs (version-bump-\* branches) to improve development performance:

### For Feature PRs (Regular Development)

- **No automatic translations** - faster reviews and fewer conflicts
- **English-only development** - new strings show in English until release
- **Focus on functionality** - reviewers see only your actual changes

### For Release PRs (version-bump-\* branches)

1. **Collects strings**: Scans the UI for translatable text
2. **Updates English files**: Ensures all strings are captured
3. **Generates translations**: Uses OpenAI API to translate to all configured languages
4. **Commits back**: Automatically updates the release PR with complete translations

### Manual Translation Updates

If urgent translation updates are needed outside of releases, maintainers can:

- Trigger the "Update Locales" workflow manually from GitHub Actions
- The workflow supports manual dispatch for emergency translation updates

## File Structure

Each language has 4 translation files:

- `main.json` - Main UI text (~2000+ entries)
- `commands.json` - Command descriptions (~200+ entries)
- `settings.json` - Settings panel (~400+ entries)
- `nodeDefs.json` - Node definitions (~varies based on installed nodes)

## Translation Quality

- **Auto-translations are high quality** but may need refinement
- **Technical terms** are preserved (flux, photomaker, clip, vae, etc.)
- **Context-aware** translations based on UI usage
- **Native speaker review** is encouraged for quality improvements

## Common Issues & Solutions

### Issue: TypeScript errors on imports

**Solution**: Ensure your language code matches exactly in all three files

### Issue: Empty translation files

**Solution**: This is normal - CI will populate them when you create a PR

### Issue: Language not appearing in dropdown

**Solution**: Check that the language code in `coreSettings.ts` matches your other files exactly

### Issue: Rate limits during local translation

**Solution**: This is expected - let CI handle the translation generation

## Regional Variants

For regional variants (like zh-TW for Taiwan), use:

- **Language-region codes**: `zh-TW`, `pt-BR`, `en-US`
- **Specific terminology**: Add region-specific context to the reference string
- **Native display names**: Use the local language name in the dropdown

## Getting Help

- **Tag translation maintainers**: @Yorha4D @KarryCharon @DorotaLuna @shinshin86
- **Check existing language PRs** for examples
- **Open an issue** describing your language addition request
- **Reference this tested process** - we've confirmed it works!

## Becoming a Language Maintainer

After your language is added:

1. **Get added to CODEOWNERS** for your language files
2. **Review future PRs** affecting your language
3. **Coordinate with other native speakers** for quality improvements
4. **Help maintain translations** as the UI evolves

---

_This process was tested and confirmed working with Traditional Chinese (Taiwan) addition._
