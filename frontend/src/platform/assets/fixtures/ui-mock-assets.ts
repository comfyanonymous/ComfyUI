import type { AssetItem } from '@/platform/assets/schemas/assetSchema'

// 🎭 OBVIOUSLY FAKE MOCK DATA - DO NOT USE IN PRODUCTION! 🎭
const fakeFunnyModelNames = [
  '🎯_totally_real_model_v420.69',
  '🚀_definitely_not_fake_v999',
  '🎪_super_legit_checkpoint_pro_max',
  '🦄_unicorn_dreams_totally_real.model',
  '🍕_pizza_generator_supreme',
  '🎸_rock_star_fake_data_v1337',
  '🌮_taco_tuesday_model_deluxe',
  '🦖_dino_nugget_generator_v3',
  '🎮_gamer_fuel_checkpoint_xl',
  '🍄_mushroom_kingdom_diffusion',
  '🏴‍☠️_pirate_treasure_model_arr',
  '🦋_butterfly_effect_generator',
  '🎺_jazz_hands_checkpoint_pro',
  '🥨_pretzel_logic_model_v2',
  '🌙_midnight_snack_generator',
  '🎭_drama_llama_checkpoint',
  '🧙‍♀️_wizard_hat_diffusion_xl',
  '🎪_circus_peanut_model_v4',
  '🦒_giraffe_neck_generator',
  '🎲_random_stuff_checkpoint_max'
]

const obviouslyFakeDescriptions = [
  '⚠️ FAKE DATA: Generates 100% authentic fake images with premium mock quality',
  '🎭 MOCK ALERT: This totally real model creates absolutely genuine fake content',
  '🚨 NOT REAL: Professional-grade fake imagery for your mock data needs',
  '🎪 DEMO ONLY: Circus-quality fake generation with extra mock seasoning',
  '🍕 FAKE FOOD: Generates delicious fake pizzas (not edible in reality)',
  "🎸 MOCK ROCK: Creates fake rock stars who definitely don't exist",
  '🌮 TACO FAKERY: Tuesday-themed fake tacos for your mock appetite',
  '🦖 PREHISTORIC FAKE: Generates extinct fake dinosaurs for demo purposes',
  '🎮 FAKE GAMING: Level up your mock data with obviously fake content',
  '🍄 FUNGI FICTION: Magical fake mushrooms from the demo dimension',
  '🏴‍☠️ FAKE TREASURE: Arr! This be mock data for ye demo needs, matey!',
  '🦋 DEMO EFFECT: Small fake changes create big mock differences',
  '🎺 JAZZ FAKERY: Smooth fake jazz for your mock listening pleasure',
  '🥨 MOCK LOGIC: Twisted fake reasoning for your demo requirements',
  '🌙 MIDNIGHT MOCK: Late-night fake snacks for your demo hunger',
  '🎭 FAKE DRAMA: Over-the-top mock emotions for demo entertainment',
  '🧙‍♀️ WIZARD MOCK: Magically fake spells cast with demo ingredients',
  '🎪 CIRCUS FAKE: Big top mock entertainment under the demo tent',
  '🦒 TALL FAKE: Reaches new heights of obviously fake content',
  '🎲 RANDOM MOCK: Generates random fake stuff for your demo pleasure'
]

// API-compliant tag structure: first tag must be root (models/input/output), second is category
const modelCategories = ['checkpoints', 'loras', 'embeddings', 'vae']
const baseModels = ['sd15', 'sdxl', 'sd35']
const fileExtensions = ['.safetensors', '.ckpt', '.pt']
const mimeTypes = [
  'application/octet-stream',
  'application/x-pytorch',
  'application/x-safetensors'
]

function getRandomElement<T>(array: T[]): T {
  return array[Math.floor(Math.random() * array.length)]
}

function getRandomNumber(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min
}

function getRandomISODate(): string {
  const start = new Date('2024-01-01').getTime()
  const end = new Date('2024-12-31').getTime()
  const randomTime = start + Math.random() * (end - start)
  return new Date(randomTime).toISOString()
}

function generateFakeAssetHash(): string {
  const chars = '0123456789abcdef'
  let hash = 'blake3:'
  for (let i = 0; i < 64; i++) {
    hash += chars[Math.floor(Math.random() * chars.length)]
  }
  return hash
}

// 🎭 CREATES OBVIOUSLY FAKE ASSETS FOR DEMO/TEST PURPOSES ONLY! 🎭
export function createMockAssets(count: number = 20): AssetItem[] {
  return Array.from({ length: count }, (_, index) => {
    const category = getRandomElement(modelCategories)
    const baseModel = getRandomElement(baseModels)
    const extension = getRandomElement(fileExtensions)
    const mimeType = getRandomElement(mimeTypes)
    const sizeInBytes = getRandomNumber(
      500 * 1024 * 1024,
      8 * 1024 * 1024 * 1024
    ) // 500MB to 8GB
    const createdAt = getRandomISODate()
    const updatedAt = createdAt
    const lastAccessTime = getRandomISODate()

    const fakeFileName = `${fakeFunnyModelNames[index]}${extension}`

    return {
      id: `mock-asset-uuid-${(index + 1).toString().padStart(3, '0')}-fake`,
      name: fakeFileName,
      asset_hash: generateFakeAssetHash(),
      size: sizeInBytes,
      mime_type: mimeType,
      tags: [
        'models', // Root tag (required first)
        category, // Category tag (required second for models)
        'fake-data', // Obviously fake tag
        ...(Math.random() > 0.5 ? ['demo-mode'] : ['test-only']),
        ...(Math.random() > 0.7 ? ['obviously-mock'] : [])
      ],
      preview_url: `/api/assets/mock-asset-uuid-${(index + 1).toString().padStart(3, '0')}-fake/content`,
      created_at: createdAt,
      updated_at: updatedAt,
      last_access_time: lastAccessTime,
      user_metadata: {
        description: obviouslyFakeDescriptions[index],
        base_model: baseModel,
        original_name: fakeFunnyModelNames[index],
        warning: '🚨 THIS IS FAKE DEMO DATA - NOT A REAL MODEL! 🚨'
      }
    }
  })
}

export const mockAssets = createMockAssets(20)

// 🧪 Test helpers for edge cases - built on mock asset foundation
export function createAssetWithoutExtension() {
  const asset = createMockAssets(1)[0]
  asset.name = 'model_no_extension'
  return asset
}

export function createAssetWithoutBaseModel() {
  const asset = createMockAssets(1)[0]
  asset.user_metadata = { description: 'A test model' }
  return asset
}

export function createAssetWithoutUserMetadata() {
  const asset = createMockAssets(1)[0]
  asset.user_metadata = undefined
  return asset
}

export function createAssetWithSpecificExtension(
  extension: string,
  isImmutable?: boolean
) {
  const asset = createMockAssets(1)[0]
  asset.name = `test-model.${extension}`
  if (isImmutable !== undefined) {
    asset.is_immutable = isImmutable
  }
  return asset
}

export function createAssetWithSpecificBaseModel(baseModel: string) {
  const asset = createMockAssets(1)[0]
  asset.user_metadata = { ...asset.user_metadata, base_model: baseModel }
  return asset
}
