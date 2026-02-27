/**
 * Survey Response Normalization Utilities
 *
 * Smart categorization system to normalize free-text survey responses
 * into standardized categories for better analytics breakdowns.
 * Uses Fuse.js for fuzzy matching against category keywords.
 */
import Fuse from 'fuse.js'
import type { SurveyResponses, SurveyResponsesNormalized } from '../types'

interface CategoryMapping {
  name: string
  keywords: string[]
  userCount?: number // For reference from analysis
}

/**
 * Industry category mappings based on ~9,000 user analysis
 */
const INDUSTRY_CATEGORIES: CategoryMapping[] = [
  {
    name: 'Film / TV / Animation',
    userCount: 2885,
    keywords: [
      'film',
      'tv',
      'television',
      'animation',
      'animation studio',
      'tv production',
      'film production',
      'story',
      'anime',
      'video',
      'cinematography',
      'visual effects',
      'vfx',
      'vfx artist',
      'movie',
      'cinema',
      'documentary',
      'documentary filmmaker',
      'broadcast',
      'streaming',
      'production',
      'director',
      'filmmaker',
      'post-production',
      'editing'
    ]
  },
  {
    name: 'Marketing / Advertising / Social Media',
    userCount: 1340,
    keywords: [
      'marketing',
      'advertising',
      'youtube',
      'tiktok',
      'social media',
      'content creation',
      'influencer',
      'brand',
      'promotion',
      'digital marketing',
      'seo',
      'campaigns',
      'copywriting',
      'growth',
      'engagement'
    ]
  },
  {
    name: 'Software / IT / AI',
    userCount: 1100,
    keywords: [
      'software',
      'software development',
      'software engineer',
      'it',
      'ai',
      'ai research',
      'corporate ai research',
      'ai research lab',
      'tech company ai research',
      'developer',
      'app developer',
      'consulting',
      'tech',
      'tech startup',
      'programmer',
      'data science',
      'machine learning',
      'coding',
      'programming',
      'web development',
      'app development',
      'saas'
    ]
  },
  {
    name: 'Product & Industrial Design',
    userCount: 1050,
    keywords: [
      'product design',
      'industrial',
      'manufacturing',
      '3d rendering',
      'product visualization',
      'mechanical',
      'automotive',
      'cad',
      'prototype',
      'design engineering',
      'invention'
    ]
  },
  {
    name: 'Fine Art / Contemporary Art',
    userCount: 780,
    keywords: [
      'fine art',
      'art',
      'illustration',
      'contemporary',
      'artist',
      'painting',
      'drawing',
      'sculpture',
      'gallery',
      'canvas',
      'digital art',
      'mixed media',
      'abstract',
      'portrait'
    ]
  },
  {
    name: 'Education / Research',
    userCount: 640,
    keywords: [
      'education',
      'student',
      'teacher',
      'research',
      'university research',
      'academic ai research',
      'university ai research',
      'ai research at university',
      'learning',
      'university',
      'school',
      'academic',
      'professor',
      'curriculum',
      'training',
      'instruction',
      'pedagogy'
    ]
  },
  {
    name: 'Architecture / Engineering / Construction',
    userCount: 420,
    keywords: [
      'architecture',
      'architecture firm',
      'construction',
      'engineering',
      'civil',
      'civil engineering',
      'cad',
      'building',
      'structural',
      'landscape',
      'landscape architecture',
      'interior design',
      'real estate',
      'planning',
      'blueprints'
    ]
  },
  {
    name: 'Gaming / Interactive Media',
    userCount: 410,
    keywords: [
      'gaming',
      'game dev',
      'game development',
      'indie game studio',
      'vr development',
      'roblox',
      'interactive',
      'interactive media',
      'virtual world',
      'vr',
      'ar',
      'metaverse',
      'simulation',
      'unity',
      'unity developer',
      'unreal',
      'indie games'
    ]
  },
  {
    name: 'Photography / Videography',
    userCount: 70,
    keywords: [
      'photography',
      'photo',
      'videography',
      'camera',
      'image',
      'portrait',
      'wedding',
      'commercial photo',
      'stock photography',
      'photojournalism',
      'event photography'
    ]
  },
  {
    name: 'Fashion / Beauty / Retail',
    userCount: 25,
    keywords: [
      'fashion',
      'fashion design',
      'beauty',
      'beauty industry',
      'jewelry',
      'jewelry design',
      'custom jewelry design',
      'retail',
      'retail store',
      'style',
      'clothing',
      'cosmetics',
      'makeup',
      'accessories',
      'boutique'
    ]
  },
  {
    name: 'Music / Performing Arts',
    userCount: 25,
    keywords: [
      'music',
      'music production',
      'vj',
      'dance',
      'projection mapping',
      'audio visual',
      'concert',
      'concert production',
      'performance',
      'theater',
      'stage',
      'live events'
    ]
  },
  {
    name: 'Healthcare / Medical / Life Science',
    userCount: 30,
    keywords: [
      'healthcare',
      'medical',
      'medical research',
      'doctor',
      'biotech',
      'life science',
      'pharmaceutical',
      'clinical',
      'clinical research',
      'hospital',
      'medicine',
      'health'
    ]
  },
  {
    name: 'E-commerce / Print-on-Demand / Business',
    userCount: 15,
    keywords: [
      'ecommerce',
      'e-commerce',
      'print on demand',
      'shop',
      'business',
      'commercial',
      'startup',
      'entrepreneur',
      'sales',
      'online store'
    ]
  },
  {
    name: 'Nonprofit / Government / Public Sector',
    userCount: 15,
    keywords: [
      '501c3',
      'ngo',
      'government',
      'public service',
      'policy',
      'nonprofit',
      'charity',
      'civic',
      'community',
      'social impact'
    ]
  },
  {
    name: 'Adult / NSFW',
    userCount: 10,
    keywords: [
      'nsfw',
      'nsfw content',
      'adult',
      'adult entertainment',
      'erotic',
      'explicit',
      'xxx',
      'porn'
    ]
  }
]

/**
 * Use case category mappings based on common patterns
 */
const USE_CASE_CATEGORIES: CategoryMapping[] = [
  {
    name: 'Content Creation & Marketing',
    keywords: [
      'content creation',
      'social media',
      'marketing',
      'marketing campaigns',
      'advertising',
      'youtube',
      'youtube thumbnail',
      'youtube thumbnail generation',
      'tiktok',
      'instagram',
      'thumbnails',
      'posts',
      'campaigns',
      'brand content'
    ]
  },
  {
    name: 'Art & Illustration',
    keywords: [
      'art',
      'illustration',
      'drawing',
      'painting',
      'concept art',
      'creating concept art',
      'character design',
      'digital art',
      'fantasy art',
      'portraits'
    ]
  },
  {
    name: 'Product Visualization & Design',
    keywords: [
      'product',
      'product mockup',
      'product mockup creation',
      'visualization',
      'prototype visualization',
      'design',
      'prototype',
      'mockup',
      '3d rendering',
      'industrial design',
      'product photos'
    ]
  },
  {
    name: 'Film & Video Production',
    keywords: [
      'film',
      'video',
      'video editing',
      'movie',
      'movie production',
      'animation',
      'vfx',
      'visual effects',
      'storyboard',
      'storyboard creation',
      'cinematography',
      'post production'
    ]
  },
  {
    name: 'Gaming & Interactive Media',
    keywords: [
      'game',
      'gaming',
      'game asset generation',
      'game assets',
      'game development',
      'game textures',
      'interactive',
      'vr',
      'vr content creation',
      'ar',
      'virtual',
      'simulation',
      'metaverse',
      'textures'
    ]
  },
  {
    name: 'Architecture & Construction',
    keywords: [
      'architecture',
      'architectural rendering',
      'building',
      'building visualization',
      'construction',
      'interior design',
      'interior design mockups',
      'landscape',
      'real estate',
      'real estate visualization',
      'floor plans',
      'renderings'
    ]
  },
  {
    name: 'Education & Training',
    keywords: [
      'education',
      'educational',
      'educational content',
      'training',
      'training materials',
      'learning',
      'teaching',
      'tutorial',
      'tutorial creation',
      'course',
      'academic',
      'academic projects',
      'instructional',
      'workshops'
    ]
  },
  {
    name: 'Research & Development',
    keywords: [
      'research',
      'research experiments',
      'development',
      'experiment',
      'prototype',
      'prototype testing',
      'testing',
      'analysis',
      'study',
      'innovation',
      'innovation projects',
      'r&d',
      'scientific visualization'
    ]
  },
  {
    name: 'Personal & Hobby',
    keywords: [
      'personal',
      'personal art projects',
      'hobby',
      'hobby work',
      'fun',
      'fun experiments',
      'experiment',
      'learning',
      'curiosity',
      'explore',
      'creative',
      'creative exploration',
      'side project'
    ]
  },
  {
    name: 'Photography & Image Processing',
    keywords: [
      'photography',
      'product photography',
      'portrait photography',
      'photo',
      'photo editing',
      'image',
      'image enhancement',
      'portrait',
      'editing',
      'enhancement',
      'restoration',
      'photo manipulation'
    ]
  }
]

/**
 * Fuse.js configuration for category matching
 */
const FUSE_OPTIONS = {
  keys: ['keywords'],
  threshold: 0.53, // Higher = more lenient matching
  minMatchCharLength: 5,
  includeScore: true,
  includeMatches: true,
  ignoreLocation: true,
  findAllMatches: true
}

/**
 * Create Fuse instances for category matching
 */
const industryFuse = new Fuse(INDUSTRY_CATEGORIES, FUSE_OPTIONS)
const useCaseFuse = new Fuse(USE_CASE_CATEGORIES, FUSE_OPTIONS)

/**
 * Normalize industry responses using Fuse.js fuzzy search
 */
export function normalizeIndustry(rawIndustry: unknown): string {
  if (!rawIndustry || typeof rawIndustry !== 'string') {
    return 'Other / Undefined'
  }

  const industry = rawIndustry.toLowerCase().trim()

  // Handle common undefined responses
  if (
    industry.match(/^(other|none|undefined|unknown|n\/a|not applicable|-|)$/)
  ) {
    return 'Other / Undefined'
  }

  // Fuse.js fuzzy search for best category match
  const results = industryFuse.search(rawIndustry)

  if (results.length > 0) {
    return results[0].item.name
  }

  // No good match found - preserve original with prefix
  return `Uncategorized: ${rawIndustry}`
}

/**
 * Normalize use case responses using Fuse.js fuzzy search
 */
export function normalizeUseCase(rawUseCase: unknown): string {
  if (!rawUseCase || typeof rawUseCase !== 'string') {
    return 'Other / Undefined'
  }

  const useCase = rawUseCase.toLowerCase().trim()

  // Handle common undefined responses
  if (
    useCase.match(/^(other|none|undefined|unknown|n\/a|not applicable|-|)$/)
  ) {
    return 'Other / Undefined'
  }

  // Fuse.js fuzzy search for best category match
  const results = useCaseFuse.search(rawUseCase)

  if (results.length > 0) {
    return results[0].item.name
  }

  // No good match found - preserve original with prefix
  return `Uncategorized: ${rawUseCase}`
}

/**
 * Apply normalization to survey responses
 * Creates both normalized and raw versions of responses
 */
export function normalizeSurveyResponses(
  responses: SurveyResponses
): SurveyResponsesNormalized {
  const normalized: SurveyResponsesNormalized = { ...responses }

  // Normalize industry
  if (typeof responses.industry === 'string') {
    normalized.industry_normalized = normalizeIndustry(responses.industry)
    normalized.industry_raw = responses.industry
  }

  // Normalize use case
  if (typeof responses.useCase === 'string') {
    normalized.useCase_normalized = normalizeUseCase(responses.useCase)
    normalized.useCase_raw = responses.useCase
  }

  return normalized
}
