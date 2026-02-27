/**
 * Unit tests for survey normalization utilities
 * Uses real example data from migration script to verify categorization accuracy
 */

import { describe, expect, it } from 'vitest'

import {
  normalizeIndustry,
  normalizeUseCase,
  normalizeSurveyResponses
} from '../surveyNormalization'

describe('normalizeIndustry', () => {
  describe('Film / TV / Animation category', () => {
    it('should categorize film and television production', () => {
      expect(normalizeIndustry('Film and television production')).toBe(
        'Film / TV / Animation'
      )
      expect(normalizeIndustry('film')).toBe('Film / TV / Animation')
      expect(normalizeIndustry('TV production')).toBe('Film / TV / Animation')
      expect(normalizeIndustry('animation studio')).toBe(
        'Film / TV / Animation'
      )
      expect(normalizeIndustry('VFX artist')).toBe('Film / TV / Animation')
      expect(normalizeIndustry('cinema')).toBe('Film / TV / Animation')
      expect(normalizeIndustry('documentary filmmaker')).toBe(
        'Film / TV / Animation'
      )
    })

    it('should handle typos and variations', () => {
      expect(normalizeIndustry('animtion')).toBe('Film / TV / Animation')
      expect(normalizeIndustry('film prod')).toBe('Film / TV / Animation')
      expect(normalizeIndustry('movie production')).toBe(
        'Film / TV / Animation'
      )
    })
  })

  describe('Marketing / Advertising / Social Media category', () => {
    it('should categorize marketing and social media', () => {
      expect(normalizeIndustry('Marketing & Social Media')).toBe(
        'Marketing / Advertising / Social Media'
      )
      expect(normalizeIndustry('digital marketing')).toBe(
        'Marketing / Advertising / Social Media'
      )
      expect(normalizeIndustry('YouTube content creation')).toBe(
        'Marketing / Advertising / Social Media'
      )
      expect(normalizeIndustry('TikTok marketing')).toBe(
        'Marketing / Advertising / Social Media'
      )
      expect(normalizeIndustry('brand promotion')).toBe(
        'Marketing / Advertising / Social Media'
      )
      expect(normalizeIndustry('influencer marketing')).toBe(
        'Marketing / Advertising / Social Media'
      )
    })

    it('should handle social media variations', () => {
      expect(normalizeIndustry('social content')).toBe(
        'Marketing / Advertising / Social Media'
      )
      expect(normalizeIndustry('content creation')).toBe(
        'Marketing / Advertising / Social Media'
      )
    })
  })

  describe('Software / IT / AI category', () => {
    it('should categorize software development', () => {
      expect(normalizeIndustry('Software Development')).toBe(
        'Software / IT / AI'
      )
      expect(normalizeIndustry('tech startup')).toBe('Software / IT / AI')
      expect(normalizeIndustry('AI research')).toBe('Software / IT / AI')
      expect(normalizeIndustry('web development')).toBe('Software / IT / AI')
      expect(normalizeIndustry('machine learning')).toBe('Software / IT / AI')
      expect(normalizeIndustry('data science')).toBe('Software / IT / AI')
      expect(normalizeIndustry('programming')).toBe('Software / IT / AI')
    })

    it('should handle IT variations', () => {
      expect(normalizeIndustry('software engineer')).toBe('Software / IT / AI')
      expect(normalizeIndustry('app developer')).toBe('Software / IT / AI')
    })

    it('should categorize corporate AI research', () => {
      expect(normalizeIndustry('corporate AI research')).toBe(
        'Software / IT / AI'
      )
      expect(normalizeIndustry('AI research lab')).toBe('Software / IT / AI')
      expect(normalizeIndustry('tech company AI research')).toBe(
        'Software / IT / AI'
      )
    })
  })

  describe('Gaming / Interactive Media category', () => {
    it('should categorize gaming industry', () => {
      expect(normalizeIndustry('Indie Game Studio')).toBe(
        'Gaming / Interactive Media'
      )
      expect(normalizeIndustry('game development')).toBe(
        'Gaming / Interactive Media'
      )
      expect(normalizeIndustry('VR development')).toBe(
        'Gaming / Interactive Media'
      )
      expect(normalizeIndustry('interactive media')).toBe(
        'Gaming / Interactive Media'
      )
      expect(normalizeIndustry('metaverse')).toBe('Gaming / Interactive Media')
      expect(normalizeIndustry('Unity developer')).toBe(
        'Gaming / Interactive Media'
      )
    })

    it('should handle game dev variations', () => {
      expect(normalizeIndustry('game dev')).toBe('Gaming / Interactive Media')
      expect(normalizeIndustry('indie games')).toBe(
        'Gaming / Interactive Media'
      )
    })
  })

  describe('Architecture / Engineering / Construction category', () => {
    it('should categorize architecture and construction', () => {
      expect(normalizeIndustry('Architecture firm')).toBe(
        'Architecture / Engineering / Construction'
      )
      expect(normalizeIndustry('construction')).toBe(
        'Architecture / Engineering / Construction'
      )
      expect(normalizeIndustry('civil engineering')).toBe(
        'Architecture / Engineering / Construction'
      )
      expect(normalizeIndustry('interior design')).toBe(
        'Architecture / Engineering / Construction'
      )
      expect(normalizeIndustry('landscape architecture')).toBe(
        'Architecture / Engineering / Construction'
      )
      expect(normalizeIndustry('real estate')).toBe(
        'Architecture / Engineering / Construction'
      )
    })
  })

  describe('Fashion / Beauty / Retail category', () => {
    it('should categorize fashion and beauty', () => {
      expect(normalizeIndustry('Custom Jewelry Design')).toBe(
        'Fashion / Beauty / Retail'
      )
      expect(normalizeIndustry('fashion design')).toBe(
        'Fashion / Beauty / Retail'
      )
      expect(normalizeIndustry('beauty industry')).toBe(
        'Fashion / Beauty / Retail'
      )
      expect(normalizeIndustry('retail store')).toBe(
        'Fashion / Beauty / Retail'
      )
      expect(normalizeIndustry('cosmetics')).toBe('Fashion / Beauty / Retail')
    })
  })

  describe('Healthcare / Medical / Life Science category', () => {
    it('should categorize medical and health fields', () => {
      expect(normalizeIndustry('Medical Research')).toBe(
        'Healthcare / Medical / Life Science'
      )
      expect(normalizeIndustry('healthcare')).toBe(
        'Healthcare / Medical / Life Science'
      )
      expect(normalizeIndustry('biotech')).toBe(
        'Healthcare / Medical / Life Science'
      )
      expect(normalizeIndustry('pharmaceutical')).toBe(
        'Healthcare / Medical / Life Science'
      )
      expect(normalizeIndustry('clinical research')).toBe(
        'Healthcare / Medical / Life Science'
      )
    })
  })

  describe('Education / Research category', () => {
    it('should categorize education and research', () => {
      expect(normalizeIndustry('university research')).toBe(
        'Education / Research'
      )
      expect(normalizeIndustry('academic')).toBe('Education / Research')
      expect(normalizeIndustry('teaching')).toBe('Education / Research')
      expect(normalizeIndustry('student')).toBe('Education / Research')
      expect(normalizeIndustry('professor')).toBe('Education / Research')
    })

    it('should categorize academic AI research', () => {
      expect(normalizeIndustry('academic AI research')).toBe(
        'Education / Research'
      )
      expect(normalizeIndustry('university AI research')).toBe(
        'Education / Research'
      )
      expect(normalizeIndustry('AI research at university')).toBe(
        'Education / Research'
      )
    })
  })

  describe('Fine Art / Contemporary Art category', () => {
    it('should categorize art fields', () => {
      expect(normalizeIndustry('fine art')).toBe('Fine Art / Contemporary Art')
      expect(normalizeIndustry('contemporary artist')).toBe(
        'Fine Art / Contemporary Art'
      )
      expect(normalizeIndustry('digital art')).toBe(
        'Fine Art / Contemporary Art'
      )
      expect(normalizeIndustry('illustration')).toBe(
        'Fine Art / Contemporary Art'
      )
      expect(normalizeIndustry('gallery')).toBe('Fine Art / Contemporary Art')
    })
  })

  describe('Photography / Videography category', () => {
    it('should categorize photography fields', () => {
      expect(normalizeIndustry('photography')).toBe('Photography / Videography')
      expect(normalizeIndustry('wedding photography')).toBe(
        'Photography / Videography'
      )
      expect(normalizeIndustry('commercial photo')).toBe(
        'Photography / Videography'
      )
      expect(normalizeIndustry('videography')).toBe('Photography / Videography')
    })
  })

  describe('Product & Industrial Design category', () => {
    it('should categorize product design', () => {
      expect(normalizeIndustry('product design')).toBe(
        'Product & Industrial Design'
      )
      expect(normalizeIndustry('industrial design')).toBe(
        'Product & Industrial Design'
      )
      expect(normalizeIndustry('manufacturing')).toBe(
        'Product & Industrial Design'
      )
      expect(normalizeIndustry('3d rendering')).toBe(
        'Product & Industrial Design'
      )
      expect(normalizeIndustry('automotive design')).toBe(
        'Product & Industrial Design'
      )
    })
  })

  describe('Music / Performing Arts category', () => {
    it('should categorize music and performing arts', () => {
      expect(normalizeIndustry('music production')).toBe(
        'Music / Performing Arts'
      )
      expect(normalizeIndustry('theater')).toBe('Music / Performing Arts')
      expect(normalizeIndustry('concert production')).toBe(
        'Music / Performing Arts'
      )
      expect(normalizeIndustry('live events')).toBe('Music / Performing Arts')
    })
  })

  describe('E-commerce / Print-on-Demand / Business category', () => {
    it('should categorize business fields', () => {
      expect(normalizeIndustry('ecommerce')).toBe(
        'E-commerce / Print-on-Demand / Business'
      )
      expect(normalizeIndustry('print on demand')).toBe(
        'E-commerce / Print-on-Demand / Business'
      )
      expect(normalizeIndustry('startup')).toBe(
        'E-commerce / Print-on-Demand / Business'
      )
      expect(normalizeIndustry('online store')).toBe(
        'E-commerce / Print-on-Demand / Business'
      )
    })
  })

  describe('Nonprofit / Government / Public Sector category', () => {
    it('should categorize nonprofit and government', () => {
      expect(normalizeIndustry('nonprofit')).toBe(
        'Nonprofit / Government / Public Sector'
      )
      expect(normalizeIndustry('government agency')).toBe(
        'Nonprofit / Government / Public Sector'
      )
      expect(normalizeIndustry('public service')).toBe(
        'Nonprofit / Government / Public Sector'
      )
      expect(normalizeIndustry('charity')).toBe(
        'Nonprofit / Government / Public Sector'
      )
    })
  })

  describe('Adult / NSFW category', () => {
    it('should categorize adult content', () => {
      expect(normalizeIndustry('adult entertainment')).toBe('Adult / NSFW')
      expect(normalizeIndustry('NSFW content')).toBe('Adult / NSFW')
    })
  })

  describe('Other / Undefined category', () => {
    it('should handle undefined responses', () => {
      expect(normalizeIndustry('other')).toBe('Other / Undefined')
      expect(normalizeIndustry('none')).toBe('Other / Undefined')
      expect(normalizeIndustry('undefined')).toBe('Other / Undefined')
      expect(normalizeIndustry('unknown')).toBe('Other / Undefined')
      expect(normalizeIndustry('n/a')).toBe('Other / Undefined')
      expect(normalizeIndustry('not applicable')).toBe('Other / Undefined')
      expect(normalizeIndustry('-')).toBe('Other / Undefined')
      expect(normalizeIndustry('')).toBe('Other / Undefined')
    })

    it('should handle null and invalid inputs', () => {
      expect(normalizeIndustry(null)).toBe('Other / Undefined')
      expect(normalizeIndustry(undefined)).toBe('Other / Undefined')
      expect(normalizeIndustry(123)).toBe('Other / Undefined')
    })
  })

  describe('Uncategorized responses', () => {
    it('should preserve unknown creative fields with prefix', () => {
      expect(normalizeIndustry('Unknown Creative Field')).toBe(
        'Uncategorized: Unknown Creative Field'
      )
      expect(normalizeIndustry('Completely Novel Field')).toBe(
        'Uncategorized: Completely Novel Field'
      )
    })
  })
})

describe('normalizeUseCase', () => {
  describe('Content Creation & Marketing', () => {
    it('should categorize content creation', () => {
      expect(normalizeUseCase('YouTube thumbnail generation')).toBe(
        'Content Creation & Marketing'
      )
      expect(normalizeUseCase('social media content')).toBe(
        'Content Creation & Marketing'
      )
      expect(normalizeUseCase('marketing campaigns')).toBe(
        'Content Creation & Marketing'
      )
      expect(normalizeUseCase('TikTok content')).toBe(
        'Content Creation & Marketing'
      )
      expect(normalizeUseCase('brand content creation')).toBe(
        'Content Creation & Marketing'
      )
    })
  })

  describe('Art & Illustration', () => {
    it('should categorize art and illustration', () => {
      expect(normalizeUseCase('Creating concept art for movies')).toBe(
        'Art & Illustration'
      )
      expect(normalizeUseCase('digital art')).toBe('Art & Illustration')
      expect(normalizeUseCase('character design')).toBe('Art & Illustration')
      expect(normalizeUseCase('illustration work')).toBe('Art & Illustration')
      expect(normalizeUseCase('fantasy art')).toBe('Art & Illustration')
    })
  })

  describe('Product Visualization & Design', () => {
    it('should categorize product work', () => {
      expect(normalizeUseCase('Product mockup creation')).toBe(
        'Product Visualization & Design'
      )
      expect(normalizeUseCase('3d product rendering')).toBe(
        'Product Visualization & Design'
      )
      expect(normalizeUseCase('prototype visualization')).toBe(
        'Product Visualization & Design'
      )
      expect(normalizeUseCase('industrial design')).toBe(
        'Product Visualization & Design'
      )
    })
  })

  describe('Gaming & Interactive Media', () => {
    it('should categorize gaming use cases', () => {
      expect(normalizeUseCase('Game asset generation')).toBe(
        'Gaming & Interactive Media'
      )
      expect(normalizeUseCase('game development')).toBe(
        'Gaming & Interactive Media'
      )
      expect(normalizeUseCase('VR content creation')).toBe(
        'Gaming & Interactive Media'
      )
      expect(normalizeUseCase('interactive media')).toBe(
        'Gaming & Interactive Media'
      )
      expect(normalizeUseCase('game textures')).toBe(
        'Gaming & Interactive Media'
      )
    })
  })

  describe('Architecture & Construction', () => {
    it('should categorize architecture use cases', () => {
      expect(normalizeUseCase('Building visualization')).toBe(
        'Architecture & Construction'
      )
      expect(normalizeUseCase('architectural rendering')).toBe(
        'Architecture & Construction'
      )
      expect(normalizeUseCase('interior design mockups')).toBe(
        'Architecture & Construction'
      )
      expect(normalizeUseCase('real estate visualization')).toBe(
        'Architecture & Construction'
      )
    })
  })

  describe('Photography & Image Processing', () => {
    it('should categorize photography work', () => {
      expect(normalizeUseCase('Product photography')).toBe(
        'Photography & Image Processing'
      )
      expect(normalizeUseCase('photo editing')).toBe(
        'Photography & Image Processing'
      )
      expect(normalizeUseCase('image enhancement')).toBe(
        'Photography & Image Processing'
      )
      expect(normalizeUseCase('portrait photography')).toBe(
        'Photography & Image Processing'
      )
    })
  })

  describe('Research & Development', () => {
    it('should categorize research work', () => {
      expect(normalizeUseCase('Scientific visualization')).toBe(
        'Research & Development'
      )
      expect(normalizeUseCase('research experiments')).toBe(
        'Research & Development'
      )
      expect(normalizeUseCase('prototype testing')).toBe(
        'Research & Development'
      )
      expect(normalizeUseCase('innovation projects')).toBe(
        'Research & Development'
      )
    })
  })

  describe('Personal & Hobby', () => {
    it('should categorize personal projects', () => {
      expect(normalizeUseCase('Personal art projects')).toBe('Personal & Hobby')
      expect(normalizeUseCase('hobby work')).toBe('Personal & Hobby')
      expect(normalizeUseCase('creative exploration')).toBe('Personal & Hobby')
      expect(normalizeUseCase('fun experiments')).toBe('Personal & Hobby')
    })
  })

  describe('Film & Video Production', () => {
    it('should categorize film work', () => {
      expect(normalizeUseCase('movie production')).toBe(
        'Film & Video Production'
      )
      expect(normalizeUseCase('video editing')).toBe('Film & Video Production')
      expect(normalizeUseCase('visual effects')).toBe('Film & Video Production')
      expect(normalizeUseCase('storyboard creation')).toBe(
        'Film & Video Production'
      )
    })
  })

  describe('Education & Training', () => {
    it('should categorize educational use cases', () => {
      expect(normalizeUseCase('educational content')).toBe(
        'Education & Training'
      )
      expect(normalizeUseCase('training materials')).toBe(
        'Education & Training'
      )
      expect(normalizeUseCase('tutorial creation')).toBe('Education & Training')
      expect(normalizeUseCase('academic projects')).toBe('Education & Training')
    })
  })

  describe('Other / Undefined category', () => {
    it('should handle undefined responses', () => {
      expect(normalizeUseCase('other')).toBe('Other / Undefined')
      expect(normalizeUseCase('none')).toBe('Other / Undefined')
      expect(normalizeUseCase('undefined')).toBe('Other / Undefined')
      expect(normalizeUseCase('')).toBe('Other / Undefined')
      expect(normalizeUseCase(null)).toBe('Other / Undefined')
    })
  })

  describe('Uncategorized responses', () => {
    it('should preserve unknown use cases with prefix', () => {
      expect(normalizeUseCase('Mysterious Use Case')).toBe(
        'Uncategorized: Mysterious Use Case'
      )
    })
  })
})

describe('normalizeSurveyResponses', () => {
  it('should normalize both industry and use case', () => {
    const input = {
      industry: 'Film and television production',
      useCase: 'Creating concept art for movies',
      familiarity: 'Expert'
    }

    const result = normalizeSurveyResponses(input)

    expect(result).toEqual({
      industry: 'Film and television production',
      industry_normalized: 'Film / TV / Animation',
      industry_raw: 'Film and television production',
      useCase: 'Creating concept art for movies',
      useCase_normalized: 'Art & Illustration',
      useCase_raw: 'Creating concept art for movies',
      familiarity: 'Expert'
    })
  })

  it('should handle partial responses', () => {
    const input = {
      industry: 'Software Development',
      familiarity: 'Beginner'
    }

    const result = normalizeSurveyResponses(input)

    expect(result).toEqual({
      industry: 'Software Development',
      industry_normalized: 'Software / IT / AI',
      industry_raw: 'Software Development',
      familiarity: 'Beginner'
    })
  })

  it('should handle empty responses', () => {
    const input = {
      familiarity: 'Intermediate'
    }

    const result = normalizeSurveyResponses(input)

    expect(result).toEqual({
      familiarity: 'Intermediate'
    })
  })

  it('should handle uncategorized responses', () => {
    const input = {
      industry: 'Unknown Creative Field',
      useCase: 'Mysterious Use Case'
    }

    const result = normalizeSurveyResponses(input)

    expect(result).toEqual({
      industry: 'Unknown Creative Field',
      industry_normalized: 'Uncategorized: Unknown Creative Field',
      industry_raw: 'Unknown Creative Field',
      useCase: 'Mysterious Use Case',
      useCase_normalized: 'Uncategorized: Mysterious Use Case',
      useCase_raw: 'Mysterious Use Case'
    })
  })

  describe('Migration script example data validation', () => {
    it('should correctly categorize all migration script examples', () => {
      const examples = [
        {
          input: {
            industry: 'Film and television production',
            useCase: 'Creating concept art for movies'
          },
          expected: {
            industry: 'Film / TV / Animation',
            useCase: 'Art & Illustration'
          }
        },
        {
          input: {
            industry: 'Marketing & Social Media',
            useCase: 'YouTube thumbnail generation'
          },
          expected: {
            industry: 'Marketing / Advertising / Social Media',
            useCase: 'Content Creation & Marketing'
          }
        },
        {
          input: {
            industry: 'Software Development',
            useCase: 'Product mockup creation'
          },
          expected: {
            industry: 'Software / IT / AI',
            useCase: 'Product Visualization & Design'
          }
        },
        {
          input: {
            industry: 'Indie Game Studio',
            useCase: 'Game asset generation'
          },
          expected: {
            industry: 'Gaming / Interactive Media',
            useCase: 'Gaming & Interactive Media'
          }
        },
        {
          input: {
            industry: 'Architecture firm',
            useCase: 'Building visualization'
          },
          expected: {
            industry: 'Architecture / Engineering / Construction',
            useCase: 'Architecture & Construction'
          }
        },
        {
          input: {
            industry: 'Custom Jewelry Design',
            useCase: 'Product photography'
          },
          expected: {
            industry: 'Fashion / Beauty / Retail',
            useCase: 'Photography & Image Processing'
          }
        },
        {
          input: {
            industry: 'Medical Research',
            useCase: 'Scientific visualization'
          },
          expected: {
            industry: 'Healthcare / Medical / Life Science',
            useCase: 'Research & Development'
          }
        },
        {
          input: {
            industry: 'Unknown Creative Field',
            useCase: 'Personal art projects'
          },
          expected: {
            industry: 'Uncategorized: Unknown Creative Field',
            useCase: 'Personal & Hobby'
          }
        }
      ]

      examples.forEach(({ input, expected }) => {
        const result = normalizeSurveyResponses(input)
        expect(result.industry_normalized).toBe(expected.industry)
        expect(result.useCase_normalized).toBe(expected.useCase)
      })
    })
  })
})
