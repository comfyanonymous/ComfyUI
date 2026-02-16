import { docsSchema } from '@astrojs/starlight/schema';
import { defineCollection, z } from 'astro:content';

export const collections = {
	docs: defineCollection({
		schema: docsSchema({
			extend: z.object({
				downloads: z.number().default(0),
				rating: z.number().default(0),
				tips: z.number().default(0),
				score: z.string().default('-'),
				type: z.string().default('LoRA'),
				trigger: z.string().default('None'),
				strength: z.string().default('1.0'),
				civitaiUrl: z.string().default(''),
				compatibility: z.string().default('FLUX'),
			}),
		}),
	}),
};
