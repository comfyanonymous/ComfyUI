// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	outDir: 'loras',
	vite: {
		resolve: {
			alias: {
				'@': '/src/',
			},
		},
	},
	integrations: [
		starlight({
			title: 'FLUX LoRA Documentation',
			description: 'Comprehensive documentation for FLUX LoRA models with preview images and statistics',
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com' }
			],
			customCss: [
				'./src/styles/custom.css',
			],
			sidebar: [
				{
					label: '🏗️ Checkpoints',
					collapsed: false,
					items: [
						{ label: 'All Checkpoints', link: '/checkpoints/' },
						{ label: 'Fluxed Up NSFW v7.1', link: '/checkpoints/fluxed_up_nsfw_fp16/' },
					],
				},
				{
					label: '🎨 Characters',
					collapsed: false,
					items: [
						{ label: 'Acorn Collection', link: '/characters/acorn/' },
						{ label: 'Other Characters', link: '/characters/other/' },
						{ label: 'Women Around The World', link: '/characters/watw/' },
					],
				},
				{
					label: '🔬 Anatomy',
					collapsed: true,
					items: [
						{ label: 'Breasts & Nipples', link: '/anatomy/breasts/' },
						{ label: 'Pussy & Vagina', link: '/anatomy/pussy/' },
						{ label: 'Ass & Butt', link: '/anatomy/ass/' },
					],
				},
				{
					label: '🧘 Poses & Angles',
					collapsed: true,
					items: [
						{ label: 'Overview', link: '/poses/' },
						{ label: 'Doggystyle & From Behind', link: '/poses/doggystyle-behind/' },
						{ label: 'Anal Positions', link: '/poses/anal/' },
						{ label: 'Sex Positions', link: '/poses/sex-positions/' },
						{ label: 'Oral & Tongue', link: '/poses/oral-tongue/' },
						{ label: 'Sexy Poses', link: '/poses/sexy-poses/' },
						{ label: 'Clothing & Flashing', link: '/poses/clothing-flashing/' },
						{ label: 'Facesitting & Misc', link: '/poses/facesitting-misc/' },
					],
				},
				{
					label: '👗 Clothing & Fashion',
					collapsed: true,
					items: [
						{ label: 'Overview', link: '/clothing/' },
						{ label: 'Panties & Underwear', link: '/clothing/panties-underwear/' },
						{ label: 'Skirts & Upskirt', link: '/clothing/skirts-upskirt/' },
						{ label: 'Pantyhose & Stockings', link: '/clothing/pantyhose-stockings/' },
						{ label: 'Lingerie & Sheer', link: '/clothing/lingerie-sheer/' },
						{ label: 'Shoes & Footwear', link: '/clothing/shoes-footwear/' },
						{ label: 'Tops, Dresses & Other', link: '/clothing/tops-dresses-other/' },
					],
				},
				{
					label: '💪 Body Types',
					collapsed: true,
					items: [
						{ label: 'Overview', link: '/body-types/' },
						{ label: 'General Anatomy', link: '/body-types/general/' },
						{ label: 'Body Shape', link: '/body-types/shape/' },
						{ label: 'Special', link: '/body-types/special/' },
					],
				},
				{
					label: '✨ Style & Enhancement',
					collapsed: true,
					items: [
						{ label: 'Overview', link: '/style/' },
						{ label: 'Detail Enhancers', link: '/style/detail/' },
						{ label: 'NSFW Styles', link: '/style/nsfw/' },
						{ label: 'Aesthetic & Themes', link: '/style/aesthetic/' },
						{ label: 'Body Styles', link: '/style/body/' },
					],
				},
			],
			components: {
				// Override default components if needed
			},
		}),
	],
});
