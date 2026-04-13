# Changelog

All notable changes to this project will be documented in this file.

## [0.0.0.1] - 2026-04-13

### Added
- Research Workbench: complete academic research pipeline integrated into ComfyUI
- Paper Search node: Semantic Scholar API integration for academic paper discovery
- Paper Claim Extract node: claim extraction from research papers
- Claim-Evidence Assemble node: assemble evidence for research claims
- Style Profile Extract node: writing style profile extraction from reference papers
- Reference Paper Select node: reference paper selection for projects
- Section Plan node: manuscript section planning
- Abstract Draft node: abstract generation with LLM
- Introduction Draft node: introduction section generation
- Methods Draft node: methods section generation
- Consistency Check node: verify claim consistency across claims and abstract
- Export Manuscript node: export manuscript to formatted document
- Review Import node: import peer review documents
- Review Atomize node: atomize review into individual comments
- Review Classify node: classify review comments by type and severity
- Review Map node: map review comments to manuscript sections
- Evidence Gap Detect node: detect evidence gaps in manuscript
- Action Route node: route actions based on manuscript state
- Evidence Pack Assemble node: assemble evidence packs for claims
- Evidence Graph Create node: create evidence citation graph
- Evidence Trace node: trace evidence provenance
- Response Draft node: draft peer review responses
- Tone Control node: control manuscript tone
- Coverage Check node: check manuscript coverage of claims
- Revision Plan node: plan manuscript revisions
- Paper Fetch node: fetch papers from arXiv and other sources
- Feed Update node: update research feed with new papers
- Feed Discovery: automatic paper discovery from journal APIs (Semantic Scholar, CrossRef)
- Research API: FastAPI-style REST API for research workbench (projects, papers, claims, sources, feed, styles)
- Research Workbench Web UI: sidebar panels for workflow and assets
- SQLAlchemy models: Project, Intent, PaperAsset, ClaimAsset, Source, FeedItem, StyleAsset
- Database indexes on frequently queried columns
- Automation scheduler for periodic feed discovery jobs
- ComfyUI custom node extension integration
