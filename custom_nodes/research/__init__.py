"""Research Workbench custom nodes for ComfyUI."""
from typing import List
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


class ResearchExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> List[type[io.ComfyNode]]:
        from custom_nodes.research.paper_search import PaperSearch
        from custom_nodes.research.claim_extract import PaperClaimExtract
        from custom_nodes.research.evidence_assemble import ClaimEvidenceAssemble
        from custom_nodes.research.style_profile import StyleProfileExtract
        from custom_nodes.research.reference_paper_select import ReferencePaperSelect
        from custom_nodes.research.section_plan import SectionPlan
        from custom_nodes.research.abstract_draft import AbstractDraft
        from custom_nodes.research.introduction_draft import IntroductionDraft
        return [PaperSearch, PaperClaimExtract, ClaimEvidenceAssemble, StyleProfileExtract, ReferencePaperSelect, SectionPlan, AbstractDraft, IntroductionDraft]


async def comfy_entrypoint() -> ComfyExtension:
    return ResearchExtension()
