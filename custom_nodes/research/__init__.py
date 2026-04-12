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
        from custom_nodes.research.methods_draft import MethodsDraft
        from custom_nodes.research.consistency_check import ConsistencyCheck
        from custom_nodes.research.export_manuscript import ExportManuscript
        from custom_nodes.research.review_import import ReviewImport
        from custom_nodes.research.review_atomize import ReviewAtomize
        from custom_nodes.research.review_classify import ReviewClassify
        from custom_nodes.research.review_map import ReviewMap
        from custom_nodes.research.evidence_gap_detect import EvidenceGapDetect
        from custom_nodes.research.action_route import ActionRoute
        from custom_nodes.research.evidence_pack_assemble import EvidencePackAssemble
        from custom_nodes.research.response_draft import ResponseDraft
        from custom_nodes.research.tone_control import ToneControl
        from custom_nodes.research.coverage_check import CoverageCheck
        from custom_nodes.research.revision_plan import RevisionPlan
        return [PaperSearch, PaperClaimExtract, ClaimEvidenceAssemble, StyleProfileExtract, ReferencePaperSelect, SectionPlan, AbstractDraft, IntroductionDraft, MethodsDraft, ConsistencyCheck, ExportManuscript, ReviewImport, ReviewAtomize, ReviewClassify, ReviewMap, EvidenceGapDetect, ActionRoute, EvidencePackAssemble, ResponseDraft, ToneControl, CoverageCheck, RevisionPlan]


async def comfy_entrypoint() -> ComfyExtension:
    return ResearchExtension()
