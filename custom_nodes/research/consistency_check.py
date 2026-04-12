"""ConsistencyCheck node - check manuscript consistency."""
import json
import re
from typing_extensions import override
from comfy_api.latest import ComfyNode, io


class ConsistencyCheck(io.ComfyNode):
    """Check consistency between claims, text, numbers, and figures in manuscript."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ConsistencyCheck",
            display_name="Check Consistency",
            category="Research",
            inputs=[
                io.String.Input(
                    "claims",
                    display_name="Claims (JSON)",
                    default="[]",
                    multiline=True,
                ),
                io.String.Input(
                    "abstract",
                    display_name="Abstract Text",
                    default="",
                    multiline=True,
                ),
                io.String.Input(
                    "methods",
                    display_name="Methods Text",
                    default="",
                    multiline=True,
                ),
                io.String.Input(
                    "results",
                    display_name="Results Text",
                    default="",
                    multiline=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="Issues List (JSON)"),
            ],
        )

    @classmethod
    def execute(cls, claims: str, abstract: str, methods: str, results: str) -> io.NodeOutput:
        try:
            claims_list = json.loads(claims) if claims else []
        except json.JSONDecodeError:
            claims_list = []

        issues = []

        # Check 1: Claims mentioned in abstract should appear in results
        for claim in claims_list:
            claim_text = claim.get("text", "").lower()
            claim_keywords = [w for w in claim_text.split() if len(w) > 4][:3]

            if claim_keywords:
                results_lower = results.lower()
                keyword_matches = sum(1 for kw in claim_keywords if kw in results_lower)

                if keyword_matches < len(claim_keywords) / 2:
                    issues.append({
                        "severity": "warning",
                        "type": "claim_not_in_results",
                        "message": f"Claim may not be supported in results: {claim.get('text', '')[:50]}...",
                        "claim_id": claim.get("id", "unknown"),
                    })

        # Check 2: Numbers should be consistent across sections
        full_text = f"{abstract} {methods} {results}"
        numbers_found = re.findall(r'\b(\d+\.?\d*)%?\b', full_text)

        if len(numbers_found) > 10:
            issues.append({
                "severity": "info",
                "type": "many_numbers",
                "message": f"Found {len(numbers_found)} numbers in text. Verify all are accurate.",
            })

        # Check 3: Abstract length
        abstract_words = len(abstract.split())
        if abstract_words > 300:
            issues.append({
                "severity": "warning",
                "type": "abstract_too_long",
                "message": f"Abstract is {abstract_words} words. Most venues prefer 150-250 words.",
            })
        elif abstract_words < 100:
            issues.append({
                "severity": "warning",
                "type": "abstract_too_short",
                "message": f"Abstract is only {abstract_words} words. Consider adding more detail.",
            })

        # Check 4: Claims without support level
        unsupported_claims = [c for c in claims_list if c.get("support_level") == "unsupported"]
        if len(unsupported_claims) > 3:
            issues.append({
                "severity": "error",
                "type": "many_unsupported_claims",
                "message": f"{len(unsupported_claims)} claims lack evidence support. Add experiments or citations.",
            })

        # Check 5: Section references
        section_refs = re.findall(r'Section (\d+)', full_text)
        if section_refs:
            max_section = max(int(s) for s in section_refs if s.isdigit())
            if max_section > 5:
                issues.append({
                    "severity": "info",
                    "type": "section_references",
                    "message": f"References to Section {max_section} found. Verify all referenced sections exist.",
                })

        report = {
            "total_issues": len(issues),
            "errors": len([i for i in issues if i.get("severity") == "error"]),
            "warnings": len([i for i in issues if i.get("severity") == "warning"]),
            "info": len([i for i in issues if i.get("severity") == "info"]),
            "issues": issues,
        }

        return io.NodeOutput(issues_list=json.dumps(report, indent=2))
