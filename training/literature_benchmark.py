"""Literature-grounded experiment benchmark utilities.

This module lets the environment run a paper-backed experiment plan, then
compare the resulting simulated findings against curated expected findings
from the literature.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Optional, Sequence

from models import (
    ActionType,
    ConclusionClaim,
    ExperimentAction,
    ExperimentObservation,
    OutputType,
    TaskSpec,
)
from server.hackathon_environment import BioExperimentEnvironment
from server.tasks.scenarios import SCENARIO_LIBRARY, Scenario

TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "using",
    "with",
}

BIO_LIBRARY_DISTRIBUTIONS = {
    "scanpy": "scanpy",
    "gseapy": "gseapy",
    "biopython": "biopython",
}


@dataclass
class PaperBenchmarkResult:
    scenario_name: str
    problem_statement: str
    matched_papers: List[str]
    bio_library_versions: Dict[str, Optional[str]]
    matched_findings: List[str] = field(default_factory=list)
    missed_findings: List[str] = field(default_factory=list)
    discovered_markers: List[str] = field(default_factory=list)
    candidate_mechanisms: List[str] = field(default_factory=list)
    conclusions: List[str] = field(default_factory=list)
    final_reward: float = 0.0
    total_steps: int = 0

    @property
    def match_ratio(self) -> float:
        total = len(self.matched_findings) + len(self.missed_findings)
        return len(self.matched_findings) / max(total, 1)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["match_ratio"] = self.match_ratio
        return data


def detect_bio_library_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for name, dist_name in BIO_LIBRARY_DISTRIBUTIONS.items():
        try:
            versions[name] = version(dist_name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def select_literature_scenario(problem_statement: str) -> Scenario:
    """Pick the closest literature-backed scenario for a prompt."""

    prompt_tokens = set(_tokenize(problem_statement))
    best_score = -1
    best_scenario: Optional[Scenario] = None

    for scenario in SCENARIO_LIBRARY:
        if not scenario.task.paper_references:
            continue
        corpus = [
            scenario.task.problem_statement,
            *(ref.title for ref in scenario.task.paper_references),
            *(finding.finding for finding in scenario.task.expected_findings),
            scenario.task.tissue,
            scenario.task.modality,
            *scenario.task.conditions,
        ]
        score = len(prompt_tokens & set(_tokenize(" ".join(corpus))))
        if scenario.task.problem_statement.lower() in problem_statement.lower():
            score += 4
        if score > best_score:
            best_score = score
            best_scenario = scenario

    if best_scenario is None:
        raise ValueError("No literature-backed scenarios are available.")
    return best_scenario


def run_paper_benchmark(
    *,
    problem_statement: str,
    scenario_name: Optional[str] = None,
    domain_randomise: bool = False,
) -> PaperBenchmarkResult:
    """Run a literature-backed episode and compare outputs to paper results."""

    scenario = _resolve_scenario(problem_statement, scenario_name)
    env = BioExperimentEnvironment(
        scenario_name=scenario.name,
        domain_randomise=domain_randomise,
    )
    obs = env.reset()

    for action in build_paper_aligned_actions(obs.task):
        obs = env.step(action)

    claims = infer_conclusion_claims(obs)
    obs = env.step(
        ExperimentAction(
            action_type=ActionType.SYNTHESIZE_CONCLUSION,
            parameters={"claims": [claim.model_dump() for claim in claims]},
            justification=(
                "Summarize the simulated experimental evidence and compare it "
                "with the paper-backed expected findings."
            ),
            confidence=0.8,
            tool_call_spec=_tool_context(
                obs.task,
                libraries=["biopython"],
            ),
        )
    )

    matched, missed = compare_expected_findings(obs.task, obs)
    return PaperBenchmarkResult(
        scenario_name=scenario.name,
        problem_statement=obs.task.problem_statement,
        matched_papers=[ref.title for ref in obs.task.paper_references],
        bio_library_versions=detect_bio_library_versions(),
        matched_findings=matched,
        missed_findings=missed,
        discovered_markers=list(obs.discovered_markers),
        candidate_mechanisms=list(obs.candidate_mechanisms),
        conclusions=[c.claim for c in obs.conclusions],
        final_reward=float(obs.metadata.get("cumulative_reward", 0.0)),
        total_steps=obs.step_index,
    )


def build_paper_aligned_actions(task: TaskSpec) -> List[ExperimentAction]:
    """Construct a pragmatic analysis plan aligned to the task modality."""

    actions: List[ExperimentAction] = [
        ExperimentAction(
            action_type=ActionType.COLLECT_SAMPLE,
            parameters={"n_samples": 8},
            justification="Collect enough samples to support downstream analysis.",
            confidence=0.75,
            tool_call_spec=_tool_context(task, libraries=["biopython"]),
        ),
        ExperimentAction(
            action_type=ActionType.PREPARE_LIBRARY,
            method="10x_chromium",
            justification="Use a standard single-cell library prep workflow.",
            confidence=0.8,
            tool_call_spec=_tool_context(task, libraries=["scanpy"]),
        ),
        ExperimentAction(
            action_type=ActionType.SEQUENCE_CELLS,
            method="NovaSeq",
            justification="Generate sufficient single-cell read depth.",
            confidence=0.8,
            tool_call_spec=_tool_context(task, libraries=["scanpy"]),
        ),
        ExperimentAction(
            action_type=ActionType.RUN_QC,
            method="scanpy.pp.calculate_qc_metrics",
            justification="Check technical quality before downstream inference.",
            confidence=0.85,
            tool_call_spec=_tool_context(task, libraries=["scanpy"]),
        ),
        ExperimentAction(
            action_type=ActionType.FILTER_DATA,
            method="scanpy.pp.filter_cells",
            justification="Remove low-quality cells and reduce technical noise.",
            confidence=0.85,
            tool_call_spec=_tool_context(task, libraries=["scanpy"]),
        ),
        ExperimentAction(
            action_type=ActionType.NORMALIZE_DATA,
            method="scanpy.pp.normalize_total",
            justification="Normalize expression to prepare comparable profiles.",
            confidence=0.85,
            tool_call_spec=_tool_context(task, libraries=["scanpy"]),
        ),
        ExperimentAction(
            action_type=ActionType.CLUSTER_CELLS,
            method="scanpy.tl.leiden",
            justification="Resolve cell states before focused interpretation.",
            confidence=0.8,
            tool_call_spec=_tool_context(task, libraries=["scanpy"]),
        ),
    ]

    categories = {finding.category for finding in task.expected_findings}
    if "trajectory" in categories:
        actions.extend([
            ExperimentAction(
                action_type=ActionType.TRAJECTORY_ANALYSIS,
                method="scanpy.tl.dpt",
                justification="Recover pseudotime structure and lineage branches.",
                confidence=0.8,
                tool_call_spec=_tool_context(task, libraries=["scanpy"]),
            ),
            ExperimentAction(
                action_type=ActionType.REGULATORY_NETWORK_INFERENCE,
                method="pySCENIC",
                justification="Infer branch-associated regulators from the trajectory.",
                confidence=0.75,
                tool_call_spec=_tool_context(task, libraries=["scanpy"]),
            ),
            ExperimentAction(
                action_type=ActionType.MARKER_SELECTION,
                method="scanpy.tl.rank_genes_groups",
                justification="Summarize lineage markers and branch-state genes.",
                confidence=0.75,
                tool_call_spec=_tool_context(task, libraries=["scanpy"]),
            ),
        ])
        return actions

    actions.extend([
        ExperimentAction(
            action_type=ActionType.DIFFERENTIAL_EXPRESSION,
            method="scanpy.tl.rank_genes_groups",
            parameters={"comparison": _default_comparison_name(task)},
            justification="Identify genes associated with the focal phenotype.",
            confidence=0.85,
            tool_call_spec=_tool_context(task, libraries=["scanpy"]),
        ),
        ExperimentAction(
            action_type=ActionType.PATHWAY_ENRICHMENT,
            method="gseapy.prerank",
            justification="Translate DE hits into pathway-level interpretation.",
            confidence=0.8,
            tool_call_spec=_tool_context(task, libraries=["gseapy"]),
        ),
        ExperimentAction(
            action_type=ActionType.MARKER_SELECTION,
            method="scanpy.tl.rank_genes_groups",
            justification="Nominate candidate markers for follow-up validation.",
            confidence=0.8,
            tool_call_spec=_tool_context(task, libraries=["scanpy"]),
        ),
        ExperimentAction(
            action_type=ActionType.VALIDATE_MARKER,
            method="immunofluorescence",
            parameters={"marker": _preferred_marker(task)},
            justification="Check whether the leading marker reproduces in validation.",
            confidence=0.75,
            tool_call_spec=_tool_context(task, libraries=["biopython"]),
        ),
    ])
    return actions


def infer_conclusion_claims(obs: ExperimentObservation) -> List[ConclusionClaim]:
    """Turn accumulated evidence into concise, paper-comparable claims."""

    markers = set(obs.discovered_markers)
    mechanisms = set(obs.candidate_mechanisms)
    network_regulators = set(_extract_network_regulators(obs))
    trajectory_output = _latest_output_data(obs, OutputType.TRAJECTORY_RESULT)

    claims: List[ConclusionClaim] = []

    if "SPP1" in markers:
        claims.append(ConclusionClaim(
            claim="SPP1-positive macrophages are enriched in IPF fibrotic tissue.",
            confidence=0.84,
            claim_type="marker",
            evidence_steps=_evidence_steps(obs, {
                OutputType.DE_RESULT,
                OutputType.MARKER_RESULT,
                OutputType.VALIDATION_RESULT,
            }),
        ))
    if {"SPP1", "MERTK"} <= markers:
        claims.append(ConclusionClaim(
            claim="MERTK co-occurs with the SPP1-positive profibrotic macrophage state.",
            confidence=0.8,
            claim_type="marker",
            evidence_steps=_evidence_steps(obs, {
                OutputType.DE_RESULT,
                OutputType.MARKER_RESULT,
            }),
        ))
    if "extracellular_matrix_organisation" in mechanisms:
        claims.append(ConclusionClaim(
            claim=(
                "Extracellular matrix organization is a dominant fibrotic "
                "program in the IPF samples."
            ),
            confidence=0.78,
            claim_type="pathway",
            evidence_steps=_evidence_steps(obs, {OutputType.PATHWAY_RESULT}),
        ))

    if trajectory_output.get("branching_detected"):
        claims.append(ConclusionClaim(
            claim=(
                "Trajectory analysis recovered branching blood lineages rooted "
                "in HSCs."
            ),
            confidence=0.82,
            claim_type="trajectory",
            evidence_steps=_evidence_steps(obs, {OutputType.TRAJECTORY_RESULT}),
        ))
    if "GATA1" in network_regulators:
        claims.append(ConclusionClaim(
            claim="GATA1 emerges as a driver of erythroid fate commitment.",
            confidence=0.8,
            claim_type="regulatory_network",
            evidence_steps=_evidence_steps(obs, {OutputType.NETWORK_RESULT}),
        ))
    if {"CEBPA", "SPI1"} & network_regulators:
        claims.append(ConclusionClaim(
            claim="CEBPA and SPI1 support myeloid branch decisions.",
            confidence=0.78,
            claim_type="regulatory_network",
            evidence_steps=_evidence_steps(obs, {OutputType.NETWORK_RESULT}),
        ))

    return claims


def compare_expected_findings(
    task: TaskSpec,
    obs: ExperimentObservation,
) -> tuple[List[str], List[str]]:
    """Compare the episode evidence against literature-backed findings."""

    evidence_text = _evidence_text(obs)
    matched: List[str] = []
    missed: List[str] = []

    for finding in task.expected_findings:
        keywords = [kw.lower() for kw in finding.keywords]
        if not keywords:
            keywords = _tokenize(finding.finding)
        hits = sum(1 for kw in keywords if kw in evidence_text)
        threshold = max(1, (len(keywords) + 1) // 2)
        if hits >= threshold:
            matched.append(finding.finding)
        else:
            missed.append(finding.finding)

    return matched, missed


def _resolve_scenario(
    problem_statement: str,
    scenario_name: Optional[str],
) -> Scenario:
    if scenario_name:
        for scenario in SCENARIO_LIBRARY:
            if scenario.name == scenario_name:
                return scenario
        raise ValueError(f"Unknown scenario_name '{scenario_name}'.")
    return select_literature_scenario(problem_statement)


def _tool_context(
    task: TaskSpec,
    *,
    libraries: Sequence[str],
    include_expected_findings: bool = False,
) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "literature_query": task.problem_statement,
        "paper_references": [
            {
                "title": ref.title,
                "doi": ref.doi,
                "pmid": ref.pmid,
                "url": ref.url,
            }
            for ref in task.paper_references
        ],
        "bioinformatics_libraries": list(libraries),
    }
    if include_expected_findings:
        context["expected_findings"] = [
            finding.finding for finding in task.expected_findings
        ]
    return context


def _default_comparison_name(task: TaskSpec) -> str:
    conditions = {condition.lower() for condition in task.conditions}
    if {"healthy", "ipf"} <= conditions:
        return "IPF_vs_healthy"
    if any("treated" in condition for condition in conditions) and any(
        "untreated" in condition for condition in conditions
    ):
        return "treated_vs_untreated"
    if any("healthy" in condition for condition in conditions):
        return "disease_vs_healthy"
    return "disease_vs_healthy"


def _preferred_marker(task: TaskSpec) -> str:
    """Derive a candidate marker from the problem statement, not expected findings."""
    tokens = [t for t in TOKEN_RE.findall(task.problem_statement) if t.isupper() and len(t) >= 3]
    if tokens:
        return tokens[0]
    return "unknown"


def _latest_output_data(
    obs: ExperimentObservation,
    output_type: OutputType,
) -> Dict[str, Any]:
    for output in reversed(obs.all_outputs):
        if output.output_type == output_type:
            return output.data
    return {}


def _extract_network_regulators(obs: ExperimentObservation) -> List[str]:
    for output in reversed(obs.all_outputs):
        if output.output_type == OutputType.NETWORK_RESULT:
            return output.data.get("top_regulators", [])
    return []


def _evidence_steps(
    obs: ExperimentObservation,
    output_types: set[OutputType],
) -> List[int]:
    return [
        output.step_index
        for output in obs.all_outputs
        if output.output_type in output_types
    ]


def _evidence_text(obs: ExperimentObservation) -> str:
    parts: List[str] = []
    parts.extend(obs.discovered_markers)
    parts.extend(obs.candidate_mechanisms)
    parts.extend(conclusion.claim for conclusion in obs.conclusions)

    for output in obs.all_outputs:
        parts.append(output.summary)
        if output.output_type == OutputType.DE_RESULT:
            parts.extend(
                gene["gene"]
                for gene in output.data.get("top_genes", [])
                if isinstance(gene, dict) and "gene" in gene
            )
        elif output.output_type == OutputType.PATHWAY_RESULT:
            parts.extend(
                pathway["pathway"]
                for pathway in output.data.get("top_pathways", [])
                if isinstance(pathway, dict) and "pathway" in pathway
            )
        elif output.output_type == OutputType.NETWORK_RESULT:
            parts.extend(output.data.get("top_regulators", []))
        elif output.output_type == OutputType.TRAJECTORY_RESULT:
            if output.data.get("branching_detected"):
                parts.append("branching lineage HSC trajectory")

    return " ".join(parts).lower()


def _tokenize(text: str) -> List[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if token and token.lower() not in STOPWORDS
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem-statement",
        default=(
            "Design a follow-up validation experiment for candidate biomarker "
            "SPP1 in idiopathic pulmonary fibrosis."
        ),
    )
    parser.add_argument("--scenario-name", default=None)
    parser.add_argument("--domain-randomise", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = run_paper_benchmark(
        problem_statement=args.problem_statement,
        scenario_name=args.scenario_name,
        domain_randomise=args.domain_randomise,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print(f"Scenario: {result.scenario_name}")
    print(f"Problem: {result.problem_statement}")
    print(f"Paper: {', '.join(result.matched_papers)}")
    print(f"Match ratio: {result.match_ratio:.2%}")
    print(f"Matched findings: {len(result.matched_findings)}")
    print(f"Missed findings: {len(result.missed_findings)}")
    print(f"Discovered markers: {', '.join(result.discovered_markers[:8])}")
    print(f"Candidate mechanisms: {', '.join(result.candidate_mechanisms[:5])}")
    print(f"Conclusions: {len(result.conclusions)}")
    print(f"Final reward: {result.final_reward:+.3f}")
    print(f"Bio libraries: {json.dumps(result.bio_library_versions, sort_keys=True)}")


if __name__ == "__main__":
    main()
