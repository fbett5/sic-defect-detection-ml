"""Map a predicted wafer-map pattern to likely process root causes.

This is a rule-based starting point drawn from common wafer-map failure
signatures. Treat it as a hypothesis generator for the engineer, not a
verdict: confirm with process data (tool history, lot genealogy, inline
metrology) before acting. Edit the table as you learn what applies to
your fab or to SiC specifically.
"""
from __future__ import annotations

ROOT_CAUSES: dict[str, list[str]] = {
    "none": [],
    "Center": [
        "Non-uniform deposition or etch rate (centre-fast/centre-slow chamber)",
        "CMP pressure profile issue at wafer centre",
        "Spin-coat / develop non-uniformity at centre",
    ],
    "Donut": [
        "Annular temperature non-uniformity (chuck heater zones)",
        "CMP retaining-ring or zone-pressure issue",
        "Gas-flow ring pattern in deposition/etch chamber",
    ],
    "Edge-Ring": [
        "Edge-bead removal or edge exposure problems",
        "Etch / deposition non-uniformity at wafer edge",
        "Edge temperature roll-off during anneal",
    ],
    "Edge-Loc": [
        "Wafer handling damage (robot end-effector, cassette slot)",
        "Clamp-ring or edge-grip contact",
        "Localised edge contamination",
    ],
    "Loc": [
        "Particle or contamination cluster",
        "Localised lithography defect (reticle, focus spot)",
        "Chuck contamination causing local focus error",
    ],
    "Random": [
        "Background particle level (cleanroom / tool cleanliness)",
        "Material defect density (for SiC: substrate/epi crystal defects such as BPDs, micropipes)",
        "Random process noise; check tool PM status",
    ],
    "Scratch": [
        "Mechanical handling scratch (transfer arms, tweezers)",
        "CMP scratch from slurry agglomerates or pad debris",
    ],
    "Near-full": [
        "Gross process excursion (wrong recipe, tool fault)",
        "Mis-processed or mis-routed wafer",
        "Test / probe-card setup fault — verify before blaming the process",
    ],
}


def likely_causes(pattern: str) -> list[str]:
    return ROOT_CAUSES.get(pattern, ["Unknown pattern — no mapping defined"])
