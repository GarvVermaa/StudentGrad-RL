"""Curated biological building blocks for procedural scenario generation.

Provides tissue-specific cell types, disease profiles, pathway libraries,
regulatory network templates, and perturbation effect profiles.  The
procedural generator composes these into complete ``Scenario`` objects
with fully populated ``LatentBiologicalState``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Cell type templates ─────────────────────────────────────────────────────


@dataclass
class CellTypeTemplate:
    name: str
    marker_genes: List[str]
    proportion_range: Tuple[float, float] = (0.05, 0.30)
    states: List[str] = field(default_factory=lambda: ["quiescent"])
    disease_responsive: bool = False
    response_range: Tuple[float, float] = (0.5, 1.5)


TISSUE_CELL_TYPES: Dict[str, List[CellTypeTemplate]] = {
    "heart": [
        CellTypeTemplate("cardiomyocyte", ["TNNT2", "MYH7", "ACTC1"], (0.25, 0.40), ["contractile", "stressed"]),
        CellTypeTemplate("cardiac_fibroblast", ["COL1A1", "DCN", "LUM"], (0.15, 0.30), ["quiescent", "activated"], True, (1.1, 1.8)),
        CellTypeTemplate("endothelial", ["PECAM1", "VWF", "CDH5"], (0.10, 0.20), ["quiescent"]),
        CellTypeTemplate("macrophage", ["CD68", "CD163", "CSF1R"], (0.05, 0.15), ["quiescent", "activated", "inflammatory"], True, (1.2, 2.0)),
        CellTypeTemplate("smooth_muscle", ["ACTA2", "MYH11", "TAGLN"], (0.08, 0.18), ["quiescent"]),
        CellTypeTemplate("pericyte", ["PDGFRB", "RGS5", "NOTCH3"], (0.03, 0.10), ["quiescent"]),
    ],
    "lung": [
        CellTypeTemplate("AT2", ["SFTPC", "SFTPB", "ABCA3"], (0.15, 0.25), ["normal", "stressed"]),
        CellTypeTemplate("AT1", ["AGER", "PDPN", "CAV1"], (0.10, 0.18), ["normal"]),
        CellTypeTemplate("alveolar_macrophage", ["MARCO", "FABP4", "MCEMP1"], (0.10, 0.20), ["resident", "activated"]),
        CellTypeTemplate("fibroblast", ["COL1A1", "COL3A1", "POSTN"], (0.12, 0.25), ["quiescent", "activated"], True, (1.2, 2.0)),
        CellTypeTemplate("endothelial", ["PECAM1", "CLDN5", "VWF"], (0.08, 0.15), ["quiescent"]),
        CellTypeTemplate("T_cell", ["CD3D", "CD3E", "IL7R"], (0.08, 0.18), ["quiescent", "activated"]),
        CellTypeTemplate("ciliated", ["FOXJ1", "DNAH5", "TPPP3"], (0.05, 0.12), ["normal"]),
    ],
    "brain": [
        CellTypeTemplate("excitatory_neuron", ["SLC17A7", "CAMK2A", "NRGN"], (0.25, 0.40), ["normal", "stressed"]),
        CellTypeTemplate("inhibitory_neuron", ["GAD1", "GAD2", "SLC32A1"], (0.12, 0.22), ["normal"]),
        CellTypeTemplate("astrocyte", ["GFAP", "AQP4", "SLC1A3"], (0.10, 0.20), ["quiescent", "activated"], True, (1.2, 1.8)),
        CellTypeTemplate("microglia", ["CX3CR1", "P2RY12", "TMEM119"], (0.05, 0.12), ["homeostatic", "activated", "inflammatory"], True, (1.3, 2.5)),
        CellTypeTemplate("oligodendrocyte", ["MBP", "PLP1", "MOG"], (0.10, 0.18), ["myelinating"]),
        CellTypeTemplate("OPC", ["PDGFRA", "CSPG4", "OLIG2"], (0.03, 0.08), ["progenitor"]),
        CellTypeTemplate("endothelial", ["CLDN5", "FLT1", "PECAM1"], (0.03, 0.08), ["quiescent"]),
    ],
    "liver": [
        CellTypeTemplate("hepatocyte", ["ALB", "APOB", "CYP3A4"], (0.55, 0.70), ["normal", "stressed"]),
        CellTypeTemplate("cholangiocyte", ["KRT19", "KRT7", "EPCAM"], (0.05, 0.10), ["normal"]),
        CellTypeTemplate("kupffer_cell", ["CD68", "CLEC4F", "MARCO"], (0.08, 0.15), ["quiescent", "activated", "inflammatory"], True, (1.2, 2.0)),
        CellTypeTemplate("stellate_cell", ["ACTA2", "LRAT", "PDGFRB"], (0.05, 0.12), ["quiescent", "activated"], True, (1.3, 2.0)),
        CellTypeTemplate("endothelial", ["PECAM1", "LYVE1", "STAB2"], (0.05, 0.10), ["quiescent"]),
        CellTypeTemplate("NK_cell", ["NKG7", "GNLY", "KLRD1"], (0.03, 0.08), ["quiescent", "activated"]),
    ],
    "bone_marrow": [
        CellTypeTemplate("HSC", ["CD34", "KIT", "THY1"], (0.03, 0.08), ["stem"]),
        CellTypeTemplate("CMP", ["CD34", "FLT3"], (0.08, 0.15), ["progenitor"]),
        CellTypeTemplate("GMP", ["CSF3R", "CEBPA"], (0.08, 0.15), ["progenitor"]),
        CellTypeTemplate("MEP", ["GATA1", "KLF1"], (0.06, 0.12), ["progenitor"]),
        CellTypeTemplate("erythrocyte", ["HBA1", "HBB", "GYPA"], (0.15, 0.25), ["mature"]),
        CellTypeTemplate("neutrophil", ["ELANE", "MPO", "CTSG"], (0.12, 0.22), ["mature"]),
        CellTypeTemplate("monocyte", ["CD14", "CSF1R", "FCGR3A"], (0.10, 0.18), ["mature"]),
        CellTypeTemplate("megakaryocyte", ["ITGA2B", "GP1BA", "PF4"], (0.05, 0.12), ["mature"]),
    ],
    "kidney": [
        CellTypeTemplate("proximal_tubule", ["SLC34A1", "LRP2", "CUBN"], (0.30, 0.45), ["normal", "stressed"]),
        CellTypeTemplate("distal_tubule", ["SLC12A3", "CALB1"], (0.10, 0.18), ["normal"]),
        CellTypeTemplate("collecting_duct", ["AQP2", "FXYD4"], (0.08, 0.15), ["normal"]),
        CellTypeTemplate("podocyte", ["NPHS1", "NPHS2", "WT1"], (0.05, 0.10), ["normal", "stressed"]),
        CellTypeTemplate("endothelial", ["PECAM1", "EMCN", "FLT1"], (0.05, 0.12), ["quiescent"]),
        CellTypeTemplate("macrophage", ["CD68", "CD163", "CSF1R"], (0.05, 0.10), ["quiescent", "inflammatory"], True, (1.3, 2.0)),
        CellTypeTemplate("fibroblast", ["COL1A1", "PDGFRA", "DCN"], (0.05, 0.12), ["quiescent", "activated"], True, (1.2, 1.8)),
    ],
    "colon": [
        CellTypeTemplate("colonocyte", ["CA2", "AQP8", "SLC26A3"], (0.25, 0.40), ["normal", "stressed"]),
        CellTypeTemplate("goblet_cell", ["MUC2", "TFF3", "FCGBP"], (0.10, 0.18), ["secretory"]),
        CellTypeTemplate("stem_cell", ["LGR5", "ASCL2", "OLFM4"], (0.05, 0.10), ["stem"]),
        CellTypeTemplate("T_cell", ["CD3D", "CD3E", "IL7R"], (0.10, 0.18), ["quiescent", "activated"]),
        CellTypeTemplate("macrophage", ["CD68", "CD163", "CSF1R"], (0.05, 0.12), ["quiescent", "inflammatory"], True, (1.3, 2.0)),
        CellTypeTemplate("fibroblast", ["COL1A1", "COL3A1", "VIM"], (0.08, 0.15), ["quiescent", "activated"], True, (1.2, 1.8)),
        CellTypeTemplate("endothelial", ["PECAM1", "VWF", "CDH5"], (0.05, 0.10), ["quiescent"]),
    ],
    "pancreas": [
        CellTypeTemplate("beta_cell", ["INS", "MAFA", "NKX6-1"], (0.25, 0.40), ["normal", "stressed"], True, (0.4, 0.8)),
        CellTypeTemplate("alpha_cell", ["GCG", "ARX", "IRX2"], (0.15, 0.25), ["normal"]),
        CellTypeTemplate("delta_cell", ["SST", "HHEX"], (0.05, 0.10), ["normal"]),
        CellTypeTemplate("ductal", ["KRT19", "SOX9", "CFTR"], (0.10, 0.18), ["normal"]),
        CellTypeTemplate("acinar", ["PRSS1", "CPA1", "CELA3A"], (0.10, 0.20), ["normal"]),
        CellTypeTemplate("stellate", ["ACTA2", "PDGFRA", "COL1A1"], (0.05, 0.10), ["quiescent", "activated"], True, (1.2, 1.8)),
        CellTypeTemplate("macrophage", ["CD68", "CD163"], (0.03, 0.08), ["quiescent", "inflammatory"]),
    ],
    "skin": [
        CellTypeTemplate("keratinocyte", ["KRT14", "KRT5", "KRT1"], (0.40, 0.55), ["basal", "differentiated"]),
        CellTypeTemplate("melanocyte", ["MLANA", "PMEL", "TYR"], (0.05, 0.10), ["normal", "activated"]),
        CellTypeTemplate("fibroblast", ["COL1A1", "COL3A1", "DCN"], (0.10, 0.20), ["quiescent", "activated"]),
        CellTypeTemplate("T_cell", ["CD3D", "CD3E", "IL7R"], (0.08, 0.15), ["quiescent", "activated"]),
        CellTypeTemplate("macrophage", ["CD68", "CD163", "CSF1R"], (0.05, 0.10), ["quiescent", "inflammatory"]),
        CellTypeTemplate("endothelial", ["PECAM1", "VWF"], (0.05, 0.10), ["quiescent"]),
    ],
    "breast": [
        CellTypeTemplate("luminal_epithelial", ["KRT8", "KRT18", "EPCAM"], (0.25, 0.40), ["normal", "stressed"]),
        CellTypeTemplate("basal_epithelial", ["KRT14", "KRT5", "TP63"], (0.10, 0.20), ["normal"]),
        CellTypeTemplate("fibroblast", ["COL1A1", "COL3A1", "FAP"], (0.10, 0.20), ["quiescent", "activated"], True, (1.2, 1.8)),
        CellTypeTemplate("T_cell", ["CD3D", "CD3E", "CD8A"], (0.08, 0.15), ["quiescent", "activated", "exhausted"]),
        CellTypeTemplate("macrophage", ["CD68", "CD163", "CSF1R"], (0.05, 0.12), ["quiescent", "inflammatory"], True, (1.3, 2.0)),
        CellTypeTemplate("endothelial", ["PECAM1", "VWF", "CDH5"], (0.05, 0.10), ["quiescent"]),
    ],
    "synovium": [
        CellTypeTemplate("fibroblast", ["COL1A1", "FAP", "THY1"], (0.20, 0.30), ["quiescent", "activated"], True, (1.2, 1.8)),
        CellTypeTemplate("CD4_T_cell", ["CD3D", "CD4", "IL7R"], (0.12, 0.22), ["quiescent", "activated"]),
        CellTypeTemplate("CD8_T_cell", ["CD3D", "CD8A", "GZMB"], (0.08, 0.15), ["quiescent", "activated"]),
        CellTypeTemplate("macrophage", ["CD68", "CD163", "MARCO"], (0.10, 0.18), ["quiescent", "inflammatory"], True, (1.3, 2.0)),
        CellTypeTemplate("B_cell", ["CD19", "MS4A1", "CD79A"], (0.05, 0.12), ["quiescent"]),
        CellTypeTemplate("endothelial", ["PECAM1", "VWF"], (0.05, 0.10), ["quiescent"]),
        CellTypeTemplate("mast_cell", ["KIT", "TPSAB1", "CPA3"], (0.03, 0.08), ["quiescent"]),
    ],
    "aorta": [
        CellTypeTemplate("smooth_muscle", ["ACTA2", "MYH11", "TAGLN"], (0.30, 0.45), ["contractile", "synthetic"], True, (0.6, 0.9)),
        CellTypeTemplate("endothelial", ["PECAM1", "VWF", "CDH5"], (0.15, 0.25), ["quiescent", "activated"]),
        CellTypeTemplate("macrophage", ["CD68", "CD163", "TREM2"], (0.08, 0.15), ["quiescent", "inflammatory"], True, (1.5, 2.5)),
        CellTypeTemplate("fibroblast", ["COL1A1", "LUM", "DCN"], (0.08, 0.15), ["quiescent", "activated"]),
        CellTypeTemplate("T_cell", ["CD3D", "CD3E", "IL7R"], (0.05, 0.12), ["quiescent", "activated"]),
        CellTypeTemplate("dendritic_cell", ["FCER1A", "CD1C", "CLEC10A"], (0.03, 0.08), ["quiescent"]),
    ],
    "blood": [
        CellTypeTemplate("CD4_T_cell", ["CD3D", "CD4", "IL7R"], (0.15, 0.25), ["quiescent", "activated"]),
        CellTypeTemplate("CD8_T_cell", ["CD3D", "CD8A", "GZMB"], (0.10, 0.18), ["quiescent", "activated"]),
        CellTypeTemplate("B_cell", ["CD19", "MS4A1", "CD79A"], (0.08, 0.15), ["quiescent"]),
        CellTypeTemplate("NK_cell", ["NKG7", "GNLY", "KLRD1"], (0.05, 0.12), ["quiescent", "activated"]),
        CellTypeTemplate("monocyte", ["CD14", "CSF1R", "FCGR3A"], (0.15, 0.25), ["classical", "non_classical"]),
        CellTypeTemplate("neutrophil", ["ELANE", "MPO", "CTSG"], (0.10, 0.20), ["mature"]),
        CellTypeTemplate("platelet", ["ITGA2B", "GP1BA", "PF4"], (0.03, 0.08), ["normal"]),
    ],
    "spleen": [
        CellTypeTemplate("B_cell", ["CD19", "MS4A1", "CD79A"], (0.20, 0.35), ["quiescent", "activated"]),
        CellTypeTemplate("T_cell", ["CD3D", "CD3E", "IL7R"], (0.15, 0.25), ["quiescent", "activated"]),
        CellTypeTemplate("macrophage", ["CD68", "CD163", "CSF1R"], (0.10, 0.18), ["quiescent", "inflammatory"]),
        CellTypeTemplate("dendritic_cell", ["FCER1A", "CD1C", "CLEC10A"], (0.05, 0.10), ["quiescent"]),
        CellTypeTemplate("NK_cell", ["NKG7", "GNLY", "KLRD1"], (0.05, 0.12), ["quiescent"]),
        CellTypeTemplate("endothelial", ["PECAM1", "STAB2"], (0.05, 0.10), ["quiescent"]),
    ],
    "thymus": [
        CellTypeTemplate("double_negative_T", ["CD3D", "PTCRA"], (0.10, 0.18), ["progenitor"]),
        CellTypeTemplate("double_positive_T", ["CD3D", "CD4", "CD8A"], (0.30, 0.45), ["progenitor"]),
        CellTypeTemplate("CD4_SP", ["CD3D", "CD4", "IL7R"], (0.10, 0.18), ["mature"]),
        CellTypeTemplate("CD8_SP", ["CD3D", "CD8A", "CD8B"], (0.08, 0.15), ["mature"]),
        CellTypeTemplate("thymic_epithelial", ["FOXN1", "KRT5", "KRT8"], (0.05, 0.12), ["cortical", "medullary"]),
        CellTypeTemplate("dendritic_cell", ["FCER1A", "CD1C"], (0.03, 0.08), ["quiescent"]),
        CellTypeTemplate("macrophage", ["CD68", "CD163"], (0.03, 0.08), ["quiescent"]),
    ],
}


# ── Disease profiles ────────────────────────────────────────────────────────


@dataclass
class DiseaseProfile:
    name: str
    display_name: str
    tissue: str
    condition_name: str
    de_genes: Dict[str, Tuple[float, float]]
    pathways: Dict[str, Tuple[float, float]]
    markers: List[str]
    mechanism_templates: List[str]
    responding_cell_types: List[str] = field(default_factory=list)
    hidden_failure_templates: List[str] = field(default_factory=list)


DISEASE_PROFILES: Dict[str, DiseaseProfile] = {
    "dilated_cardiomyopathy": DiseaseProfile(
        name="dilated_cardiomyopathy",
        display_name="dilated cardiomyopathy",
        tissue="heart",
        condition_name="dilated_cardiomyopathy",
        de_genes={
            "NPPA": (1.8, 3.5), "NPPB": (2.0, 4.0), "MYH7": (1.0, 2.5),
            "COL1A1": (1.0, 2.2), "COL3A1": (0.8, 1.8), "POSTN": (1.5, 3.0),
            "CCL2": (0.8, 1.8), "IL6": (0.5, 1.5), "TGFB1": (0.7, 1.6),
            "ANKRD1": (1.5, 3.0), "XIRP2": (-2.0, -0.8), "MYL2": (-1.5, -0.5),
        },
        pathways={
            "cardiac_muscle_contraction": (0.3, 0.6),
            "extracellular_matrix_organisation": (0.7, 0.95),
            "inflammatory_response": (0.5, 0.8),
            "TGF_beta_signalling": (0.6, 0.85),
            "apoptosis": (0.4, 0.65),
        },
        markers=["NPPA", "NPPB", "POSTN", "COL1A1"],
        mechanism_templates=[
            "TGF-beta-driven fibrosis",
            "inflammatory macrophage infiltration",
        ],
        responding_cell_types=["cardiac_fibroblast", "macrophage"],
    ),
    "IPF": DiseaseProfile(
        name="IPF",
        display_name="idiopathic pulmonary fibrosis",
        tissue="lung",
        condition_name="IPF",
        de_genes={
            "SPP1": (2.0, 4.0), "MERTK": (0.8, 2.0), "MMP9": (1.0, 2.5),
            "TREM2": (0.8, 2.0), "COL1A1": (1.5, 3.0), "COL3A1": (1.2, 2.5),
            "POSTN": (1.5, 3.5), "SFTPC": (-2.0, -0.8), "AGER": (-2.5, -1.0),
        },
        pathways={
            "extracellular_matrix_organisation": (0.75, 0.95),
            "integrin_signalling": (0.6, 0.85),
            "macrophage_activation": (0.65, 0.9),
            "Wnt_signalling": (0.4, 0.7),
        },
        markers=["SPP1", "MERTK", "POSTN", "MMP9"],
        mechanism_templates=[
            "SPP1+ macrophage-driven fibroblast activation",
            "integrin-mediated SPP1 signalling in fibrosis",
        ],
        responding_cell_types=["fibroblast", "alveolar_macrophage"],
    ),
    "Alzheimer": DiseaseProfile(
        name="Alzheimer",
        display_name="Alzheimer's disease",
        tissue="brain",
        condition_name="Alzheimer",
        de_genes={
            "TREM2": (1.0, 2.5), "APOE": (1.2, 2.8), "CLU": (0.8, 2.0),
            "C1QA": (1.0, 2.2), "C1QB": (0.9, 2.0), "GFAP": (1.5, 3.0),
            "AQP4": (0.6, 1.5), "SLC17A7": (-1.5, -0.5), "NRGN": (-2.0, -0.8),
            "SNAP25": (-1.2, -0.4),
        },
        pathways={
            "complement_cascade": (0.7, 0.9),
            "neuroinflammation": (0.65, 0.85),
            "amyloid_processing": (0.6, 0.8),
            "synaptic_signalling": (0.3, 0.5),
            "lipid_metabolism": (0.5, 0.7),
        },
        markers=["TREM2", "APOE", "GFAP", "C1QA"],
        mechanism_templates=[
            "TREM2-mediated microglial activation in amyloid clearance",
            "complement-driven synaptic pruning",
            "reactive astrogliosis amplifying neuroinflammation",
        ],
        responding_cell_types=["microglia", "astrocyte"],
    ),
    "colorectal_cancer": DiseaseProfile(
        name="colorectal_cancer",
        display_name="colorectal cancer",
        tissue="colon",
        condition_name="colorectal_cancer",
        de_genes={
            "MYC": (1.5, 3.0), "KRAS": (0.8, 1.8), "TP53": (-1.5, -0.5),
            "APC": (-1.8, -0.8), "CDH1": (-1.2, -0.4), "VIM": (1.0, 2.5),
            "MKI67": (1.5, 3.0), "CD44": (1.0, 2.2), "LGR5": (0.8, 2.0),
        },
        pathways={
            "Wnt_signalling": (0.75, 0.95),
            "cell_cycle": (0.7, 0.9),
            "EMT": (0.6, 0.85),
            "p53_signalling": (0.3, 0.5),
            "MAPK_signalling": (0.55, 0.75),
        },
        markers=["MYC", "CD44", "VIM", "MKI67"],
        mechanism_templates=[
            "Wnt/beta-catenin-driven tumour stem cell expansion",
            "epithelial-mesenchymal transition promoting invasion",
        ],
        responding_cell_types=["stem_cell", "macrophage", "fibroblast"],
    ),
    "type2_diabetes": DiseaseProfile(
        name="type2_diabetes",
        display_name="type 2 diabetes",
        tissue="pancreas",
        condition_name="type2_diabetes",
        de_genes={
            "INS": (-2.0, -0.8), "MAFA": (-1.5, -0.5), "PDX1": (-1.2, -0.4),
            "UCN3": (-1.8, -0.6), "GCG": (0.8, 2.0), "ARX": (0.5, 1.5),
            "IAPP": (0.6, 1.8), "TXNIP": (1.0, 2.5), "DDIT3": (0.8, 2.0),
        },
        pathways={
            "insulin_signalling": (0.3, 0.5),
            "ER_stress_response": (0.7, 0.9),
            "oxidative_stress": (0.6, 0.8),
            "glucagon_signalling": (0.6, 0.8),
            "apoptosis": (0.5, 0.7),
        },
        markers=["INS", "TXNIP", "IAPP", "DDIT3"],
        mechanism_templates=[
            "ER stress-induced beta cell apoptosis",
            "glucotoxicity-driven beta cell dedifferentiation",
        ],
        responding_cell_types=["beta_cell", "stellate"],
    ),
    "AML": DiseaseProfile(
        name="AML",
        display_name="acute myeloid leukemia",
        tissue="bone_marrow",
        condition_name="AML",
        de_genes={
            "FLT3": (1.5, 3.0), "NPM1": (0.8, 2.0), "IDH2": (0.6, 1.5),
            "RUNX1": (-1.5, -0.5), "CEBPA": (-1.2, -0.4), "KIT": (1.0, 2.5),
            "WT1": (1.2, 2.8), "MYC": (0.8, 2.0),
        },
        pathways={
            "hematopoietic_cell_lineage": (0.3, 0.5),
            "MAPK_signalling": (0.7, 0.9),
            "PI3K_AKT_signalling": (0.65, 0.85),
            "cell_cycle": (0.7, 0.9),
            "apoptosis": (0.3, 0.5),
        },
        markers=["FLT3", "NPM1", "WT1", "KIT"],
        mechanism_templates=[
            "FLT3-ITD-driven proliferative advantage",
            "myeloid differentiation block via RUNX1 loss",
        ],
        responding_cell_types=["HSC", "GMP"],
    ),
    "rheumatoid_arthritis": DiseaseProfile(
        name="rheumatoid_arthritis",
        display_name="rheumatoid arthritis",
        tissue="synovium",
        condition_name="rheumatoid_arthritis",
        de_genes={
            "IFNG": (1.0, 2.5), "TBX21": (0.8, 1.8), "IL17A": (1.0, 2.2),
            "RORC": (0.6, 1.5), "TNF": (1.2, 2.5), "IL6": (1.0, 2.2),
            "MMP3": (1.5, 3.0), "MMP1": (1.2, 2.5), "CXCL13": (1.0, 2.5),
        },
        pathways={
            "JAK_STAT_signalling": (0.7, 0.9),
            "TNF_signalling": (0.7, 0.9),
            "Th17_differentiation": (0.6, 0.8),
            "NF_kB_signalling": (0.65, 0.85),
            "matrix_metalloproteinase_activity": (0.7, 0.9),
        },
        markers=["TNF", "IL6", "MMP3", "CXCL13"],
        mechanism_templates=[
            "TNF/NF-kB-driven synovial inflammation",
            "Th17-mediated cartilage destruction via MMPs",
        ],
        responding_cell_types=["fibroblast", "macrophage", "CD4_T_cell"],
    ),
    "hepatocellular_carcinoma": DiseaseProfile(
        name="hepatocellular_carcinoma",
        display_name="hepatocellular carcinoma",
        tissue="liver",
        condition_name="HCC",
        de_genes={
            "GPC3": (2.0, 4.0), "AFP": (1.5, 3.5), "EPCAM": (1.0, 2.5),
            "MYC": (1.0, 2.5), "VEGFA": (1.2, 2.8), "MKI67": (1.5, 3.0),
            "ALB": (-2.0, -0.8), "CYP3A4": (-1.8, -0.6), "APOB": (-1.5, -0.5),
        },
        pathways={
            "Wnt_signalling": (0.7, 0.9),
            "cell_cycle": (0.75, 0.95),
            "angiogenesis": (0.6, 0.8),
            "PI3K_AKT_signalling": (0.65, 0.85),
            "p53_signalling": (0.3, 0.5),
        },
        markers=["GPC3", "AFP", "VEGFA", "MKI67"],
        mechanism_templates=[
            "Wnt/beta-catenin-driven hepatocyte dedifferentiation",
            "VEGF-mediated tumour angiogenesis",
        ],
        responding_cell_types=["kupffer_cell", "stellate_cell"],
        hidden_failure_templates=[
            "Tumour heterogeneity may confound DE in mixed biopsies",
        ],
    ),
    "atherosclerosis": DiseaseProfile(
        name="atherosclerosis",
        display_name="atherosclerosis",
        tissue="aorta",
        condition_name="atherosclerosis",
        de_genes={
            "TREM2": (1.5, 3.0), "CD9": (1.0, 2.2), "LGALS3": (1.2, 2.5),
            "APOE": (0.8, 2.0), "MMP9": (1.0, 2.5), "IL1B": (0.8, 2.0),
            "ACTA2": (-1.5, -0.5), "MYH11": (-2.0, -0.8), "CNN1": (-1.5, -0.5),
        },
        pathways={
            "lipid_metabolism": (0.7, 0.9),
            "inflammatory_response": (0.65, 0.85),
            "foam_cell_formation": (0.7, 0.9),
            "smooth_muscle_contraction": (0.3, 0.5),
            "complement_cascade": (0.5, 0.7),
        },
        markers=["TREM2", "LGALS3", "MMP9", "CD9"],
        mechanism_templates=[
            "TREM2+ macrophage-driven foam cell formation",
            "smooth muscle cell phenotypic switching in plaque",
        ],
        responding_cell_types=["macrophage", "smooth_muscle"],
    ),
    "breast_cancer": DiseaseProfile(
        name="breast_cancer",
        display_name="breast cancer",
        tissue="breast",
        condition_name="breast_cancer",
        de_genes={
            "ERBB2": (1.5, 3.5), "ESR1": (-1.5, 1.5), "MKI67": (1.5, 3.0),
            "MYC": (1.0, 2.5), "CDH1": (-1.5, -0.3), "VIM": (0.8, 2.2),
            "CD274": (0.8, 2.0), "FOXP3": (0.6, 1.5), "GZMB": (0.8, 2.0),
        },
        pathways={
            "cell_cycle": (0.7, 0.9),
            "PI3K_AKT_signalling": (0.65, 0.85),
            "EMT": (0.55, 0.8),
            "immune_checkpoint": (0.5, 0.75),
            "estrogen_signalling": (0.3, 0.7),
        },
        markers=["ERBB2", "MKI67", "CD274", "VIM"],
        mechanism_templates=[
            "ERBB2-driven proliferative signalling",
            "immune evasion via PD-L1 upregulation",
        ],
        responding_cell_types=["macrophage", "T_cell", "fibroblast"],
    ),
    "multiple_sclerosis": DiseaseProfile(
        name="multiple_sclerosis",
        display_name="multiple sclerosis",
        tissue="brain",
        condition_name="multiple_sclerosis",
        de_genes={
            "CD68": (1.0, 2.5), "CXCL10": (1.2, 2.8), "STAT1": (0.8, 2.0),
            "IRF1": (0.8, 1.8), "MBP": (-2.0, -0.8), "PLP1": (-1.8, -0.6),
            "MOG": (-1.5, -0.5), "GFAP": (1.0, 2.5), "C3": (0.8, 2.0),
        },
        pathways={
            "interferon_signalling": (0.7, 0.9),
            "neuroinflammation": (0.7, 0.9),
            "complement_cascade": (0.6, 0.8),
            "myelination": (0.2, 0.4),
            "T_cell_activation": (0.6, 0.8),
        },
        markers=["CXCL10", "STAT1", "GFAP", "C3"],
        mechanism_templates=[
            "interferon-driven microglial activation in demyelination",
            "complement-mediated oligodendrocyte damage",
        ],
        responding_cell_types=["microglia", "astrocyte"],
    ),
    "diabetic_nephropathy": DiseaseProfile(
        name="diabetic_nephropathy",
        display_name="diabetic nephropathy",
        tissue="kidney",
        condition_name="diabetic_nephropathy",
        de_genes={
            "HAVCR1": (1.5, 3.0), "LCN2": (1.2, 2.8), "COL4A1": (1.0, 2.5),
            "VEGFA": (0.8, 2.0), "NPHS1": (-1.8, -0.6), "NPHS2": (-1.5, -0.5),
            "WT1": (-1.2, -0.4), "TGFB1": (1.0, 2.2), "FN1": (1.2, 2.5),
        },
        pathways={
            "TGF_beta_signalling": (0.7, 0.9),
            "extracellular_matrix_organisation": (0.7, 0.9),
            "oxidative_stress": (0.6, 0.8),
            "VEGF_signalling": (0.5, 0.7),
            "apoptosis": (0.5, 0.7),
        },
        markers=["HAVCR1", "LCN2", "TGFB1", "FN1"],
        mechanism_templates=[
            "TGF-beta-driven glomerular fibrosis",
            "podocyte loss via oxidative stress",
        ],
        responding_cell_types=["fibroblast", "macrophage"],
    ),
    "melanoma": DiseaseProfile(
        name="melanoma",
        display_name="melanoma",
        tissue="skin",
        condition_name="melanoma",
        de_genes={
            "MLANA": (1.5, 3.0), "PMEL": (1.2, 2.5), "SOX10": (1.0, 2.2),
            "MKI67": (1.5, 3.0), "CD274": (0.8, 2.0), "PDCD1": (0.8, 2.0),
            "GZMB": (0.8, 2.0), "HAVCR2": (0.6, 1.5), "LAG3": (0.6, 1.5),
        },
        pathways={
            "MAPK_signalling": (0.7, 0.9),
            "immune_checkpoint": (0.6, 0.85),
            "cell_cycle": (0.7, 0.9),
            "melanogenesis": (0.5, 0.7),
            "T_cell_exhaustion": (0.55, 0.8),
        },
        markers=["MLANA", "CD274", "GZMB", "MKI67"],
        mechanism_templates=[
            "MAPK-driven melanocyte proliferation",
            "T cell exhaustion via immune checkpoint upregulation",
        ],
        responding_cell_types=["T_cell", "macrophage"],
    ),
}


# ── Pathway library ─────────────────────────────────────────────────────────

PATHWAY_LIBRARY: Dict[str, List[str]] = {
    "TGF_beta_signalling": ["TGFB1", "TGFB2", "SMAD2", "SMAD3", "SMAD4", "ACVR1"],
    "Wnt_signalling": ["WNT3A", "CTNNB1", "APC", "AXIN2", "LGR5", "TCF7L2"],
    "MAPK_signalling": ["KRAS", "BRAF", "MAP2K1", "MAPK1", "MAPK3", "FOS", "JUN"],
    "JAK_STAT_signalling": ["JAK1", "JAK2", "STAT1", "STAT3", "STAT5A", "SOCS1", "SOCS3"],
    "PI3K_AKT_signalling": ["PIK3CA", "AKT1", "MTOR", "PTEN", "TSC2"],
    "NF_kB_signalling": ["NFKB1", "RELA", "IKBKB", "TNF", "IL1B"],
    "cell_cycle": ["CDK4", "CDK6", "CCND1", "CCNE1", "RB1", "E2F1", "MKI67"],
    "apoptosis": ["BCL2", "BAX", "BAK1", "CASP3", "CASP9", "TP53", "BID"],
    "inflammatory_response": ["TNF", "IL6", "IL1B", "CCL2", "CXCL8", "NFKB1"],
    "extracellular_matrix_organisation": ["COL1A1", "COL3A1", "FN1", "POSTN", "MMP2", "MMP9", "TIMP1"],
    "complement_cascade": ["C1QA", "C1QB", "C3", "C4A", "C5", "CFB"],
    "neuroinflammation": ["TREM2", "CX3CR1", "P2RY12", "IL1B", "TNF", "C1QA"],
    "synaptic_signalling": ["SLC17A7", "GRIA1", "GRIN1", "DLG4", "SNAP25", "SYP"],
    "hematopoietic_cell_lineage": ["CD34", "KIT", "FLT3", "GATA1", "CEBPA", "SPI1"],
    "insulin_signalling": ["INS", "INSR", "IRS1", "PIK3CA", "AKT1", "SLC2A4"],
    "ER_stress_response": ["DDIT3", "ATF4", "XBP1", "HSPA5", "EIF2AK3"],
    "oxidative_stress": ["SOD1", "SOD2", "CAT", "GPX1", "NFE2L2", "HMOX1"],
    "angiogenesis": ["VEGFA", "VEGFB", "KDR", "FLT1", "ANGPT1", "ANGPT2"],
    "EMT": ["CDH1", "CDH2", "VIM", "SNAI1", "SNAI2", "TWIST1", "ZEB1"],
    "immune_checkpoint": ["CD274", "PDCD1", "CTLA4", "HAVCR2", "LAG3", "TIGIT"],
    "T_cell_activation": ["CD3D", "CD28", "LCK", "ZAP70", "IL2", "IFNG"],
    "T_cell_exhaustion": ["PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX", "ENTPD1"],
    "TNF_signalling": ["TNF", "TNFRSF1A", "TRADD", "RIPK1", "NFKB1", "CASP8"],
    "Th17_differentiation": ["IL17A", "IL17F", "RORC", "IL23R", "CCR6", "STAT3"],
    "interferon_signalling": ["IFNG", "IFNB1", "STAT1", "IRF1", "IRF7", "MX1", "OAS1"],
    "lipid_metabolism": ["APOE", "APOB", "LDLR", "HMGCR", "ABCA1", "PPARG"],
    "myelination": ["MBP", "PLP1", "MOG", "MAG", "OLIG2", "SOX10"],
    "foam_cell_formation": ["CD36", "MSR1", "ABCA1", "APOE", "LGALS3", "TREM2"],
    "smooth_muscle_contraction": ["ACTA2", "MYH11", "TAGLN", "CNN1", "MYLK"],
    "glucagon_signalling": ["GCG", "GCGR", "CREB1", "PCK1", "G6PC"],
    "matrix_metalloproteinase_activity": ["MMP1", "MMP2", "MMP3", "MMP9", "TIMP1", "TIMP2"],
    "estrogen_signalling": ["ESR1", "ESR2", "PGR", "GREB1", "TFF1"],
    "melanogenesis": ["MITF", "TYR", "TYRP1", "DCT", "MLANA", "PMEL"],
    "VEGF_signalling": ["VEGFA", "VEGFB", "KDR", "FLT1", "NRP1"],
}


# ── Regulatory network templates ────────────────────────────────────────────

REGULATORY_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "erythroid": {
        "GATA1": ["KLF1", "HBB", "HBA1", "GYPA", "ALAS2"],
        "KLF1": ["HBB", "HBA1", "SLC4A1"],
    },
    "myeloid": {
        "CEBPA": ["CSF3R", "ELANE", "MPO", "CTSG"],
        "SPI1": ["CSF1R", "CD14", "FCGR3A", "CD68"],
    },
    "lymphoid": {
        "TCF7": ["CD3D", "CD3E", "IL7R", "LEF1"],
        "PAX5": ["CD19", "MS4A1", "CD79A"],
    },
    "fibrotic": {
        "SMAD3": ["COL1A1", "COL3A1", "FN1", "POSTN"],
        "TGFB1": ["ACTA2", "COL1A1", "CTGF"],
    },
    "inflammatory": {
        "NFKB1": ["TNF", "IL6", "IL1B", "CCL2", "CXCL8"],
        "STAT1": ["IRF1", "CXCL10", "MX1", "OAS1"],
    },
    "stem_cell": {
        "RUNX1": ["CD34", "KIT", "FLT3"],
        "MYC": ["CDK4", "CCND1", "E2F1"],
    },
    "neuronal": {
        "NEUROD1": ["SLC17A7", "NRGN", "SNAP25"],
        "DLX1": ["GAD1", "GAD2", "SLC32A1"],
    },
}


# ── Perturbation templates ──────────────────────────────────────────────────

@dataclass
class PerturbationTemplate:
    name: str
    target_pathway: str
    gene_effects: Dict[str, float]
    description: str


PERTURBATION_TEMPLATES: Dict[str, PerturbationTemplate] = {
    "JAK_inhibitor": PerturbationTemplate(
        name="JAK_inhibitor",
        target_pathway="JAK_STAT_signalling",
        gene_effects={"STAT1": -0.8, "STAT3": -0.7, "IFNG": -1.5, "IL17A": -1.3, "SOCS1": 1.2},
        description="JAK inhibitor treatment",
    ),
    "anti_TNF": PerturbationTemplate(
        name="anti_TNF",
        target_pathway="TNF_signalling",
        gene_effects={"TNF": -1.5, "IL6": -1.0, "IL1B": -0.8, "MMP3": -1.2, "SOCS3": 0.8},
        description="anti-TNF biologic therapy",
    ),
    "PD1_blockade": PerturbationTemplate(
        name="PD1_blockade",
        target_pathway="immune_checkpoint",
        gene_effects={"PDCD1": -1.0, "GZMB": 1.5, "IFNG": 1.2, "PRF1": 1.0, "TNF": 0.8},
        description="anti-PD-1 immune checkpoint blockade",
    ),
    "BRAF_inhibitor": PerturbationTemplate(
        name="BRAF_inhibitor",
        target_pathway="MAPK_signalling",
        gene_effects={"BRAF": -0.5, "MAPK1": -1.0, "MKI67": -1.5, "CCND1": -1.2, "FOS": -0.8},
        description="BRAF inhibitor treatment",
    ),
    "TGFb_inhibitor": PerturbationTemplate(
        name="TGFb_inhibitor",
        target_pathway="TGF_beta_signalling",
        gene_effects={"TGFB1": -0.8, "COL1A1": -1.2, "COL3A1": -1.0, "POSTN": -1.5, "ACTA2": -0.8},
        description="TGF-beta pathway inhibitor",
    ),
    "mTOR_inhibitor": PerturbationTemplate(
        name="mTOR_inhibitor",
        target_pathway="PI3K_AKT_signalling",
        gene_effects={"MTOR": -0.8, "AKT1": -0.6, "MKI67": -1.2, "CCND1": -1.0, "HIF1A": -0.7},
        description="mTOR inhibitor treatment",
    ),
    "CRISPR_TP53_KO": PerturbationTemplate(
        name="CRISPR_TP53_KO",
        target_pathway="p53_signalling",
        gene_effects={"TP53": -2.0, "BAX": -1.0, "CDKN1A": -1.5, "MDM2": -0.8, "MKI67": 1.0},
        description="CRISPR knockout of TP53",
    ),
}


# ── Trajectory templates ────────────────────────────────────────────────────

@dataclass
class TrajectoryTemplate:
    """Template for a developmental trajectory through cell populations."""
    root_population: str
    branches: List[List[str]]
    n_lineages: int
    tissue: str


TRAJECTORY_TEMPLATES: List[TrajectoryTemplate] = [
    TrajectoryTemplate(
        root_population="HSC",
        branches=[
            ["HSC", "CMP", "GMP", "neutrophil"],
            ["HSC", "CMP", "GMP", "monocyte"],
            ["HSC", "MEP", "erythrocyte"],
            ["HSC", "MEP", "megakaryocyte"],
        ],
        n_lineages=3,
        tissue="bone_marrow",
    ),
    TrajectoryTemplate(
        root_population="double_negative_T",
        branches=[
            ["double_negative_T", "double_positive_T", "CD4_SP"],
            ["double_negative_T", "double_positive_T", "CD8_SP"],
        ],
        n_lineages=2,
        tissue="thymus",
    ),
    TrajectoryTemplate(
        root_population="stem_cell",
        branches=[
            ["stem_cell", "colonocyte"],
            ["stem_cell", "goblet_cell"],
        ],
        n_lineages=2,
        tissue="colon",
    ),
    TrajectoryTemplate(
        root_population="OPC",
        branches=[
            ["OPC", "oligodendrocyte"],
        ],
        n_lineages=1,
        tissue="brain",
    ),
]


# ── Hidden failure condition templates ──────────────────────────────────────

HIDDEN_FAILURE_TEMPLATES: List[str] = [
    "High ambient RNA may confound DE in low-abundance transcripts",
    "Strong batch effects between conditions may inflate false positives",
    "Low cell viability in disease samples reduces statistical power",
    "Doublet contamination in dense populations obscures rare cell types",
    "Sample degradation during processing introduces 3' bias artefacts",
    "Dissociation-induced gene expression changes confound stress signatures",
    "Unbalanced sample sizes between conditions reduce DE sensitivity",
]
