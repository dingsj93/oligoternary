"""Shared constants for linker refinement."""
import re

ATOM_LABEL_PATTERN = re.compile(
    r"^(?P<chain>[^:]+):(?P<resnum>-?\d+)(?P<icode>[A-Za-z]?):(?P<atom>[^:]+)$"
)
DEFAULT_RANDOM_SEED = 42
LINKER_RESNAME = "LNK"
E3L_RESNAME = "E3L"
ATTACHMENT_DIRECTION_SAMPLES = 256
ATTACHMENT_TOP_CANDIDATES = 48
ATTACHMENT_ENDPOINT_CANDIDATES = 48
PREPARE_CONFORMER_CANDIDATES = 2
MIN_ATTACHMENT_ANGLE_DEGREES = 80.0
CONFORMER_RANKING = "clash-first"
CONFORMER_RANKINGS = ("attachment-first", "clash-first")
LINKER_ENVIRONMENT_CLASH_MIN_DISTANCE = 1.8
LINKER_ENVIRONMENT_CLASH_WARNING_OVERLAP = 1.0
VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "P": 1.80,
    "S": 1.80,
}
