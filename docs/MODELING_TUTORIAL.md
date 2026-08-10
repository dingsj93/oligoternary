# Modeling tutorial

OligoTernary starts from a prepared ternary-complex PDB, evaluates E2
active-site accessibility in an E3–E2 reference context, and reconstructs the
covalent linker between the oligonucleotide warhead and the bound E3 ligand.

## 1. Install the runtime

```bash
conda env create -f environment.yml
conda activate oligoternary
python -m pip install -e .
```

Install Rosetta and PyRosetta separately, then expose the Rosetta root:

```bash
export ROSETTA_ROOT=/path/to/rosetta
python -c "import pyrosetta; print(pyrosetta.version())"
oligoternary-preflight
```

The complete example has been tested with Python 3.10 and PyRosetta
2025.22+release.957f89124e.

## 2. Run the bundled miRNA example

The example already contains a prepared PDB, compressed CRL4–CRBN–E2
reference, marked SMILES, chain maps, and anchor labels.

```bash
oligoternary validate examples/mirna/modeling.yaml
oligoternary run examples/mirna/modeling.yaml --dry-run
oligoternary run examples/mirna/modeling.yaml
```

The input uses chains `A` (POI), `B` (miRNA warhead), `C` (E3), and `X`
(bound E3-ligand fragment). OligoTernary creates linker chain `L` between
`B:24:O3'` and `X:1:N3`.

The workflow first aligns input CRBN chain `C` to reference chain `V`, places
E2 chain `E`, requires at least 30 aligned residues and no more than `3 Å`
alignment RMSD, applies the configured target–E2 steric criteria, and measures
target Lys Nζ distances to catalytic atom `E:85:SG`. Passing structures continue
to linker reconstruction.

## 3. Run a custom system

Copy the example directory at the same depth so that its relative paths remain
valid:

```bash
cp -R examples/mirna examples/my-system
```

Update these fields together:

- the prepared input, E2-accessibility input, and refinement input;
- the catalytic reference path, E3 chain pair, E2 chain, and active-site residue;
- the E3 alignment, E2 steric, and Lys-distance thresholds when the protocol differs;
- `chemistry.linker_smiles` and `chemistry.e3_ligand_smiles`;
- the five chain IDs;
- `anchors.warhead` and `anchors.e3_ligand`;
- the linker connection atom names;
- output paths and protocol settings.

The linker SMILES uses two `[*]` markers. The E3-ligand SMILES uses one.
Anchor labels follow `chain:residue-number:atom-name`.

The custom PDB must already contain the docked target–oligonucleotide and
E3–ligand modules. OligoTernary does not perform the upstream docking step. Use
a catalytic reference appropriate to the recruited E3 ligase rather than the
bundled CRBN reference for unrelated systems.

```bash
oligoternary validate examples/my-system/modeling.yaml
oligoternary run examples/my-system/modeling.yaml --dry-run
oligoternary run examples/my-system/modeling.yaml
```

## 4. Outputs

The example writes:

```text
runs/mirna-example/
├── e2-accessibility-screen/
│   ├── screen.json
│   └── stage_result.json
├── modeling/
│   ├── models/
│   ├── params/
│   ├── tmp/
│   ├── metrics.csv
│   ├── refinement_result.json
│   └── metrics_result.json
└── workflow/
    ├── logs/
    ├── run_manifest.json
    └── run_manifests/
```

For the bundled example, the final structure is
`runs/mirna-example/modeling/models/prepared_complex_example_linker_full_optimized.pdb`.
A completed workflow reports `"overall_status": "succeeded"`.
