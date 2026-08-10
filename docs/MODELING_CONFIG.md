# Modeling configuration

`oligoternary` accepts YAML or JSON run specifications with `version: 1`.
Relative paths are resolved from the configuration file.

```yaml
version: 1
project: my-model
output_dir: runs/workflow
stages: []
```

Each stage has a unique `name`, optional earlier `depends_on` entries, and one
`adapter`.

## Existing input

```yaml
- name: prepared-input
  adapter:
    type: existing-artifact
    artifact: prepared_complex.pdb
```

## E2 active-site accessibility screen

Use a command stage to align the recruited E3 chain to its catalytic CRL/E2
reference and screen its active-site accessibility before linker reconstruction:

```yaml
- name: e2-accessibility-screen
  depends_on: [prepared-input]
  adapter:
    type: command
    command:
      - "{python}"
      - -m
      - oligoternary.cli.e2_accessibility
      - --input-pdb
      - prepared_complex.pdb
      - --reference-pdb
      - crl4crbn_e2_reference.pdb.gz
      - --output-json
      - runs/e2-accessibility-screen/screen.json
      - --poi-chain
      - A
      - --e3-chain
      - C
      - --reference-e3-chain
      - V
      - --e2-chain
      - E
      - --e2-residue
      - "85"
      - --minimum-alignment-residues
      - "30"
      - --maximum-alignment-rmsd
      - "3"
      - --minimum-separation
      - "3"
      - --contact-cutoff
      - "2"
      - --severe-cutoff
      - "1.5"
      - --maximum-contacts
      - "5"
      - --max-lysine-distance
      - "25"
      - "{stage-result}"
    artifact: runs/e2-accessibility-screen/screen.json
    result_summary: runs/e2-accessibility-screen/stage_result.json
```

The stage passes when the E3 alignment satisfies its residue-count and RMSD
limits, target–E2 steric criteria are satisfied, and at least one target Lys Nζ
atom is within the configured distance of the E2 catalytic Cys Sγ atom. All
thresholds are CLI options; the shown values reproduce the bundled protocol.
This is a geometry screen of E2 active-site accessibility, not a prediction of
downstream biochemical activity.

## Linker refinement

The typed `linker-refinement` adapter collects the chemistry and structural
mapping needed by `oligoternary-refine`.

```yaml
- name: linker-refinement
  depends_on: [e2-accessibility-screen]
  adapter:
    type: linker-refinement
    input: prepared_complex.pdb
    output_dir: runs/modeling
    result_summary: runs/modeling/refinement_result.json
    chemistry:
      linker_prefix: my_linker
      linker_smiles: "[*]CCOCC[*]"
      e3_ligand_smiles: "CCOC[*]"
      linker_to_warhead_atom: C1
      linker_to_e3_ligand_atom: C4
    chains:
      poi: A
      warhead: B
      e3: C
      linker: L
      e3_ligand: X
    anchors:
      warhead: "B:1:O3'"
      e3_ligand: "X:1:C3"
    protocol:
      relax_cycles: 5
      minimize_steps: 1000
      minimize_tolerance: 0.001
      random_seed: 42
      endpoint_candidates: 48
      conformers_per_endpoint: 2
      minimum_attachment_angle_degrees: 80
      conformer_ranking: clash-first
```

`endpoint_candidates` and `conformers_per_endpoint` set the conformer-search
size. `minimum_attachment_angle_degrees: 0` disables the attachment-angle
filter. `conformer_ranking` accepts `attachment-first` or `clash-first`.
Together with `random_seed`, these settings should be recorded when a specific
modeled linker must be reproduced.

Rosetta's `molfile_to_params.py` is discovered from `MOLFILE_TO_PARAMS`,
`ROSETTA_ROOT`, or `PATH`. A run specification may instead set
`tools.molfile_to_params` or `tools.rosetta_root`.

The adapter declares the refined PDB, `LNK.params`, and `E3L.params` as stage
artifacts.

## Command stage

Installed commands can be used as later stages. Commands are argument lists,
not shell strings.

```yaml
- name: structural-metrics
  depends_on: [linker-refinement]
  adapter:
    type: command
    command:
      - "{python}"
      - -m
      - oligoternary.cli.metrics
      - --input-folder
      - runs/modeling/models
      - --output-csv
      - runs/modeling/metrics.csv
      - --poi-chain
      - A
      - --warhead-chain
      - B
      - --e3-chain
      - C
      - --protac-chains
      - B
      - L
      - X
      - --params-files
      - runs/modeling/params/LNK.params
      - runs/modeling/params/E3L.params
      - "{stage-result}"
    artifact: runs/modeling/metrics.csv
    result_summary: runs/modeling/metrics_result.json
```

`{python}` expands to the active interpreter. `{stage-result}` expands to the
declared result-summary and artifact arguments. This stage analyzes the refined
PDB files and writes the declared CSV artifact. The complete command, including
linker-geometry arguments, is in
[the miRNA example](../examples/mirna/modeling.yaml).

## Execute

```bash
oligoternary validate my-modeling.yaml
oligoternary run my-modeling.yaml --dry-run
oligoternary run my-modeling.yaml
```
