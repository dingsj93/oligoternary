# RESP charge fitting and Amber simulation

The bundled miRNA example includes the three-conformer ESP inputs,
fitted-charge constraints, and solvated Amber inputs used for the publication
simulation.

## 1. Reproduce the RESP charges

Run from the repository root:

```bash
oligoternary-charge validate examples/mirna/charges.yaml
oligoternary-charge fit examples/mirna/charges.yaml
```

This fits one charge vector to three precomputed HF/6-31G* gas-phase ESP grids
with the publication two-stage constraints. Results are written to
`runs/mirna-resp/`. `charges.csv` contains both fit stages, and
`fit_report.json` records the conformer weights, restraint settings, total
charge, and relative RMS errors. Quantum-chemistry checkpoints are not required
for this fit and are not included in the repository.

## 2. Install Amber

Install Amber separately and make an engine such as `pmemd.cuda` available on
`PATH`:

```bash
pmemd.cuda -h
```

Amber is licensed separately and is not included with OligoTernary.

## 3. Check the Amber inputs

Run from the repository root:

```bash
oligoternary-simulate validate examples/mirna/simulation.yaml
oligoternary-simulate run examples/mirna/simulation.yaml --dry-run
```

Validation checks that the topology and coordinates both contain 151,815
atoms and that the coordinate payload is complete. The dry run prints the six
Amber commands without writing outputs.

## 4. Run the simulation

```bash
oligoternary-simulate run examples/mirna/simulation.yaml
```

The protocol runs restrained and unrestrained minimization, NVT heating,
restrained and unrestrained NPT equilibration, and 50 ns NPT production.
Outputs are written to `runs/mirna-md/`.

The production trajectory is large and is intentionally excluded from Git.
The bundled Amber inputs reproduce this example system; they are not a general
parameter set for a different linker or oligonucleotide chemistry.
