# miRNA-PROTAC example

This directory contains a prepared miRNA-PROTAC ternary-complex model and the
configurations for linker modeling, RESP charge fitting, and Amber simulation.

```text
examples/mirna/
├── structures/       # prepared complex and CRL/E2 reference
├── charges/          # RESP inputs
├── amber/            # Amber inputs
├── modeling.yaml
├── charges.yaml
└── simulation.yaml
```

## Modeling

| Chain | Component |
| --- | --- |
| `A` | protein of interest |
| `B` | miRNA warhead |
| `C` | E3 ligase |
| `X` | E3-ligand fragment |
| `L` | generated linker |

The attachment atoms are `B:24:O3'` and `X:1:N3`. Reference chains `V` and
`E` provide the E3 alignment and E2 active site.

```bash
export ROSETTA_ROOT=/path/to/rosetta
oligoternary-preflight
oligoternary validate examples/mirna/modeling.yaml
oligoternary run examples/mirna/modeling.yaml --dry-run
oligoternary run examples/mirna/modeling.yaml
```

Outputs are written to `runs/mirna-example/`.

## RESP charge fitting

```bash
oligoternary-charge validate examples/mirna/charges.yaml
oligoternary-charge fit examples/mirna/charges.yaml
```

Outputs are written to `runs/mirna-resp/`.

## Amber simulation

```bash
oligoternary-simulate validate examples/mirna/simulation.yaml
oligoternary-simulate run examples/mirna/simulation.yaml --dry-run
oligoternary-simulate run examples/mirna/simulation.yaml
```

Outputs are written to `runs/mirna-md/`. Amber must be installed separately.
