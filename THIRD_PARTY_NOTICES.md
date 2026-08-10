# Third-party software notices

OligoTernary is distributed under the MIT License. The external software below
is not included in this repository and remains subject to its own license.

## Rosetta and PyRosetta

Linker parameterization and refinement require separately obtained Rosetta and
PyRosetta software. RosettaCommons permits non-commercial use under its
non-commercial license and requires a separate license for commercial use.
OligoTernary does not redistribute Rosetta, PyRosetta, or
`molfile_to_params.py`.

- License information: <https://rosettacommons.org/software/licensing-faq/>
- Software access: <https://rosettacommons.org/software/documentation/>

## Amber

Molecular dynamics execution requires a separately obtained Amber executable,
such as `pmemd.cuda`. Amber is not included in this repository and remains
subject to its own license. The bundled `prmtop` and `inpcrd` files are research
inputs, not copies of Amber software.

- License information: <https://ambermd.org/AmberMD.php>

## Python dependencies

Python dependencies installed separately through conda or pip retain their own
licenses. Their inclusion as package requirements does not place them under the
OligoTernary MIT License.

## RESP numerical reference

The RESP solver in `src/oligoternary/simulation/resp.py` follows the published
RESP equations and was checked for numerical compatibility with the
BSD-licensed `cdsgroup/resp` implementation. No files from that package are
bundled here.

- Reference implementation: <https://github.com/cdsgroup/resp>

## Structural reference data

The bundled `examples/mirna/structures/crl4crbn_e2_reference.pdb.gz` file is
a project-generated CRL4–CRBN catalytic-context model. Its construction used
structural information from RCSB PDB entries
[8B3G](https://www.rcsb.org/structure/8B3G),
[1LDJ](https://www.rcsb.org/structure/1LDJ), and
[6D4P](https://www.rcsb.org/structure/6D4P), together with a predicted
DDB1–CRBN module. These source structures should be cited independently when
the reference model is used.
