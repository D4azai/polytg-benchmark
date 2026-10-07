# Data contract

Place the source CSV in `data/raw/` and keep it immutable. The default configuration expects
`data/raw/polytg.csv` with columns named `smiles` and `tg`; both names and all paths are configurable.
Tg units are not inferred or converted. Curate a dataset to one documented unit and set `target_unit`
in the configuration.

Preparation writes a cleaned CSV, a JSON cleaning report, and hash-verified split manifests below
`data/processed/`. Raw and processed datasets are ignored by Git to prevent accidental redistribution.
Only `.gitkeep` files are tracked.

The cleaner removes missing or nonnumeric targets, invalid SMILES, and exact duplicate source rows. It
reports repeated canonical structures and disagreements in Tg. Repeats can remain as distinct
measurements or be aggregated by mean/median. Every split groups identical canonical structures, so
repeated measurements cannot leak across partitions.

Polymer repeat-unit SMILES, end groups, stereochemistry, copolymer composition, molecular weight, and
measurement conditions are scientifically consequential. Record these fields when available; a single
SMILES and Tg cannot express all sources of experimental variation.

