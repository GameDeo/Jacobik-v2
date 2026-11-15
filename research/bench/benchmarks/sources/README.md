# Source archive

`challenge/` adds metadata for twelve research papers, the publisher's
Zr-HOPO CIF, the complete 26-frame Xantphos/XL-Xantphos computed XYZ supplement,
publisher Figshare provenance records, and five pinned upstream source files
used to inspect haptic/site conventions. The challenge's experimental and
computed coordinate sets have separate labels and atom mappings under
`../data/challenge/`. Paper text and figures are not included in this archive.

Public crystallographic supplementary files and upstream installation metadata
are archived here with SHA-256 in `manifest.json`. Paper citations, download
URLs, nominated crystal datablocks, and preparation notes are in
`../data/sources.json`.
No tmQM files or repository datasets are used to construct this benchmark.

The CIFs are experimental crystal structures, including counterions, solvent,
symmetry, and potentially disorder. The original files are unchanged. Isolated,
graph-mapped heavy-atom references for all eight literature cases are under
`../data/references/`; the extraction audit is in `../data/references.json`.

A01-A05 and A07-A08 were downloaded from the publisher's official `sendcif`
endpoint. A06 is COD entry 2202758, which cites the original paper. Preserve
the embedded source notices and cite the original articles when using the data.
Article text and figures are not included.

`architector-conda.json`, `xtb-python-conda.json`, and
`architector-upstream.json` record the installation metadata retrieved during
setup. The inspected upstream revision is
`de3af7656096ea7c5a039a970585155327e253a8`. `io_process_input.py.txt` and
`upstream-environment.yml` are upstream API/installation snapshots, covered by
`ARCHITECTOR_LICENSE.txt`; the installed comparator is the conda-forge 0.0.10
package pinned by `../pixi.lock`, rather than a checkout of upstream main.
