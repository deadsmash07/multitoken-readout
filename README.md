# Reading Multi-Token Answers from a Single Hidden State

Code, data and results for the paper *Reading Multi-Token Answers from a Single Hidden State* (anonymous submission).

We read one hidden state of a language model with an identity Patchscope: the state is written into a short copy prompt and the model generates the readout. Writing the state into an **early layer** of the copy prompt, rather than at the layer it was read from, more than doubles exact recovery of multi-token answers on Qwen3-14B (21% to 47% of held-out two-word names), with item-level controls at zero.

## Layout

```
sjlens/                 library: minimal Qwen3 forward with activation patching, J-lens utilities,
                        WorkspaceBench scorer (sjlens/wsbench_regex.py), evaluation helpers
scripts/x3_patchscope.py    the Patchscope readers (copy prompt, continuation prompt), all layers,
                            controls (shuffled, cross-category, random) and repeated insertion
scripts/x8_wsbench_items.py WorkspaceBench runs (token lenses and Patchscope readers, bench regex)
scripts/x1_gap1_patch.py, x4_probe.py, x6_future_probe.py, e12_*.py, e18_*.py, ...
                            linear readers of the second token (step readout, beam decoder, probe)
scripts/make_fresh_items.py builds the fresh item set
data/                   item sets (two-word names, sub-word strings, fresh set, bridge and neutral items);
                        data/wsbench/ holds the WorkspaceBench item banks (see data/wsbench/NOTICE.md)
contexts/               generic token contexts used to fit the linear readers
runs/<name>/results.json    result file behind every number in the paper (full configuration inside)
paper/analysis/         scripts that recompute the paper's tables and intervals from runs/
paper/figures/src/      figure scripts
PROVENANCE.md           maps every number in the paper and appendix to its result file
jobs/                   the run commands, one run per line (name, then the arguments to python)
tests/                  CPU tests on a tiny random Qwen3 model
```

## Setup

Python 3.11 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

Models are loaded from the Hugging Face Hub (`Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-14B`, `Qwen/Qwen3.6-27B`). The 27B uses hybrid attention and additionally needs `flash-linear-attention`. The token-lens arms use the released J-lens files for each model (Gurnee et al. 2026). One GPU with 80 GB of memory runs every experiment; the 1.7B runs on a smaller GPU.

## Reproducing the paper

Every run writes `runs/<name>/results.json`, which stores its full configuration and per-item outputs. The commands for each run are in `jobs/`. For example, the main result (Figure 2, Qwen3-14B two-word names, same layer vs. layer 4):

```
python scripts/x3_patchscope.py --model Qwen/Qwen3-14B --dtype bf16-mixed \
  --items data/phrase_items.json --split-file data/phrase_split.json --layers 32 \
  --target-layer same,4 --split heldout --kinds answer --carriers identity3 --n-carriers 8 \
  --modes replace --gen 8 --positions last --controls shuffled,crosscat,random \
  --device cuda --tag w3_B_x3i3_14b_ans_L32
```

Writing the state into every lower layer (Figure 3) uses `--target-layer all4,all32`; the neutral-context control uses `--items data/a26_neutral_items.json --positions firsthop --no-item-gate`. WorkspaceBench runs use `scripts/x8_wsbench_items.py` (see `--help`).

To recompute the tables and intervals in the paper from the shipped result files (CPU only):

```
cd paper/analysis
python provenance.py         # every headline number, with its run and configuration
python paired_clustered.py   # paired gains with intervals clustered by target string; binomial tests
python carriers.py           # per-carrier counts
```

These scripts reproduce the files in `paper/analysis/` exactly.

## Tests

```
python tests/test_wave5_tiny.py
python tests/test_wave4_tiny.py
```

The tests run on CPU with a tiny random Qwen3 model and check patching, the readers, the controls and the benchmark scorer.

## License

The code will be released under an open-source license on publication. The WorkspaceBench item banks in `data/wsbench/` are redistributed under their MIT license (see `data/wsbench/NOTICE.md`).
