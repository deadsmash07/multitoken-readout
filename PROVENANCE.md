# Archive: claim-to-run provenance and full result tables

This file accompanies the anonymised code and data archive. The section "Claim-to-Run Provenance" maps every headline number of the main paper to the results file it was computed from (run directory under `runs/`), the cell it was read in (model, source layer, target block), the carrier, the token budget, the scoring rule, the gate, the exact numerator and denominator recomputed from the item rows, the coverage of each control on that cell, and whether the cell was fixed before the run or chosen after it. `paper/analysis/provenance.py` regenerates those tables from the archive. The other sections hold the full result tables that the technical supplement summarises in prose (per-layer sweeps, per-carrier counts, every benchmark denominator, clustered paired intervals, binomial tests). Registration labels such as R3 or A15 are entry numbers in the registration file, which is also in the archive. Every number is recomputed from the item rows of the named results file.

## Experimental Details


### tab:carrier-text

| copy prompt, eight variants (the state replaces the residual at "?"; eight greedy tokens, cut at ";", "->" or newline) |
|---|
| `Grand Rapids -> Grand Rapids; kaleidoscope -> kaleidoscope; 3817 -> 3817; ?` |
| `Chiang Mai -> Chiang Mai; serendipity -> serendipity; 5926 -> 5926; ?` |
| `Alice Springs -> Alice Springs; ventriloquist -> ventriloquist; 7403 -> 7403; ?` |
| `Milton Keynes -> Milton Keynes; gobbledygook -> gobbledygook; 6158 -> 6158; ?` |
| `Punta Arenas -> Punta Arenas; flabbergasted -> flabbergasted; 4271 -> 4271; ?` |
| `Ann Arbor -> Ann Arbor; pomegranate -> pomegranate; 8539 -> 8539; ?` |
| `Boca Raton -> Boca Raton; Schadenfreude -> Schadenfreude; 2764 -> 2764; ?` |
| `Coffs Harbour -> Coffs Harbour; paraphernalia -> paraphernalia; 9385 -> 9385; ?` |
| continuation prompt, eight variants (the state replaces the residual at the last token; twelve greedy tokens, uncut; the first four are prose prefixes, the last four answer frames) |
| `The committee met on Tuesday to review the proposal, and after a short discussion the chair announced that the` |
| `According to the report published last spring, the most important factor turned out to be the` |
| `She opened the letter slowly and read the first line, which mentioned the` |
| `In the second chapter the author turns to the question of the` |
| `Question: what is it called?\nAnswer: the` |
| `The answer is` |
| `Q: What is its name?\nA:` |
| `The correct term for it is` |

Caption: The two carrier prompts, verbatim. Each line is one variant; the eight variants of a carrier are the eight carriers whose agreement defines the majority and any-of-8 rates. The carrier order here is the order used in the per-carrier counts of Section app:carriers.


### tab:gates

| llrr@{}} model | family | gated | immediate |
|---|---|---|---|
| Qwen3.6-27B | basic readout | 85 | 51 |
|  | typo correction | 34 | 0 |
|  | multihop | 86 | 83 |
| Qwen3-14B | basic readout | 63 | 28 |
|  | typo correction | 25 | 0 |
|  | multihop | 66 | 60 |
| Qwen3-8B | basic readout | 49 | 25 |

Caption: Number of the 100 WorkspaceBench items of each family that pass the plain-prompt gate on each model, and the number whose greedy answer starts with the target (immediate). No typo item is immediate on either model.


## Decision Rules and Outcomes

Each row lists a test, the success and stop lines written down before the run, the result, and the outcome.


### tab:prereg

| test | registered line (pre-named cell) | result | verdict |
|---|---|---|---|
| second piece from a sequence J-lens, given the first piece (step readout) | second piece top-1 >= .50 pass, < .30 kill | 0 of 325 (1.7B), 0 of 285 (14B) given the first piece in the lens top-10 | kill |
| copy prompt, exact string, registered grid of layers and targets (R3) | majority >= .30 at the best layer with shuffled and random <= .05; kill < .05 everywhere on both models; Holm across layers within a hypothesis | 14B two-word .469 (L32 -> block 4), .417 (L36 -> block 4); sub-word .310 (L32 -> block 4, lower bound .263); 1.7B .219 | pass on the 14B (two-word; Holm-adjusted binomial p<10^-5); sub-word by point estimate only (p=.36); 1.7B not pass, not kill |
| control-position fix (Section app:controls) | withdraw any cell whose corrected control exceeds .05 | maximum corrected control .007 | no cell withdrawn |
| bridge entity read at the last token (R5) | first-hop Patchscope beats the last-token dictionary by >= .15 on person bridges; no kill line | copy prompt at the last token .028 majority (n 143) | did not pass; no kill line was registered |
| continuation prompt on basic readout, 14B L32 -> block 4 (A15) | some arm >= .15 immediate and >= .10 gated, shuffled <= .02; kill: every arm <= .05 | 8 of 28, 9 of 63; shuffled 0 | pass (continuation arm only) |
| four controls for the continuation arms (A20) | withdrawn if random or cross-category > .02 or a counted item is produced with no patch | shuffled, random, no-patch 0 of 63; cross-category 0 of 28 decided | confirmed, on the covered items |
| spelling-correction copy prompt on typo items (A17) | >= .20 on 25 gated items; kill <= .04 at every cell | maximum over 9 cells .040 (cut rule); uncut diagnostic up to .20 | kill of the cut-scored carrier |
| 27B decision rule for the benchmark table (A19) | pilot cell L52 -> block 8 >= .10 immediate with shuffled 0; then all-100 basic >= .10 at L52 -> block 8 | pilot 8 of 20; full .14 (14 of 100), shuffled 0 | headline table (a decision rule, not a significance test) |
| repeated insertion through the lower blocks (A21) | contamination if |all-32 - block-4| <= .05 and all-32 - same >= .15; computation if |all-32 - same| <= .05 and block-4 - all-32 >= .15; else unresolved | .206 / .349 / .474 / .469 (same / all-32 / all-4 / block 4) | unresolved, as registered |
| bridge entity read at the first-hop token (A22, part i) | majority exact >= .30 or benchmark any-of-8 >= .30 with controls <= .05, per arm; Holm over the four arms | continuation any-of-8 .867 (26 of 30; Holm-adjusted p<10^-9); exact 0; copy .133 | pass on one arm |
| bridge unit on benchmark multihop items, last 40 positions (A22, part ii) | maximum over positions of the bridge rate >= .20 with shuffled maximum <= .05; the registered kill is joint: (i) <= .05 at every position and carrier *and* (ii) <= .05 | maximum .039 | (ii) below its low-signal line; the joint kill did not fire because (i) passed |
| neutral-context control for the first-hop pass (A26) | entity-recompute if neutral >= two-hop - .10; bridge-beyond-entity if two-hop - neutral >= .20 with both paired intervals excluding 0 | neutral .717 vs .867; differences .200 [.067, .367], .100 [.000, .233] | unresolved; bridge reading not established |
| Qwen3-8B replication at L29 -> block 4 (A23) | copy prompt majority >= .30, controls <= .05, both tracks; Holm over L29 and L32 on two-word | .000 on both tracks at block 4 (carrier echoes); block 8, in the registered grid: .494, .379 | fail at the pre-named cell; block 8 promoted after the fact |
| fresh entity-disjoint items, 14B L32 -> block 4 (A24) | majority >= .30, controls <= .05; replicates iff inside the original interval | two-word .337 (below [.389, .543]); sub-word .169 | two-word passes the line, does not replicate the magnitude; sub-word not pass, not kill |
| false-alarm certificate on carrier agreement (A25) | |FPR - q| <= 2 s.e.\ at q = .05 and .01 on held-out background records | threshold at the maximum score; FPR 0.000 | pass as registered, vacuous |

Caption: Every registered test that bears on the paper, with the verdict judged against the registered text. The label in parentheses is the entry number in the registration file. "Majority" and "any-of-8" count items for which more than four, or at least one, of the eight carriers emit the exact string; FPR is the false-positive rate and s.e.\ its standard error; Holm refers to Holm's step-down adjustment for multiple tests. Earlier registered lines for the linear probe, the step readout and beam decoder, and the closed-list retrieval (all negative or retired) are summarised in Section app:more; the binomial tests and Holm adjustments are in Section app:clustered.


## Additional Results


### tab:linear

Source comment: Sources: runs/s2_g8_L16 (step readout), runs/hp_heldout_L{18,22} (beam), runs/p2_{A,B,S}_14b (probe), runs/w2B_e14_14b_L32 (closed-list retrieval), runs/w2A_x1_14b_L32_s{0,1,2} (one-position patch). See paper/analysis/provenance.py.

| llrl@{}} reader | statistic | value | control |
|---|---|---|---|
| step readout^* | 2nd top-1 | 1/285 | 0 given 1st |
| beam decoder, 1.7B | recall@10 | 0/60, 0/278 | (L18, L22) |
| probe, sub-word | 2nd top-10 | .149 | shuffled .150 |
| probe, two-word | 2nd top-10 | .046 | shuffled .000 |
| J-lens, two-word | 2nd top-10 | .434 |  |
| probe, 1-token set | 1st top-1 | .23--.61 | (L32, L36) |
| closed list of 488 | top-1 | .777 | chance .002 |
| patch^*, replace | 2nd top-10 | **.395** | same-cat .053 |
| patch^*, add | 2nd top-10 | **.436** | same-cat .127 |
| tangent of add.^* | 2nd top-10 | .073 | same-cat .027 |

Caption: Readers of the second answer piece on Qwen3-14B at layer 32, the paper's read layer (higher is better; other layers and the 1.7B are in Tables tab:step to tab:onepos). "2nd" is the true second answer piece and "1st" the first; "top-k" means the true piece is among the reader's k highest-ranked tokens, and recall@10 is the fraction of items whose exact string is among the decoder's ten outputs. Rows marked ^* receive the true first piece as an oracle prefix. The probe is a ridge regression from h_l to the final residual one position ahead, fit on generic text; on the one-token set it predicts the first answer token, which calibrates it. The closed-list row is given the 488 candidate strings, so it is a diagnostic rather than an open-vocabulary reader. The patch rows write h_l into a generic carrier one position before the first piece, either replacing the carrier's residual or adding to it; the tangent row is the exact directional derivative of the additive patch and is compared with the additive arm.


### tab:step

| llrr@{}} model | layer | n | second piece at top-1 |
|---|---|---|---|
| 1.7B | 6 | 267 | 0 |
|  | 10 | 302 | 1 |
|  | 14, 18, 22, 26 | 325 | 0 |
| 14B | 16 | 285 | 1 |
|  | 24, 32 | 285 | 0 |

Caption: First-order step readout: number of items for which the true second piece is ranked first, given the true first piece, by model and layer.


### tab:probe

| llrrrrrr@{}} model, items | layer | probe | shuffled | cross-category | random | J-lens | logit lens |
|---|---|---|---|---|---|---|---|
| 14B, sub-word | 24 | .146 | .156 | .035 | .029 | .018 | .000 |
|  | 32 | .149 | .150 | .032 | .082 | .029 | .026 |
|  | 36 | .155 | .142 | .029 | .076 | .041 | .053 |
| 14B, two-word | 32 | .046 | .000 | .006 | .000 | .434 | .400 |
|  | 36 | .034 | .006 | .006 | .006 | .457 | .446 |
| 1.7B, sub-word | 14 to 26 | .158 to .176 | .167 to .175 |  |  |  |  |

Caption: Linear future-token probe: fraction of items whose true second piece is in the probe's top ten, with the shuffled, cross-category and random controls and the two token lenses on the same items. The probe is a ridge map from h_l to the final residual one position ahead, fit on generic text. The 1.7B row gives the range over layers 14 to 26.


### tab:decomp

| llrrlrrr@{}} model | layer | same-position / future norm | same-position share | share of the future term (%) |  |  |  |
|---|---|---|---|---|---|---|---|
|  |  | resampled contexts | single documents | of a word's row | next position | positions 1 to 5 | positions 20 and beyond |
| 1.7B | 11 | 0.61 | 0.64 | 0.24 to 0.37 | 9.5 | 27 | 48 |
|  | 17 | 1.23 | 1.12 | 0.42 to 0.59 | 13.1 | 32 | 44 |
|  | 22 | 3.74 | 3.90 | 0.75 to 0.83 | 15.2 | 37 | 40 |
| 14B | 16 | 0.78 |  | 0.23 to 0.44 | 6.7 | 20.5 | 55 |
|  | 24 | 0.99 |  | 0.39 to 0.58 | 11.5 | 29.5 | 46 |
|  | 32 | 2.41 |  | 0.64 to 0.82 |  |  |  |

Caption: Decomposition of the released J-lens into its same-position term and its future terms. The norm ratio is measured at the activation level (on resampled contexts, and on single documents for the 1.7B); the same-position share is the range over nine space-prefixed words; the last three columns distribute the future term over distance from the read position.


### tab:onepos

| lrrrrr@{}} arm | top-10 | top-1 | same-cat. | cross-cat. | random |
|---|---|---|---|---|---|
| replacement | .395 | .301 | .053 / .027 | .003 | .000 |
| additive | .436 |  | .127 | .020 | .000 |
| tangent of additive | .073 |  | .027 |  |  |

Caption: One-position patch on 342 Qwen3-14B sub-word items at layer 32: fraction of items whose true second piece is in the top ten (or ranked first) by change in log-probability, with the same-category (top-10 / top-1 where both were computed), cross-category and random controls. The additive top-10 is 149 of 342 and the tangent top-10 is 25 of 342.


### tab:identity

Source comment: Sources: runs/w3_{B,A}_x3i3_14b_ans_L32 (+ w3c_* corrected controls), runs/w5_fresh_14b_{B,A}_x3i3_L32, runs/w5_8b_{B,A}_x3i3_L29, runs/w3_A_x3i3_1p7b_ans_L22 (+ w3c_A_x3i3_1p7b_ans_L22). Majority-of-8 exact unless stated. Paired-gap intervals: paper/analysis/paired_clustered.py (bootstrap over target strings, 10,000 resamples).

| lrrll@{}} model, items | n | same | early [95% CI] | gap [CI] |
|---|---|---|---|---|
| 14B two-word (orig.) | 175 | .206 | **.469** [.39,.54] | .26 [.18,.34] |
| 14B two-word (fresh) | 175 | .086 | .337 [.27,.41] | .25 [.18,.33] |
| 14B sub-word (orig.) | 342 | .129 | .310 [.26,.36] | .18 [.13,.24] |
| 14B sub-word (fresh) | 431 | .044 | .169 [.14,.20] | .13 [.09,.16] |
| 8B two-word, \tau{=}8 | 160 | .250 | .494 [.41,.57] | .24 [.15,.34] |
| 8B sub-word, \tau{=}8 | 340 | .168 | .379 [.33,.43] | .21 [.16,.27] |
| 1.7B sub-word, \tau{=}8 | 278 | .115 | .219 [.17,.27] | .10 [.06,.15] |

Caption: Exact multi-token recovery from one activation (higher is better), same-layer patch versus early target. The source layer is about 80% of depth (14B L32, 8B L29, 1.7B L22); the early target is block 4 on the 14B and block 8 on the 8B and 1.7B. "gap" is the paired early-minus-same difference with a 95% interval from a bootstrap over target strings. The largest majority rate over the shuffled, cross-category and random controls is .000 in every 14B row, .003 on the 8B sub-word row and .007 on the 1.7B row; token lenses score 0 by construction. Bold marks cells that clear the registered line (majority >= .30, controls <= .05) with a one-sided exact binomial test against .30 that stays significant after the Holm correction for multiple tests; the sub-word original and fresh two-word cells clear the line by point estimate only. The 8B block-8 cells were chosen after the pre-named block-4 cell failed.


### tab:layers

Source comment: Sources: runs/w3_{A,B}_x3i3_14b_ans_L{32,36}, runs/w3_A_x3i3_1p7b_ans_L{22,25}, runs/w3_S_x3i3_14b; controls from runs/w3c_*.

| llrlll@{}} model, items | source -> target | n | majority [95% CI] | any-of-8 | first piece |
|---|---|---|---|---|---|
| 14B, two-word | L32 -> same | 175 | .206 [.143, .263] | .417 | .954 |
|  | L32 -> block 4 | 175 | .469 [.389, .543] | .709 | .811 |
|  | L36 -> same | 175 | .194 [.131, .251] | .417 | .960 |
|  | L36 -> block 4 | 175 | .417 [.337, .491] | .680 | .891 |
| 14B, sub-word | L32 -> same | 342 | .129 [.094, .167] | .284 | .916 |
|  | L32 -> block 4 | 342 | .310 [.263, .357] | .518 | .670 |
|  | L36 -> same | 342 | .114 [.082, .149] | .263 | .967 |
|  | L36 -> block 4 | 342 | .298 [.249, .345] | .561 | .884 |
| 1.7B, sub-word | L22 -> same / 4 / 8 | 278 | .115 / .205 / .219 [.173, .266] | .241 / .406 / .381 | .930 / .674 / .695 |
|  | L25 -> same / 4 / 8 | 278 | .058 / .169 / .176 | .169 / .288 / .291 | .984 / .652 / .658 |
| 14B, one-token answers | L32 -> same / 4 | 128 | .891 / .789 | .984 / .875 |  |
|  | L36 -> same / 4 | 120 | .908 / .892 | .992 / .950 |  |

Caption: Copy-prompt recovery by source layer and target on held-out items, with corrected controls at or below .007 majority in every cell. "First piece" is the majority rate at which the first token of the string is emitted (first pieces of at least two characters). The early target lowers the first-piece rate and raises the exact rate: it re-derives the string rather than transplanting its first token.


### tab:hard

| llrllll@{}} items | rank of 2nd piece | n | same layer | block 4 |  |  |
|---|---|---|---|---|---|---|
|  |  |  | maj. | any | maj. | any |
| two-word | 1 | 44 | .523 | .636 | .614 | .773 |
|  | 2 to 10 | 35 | .314 | .686 | .629 | .857 |
|  | above 10 (hard) | 96 | .021 | .219 | .344 | .625 |
| sub-word | 1 | 29 | .552 | .828 | .621 | .690 |
|  | 2 to 10 | 93 | .204 | .505 | .398 | .484 |
|  | above 10 (hard) | 220 | .041 | .118 | .232 | .509 |

Caption: Exact recovery on Qwen3-14B at layer 32, split post hoc by the rank of the true second piece in a clean carrier given only the first piece. Majority (maj.) and any-of-8 (any) rates for the same-layer patch and the block-4 target; shuffled controls are 0 on every subset.


### tab:categories

Source comment: Source: runs/w3_A_x3i3_14b_ans_L36.

| lrll@{}} category | n | majority | any-of-8 |
|---|---|---|---|
| country of a capital | 16 | .88 | 1.00 |
| animals | 17 | .76 | .88 |
| languages | 8 | .62 | .62 |
| number words | 14 | .50 | .71 |
| arithmetic | 20 | .35 | .60 |
| surnames | 17 | .35 | .82 |
| capitals | 28 | .29 | .79 |
| Spanish, German, French translations |  | .20 to .28 | .39 to .68 |
| years |  | .07 | .27 |
| formulas |  | .00 | .43 |

Caption: Copy-prompt recovery by item category, Qwen3-14B sub-word items at L36 -> block 4, split post hoc. The translation row gives the range over the three languages.


### fig:mechanism

Figure file: `figures/fig3_mechanism.pdf`

Caption: Repeated insertion of the source state. Qwen3-14B, layer 32, 175 two-word items, 95% item bootstrap intervals. "Every block <= k" writes the state at the patch position at the embedding and after each block up to k, so no representation of the carrier's own token survives below k; the patched position's trajectory, the lower-layer keys and values that later positions attend to, and the number of times the state is presented also change. The copy prompt is scored by exact majority, the continuation prompt by whether the string is a prefix of its output.


### tab:bench

Source comment: Sources: runs/w5_27b_full_{basic,typo,multihop}/results.json (all 100 items, bench regex, HF-hooks backend, bf16). Published rows: WorkspaceBench repository at commit 92d763e (evals/*/README.md), as transcribed in docs/PRIOR_WORK_PATCHSCOPES.md 2.5; the typo_mt token-lens cells (.17, .20) were verified directly against evals/typo_mt/README.md at that commit on 30 Sep 2026.

| lrrr@{}} reader on Qwen3.6-27B, all 100 items | basic | typo | multihop |
|---|---|---|---|
| *ours (prompt-blind, training-free, no summariser)* |  |  |  |
| continuation, pre-named cell (L52 -> block 8) | **.14** | .21 | .02 |
| continuation, block 4, union over L52 and L56^+ | .22 | **.33** | .06 |
| continuation, same layer | .06 | .00 | .01 |
| copy prompt, block 4 (uncut in parentheses) | .04 (.06) | .00 (.09) | .01 (.01) |
| J-lens, raw regex | .00 | .00 | .00 |
| logit lens, raw regex | .00 | .00 | .00 |
| shuffled-state control (union cell) | .00 | .00 | .00 |
| *published on the same model and items* |  |  |  |
| J-lens, raw regex | .00 | .00 | .00 |
| J-lens, LLM-summarised | .24 | .17 | .08 |
| logit lens, LLM-summarised | .20 | .20 | .08 |
| natural-language autoencoder (trained) | .53 | .75 | .22 |
| oracle lens (trained) | .71 | .83 | .32 |
| prompt-only (sees the prompt, no activation) | .59 | .67 | .69 |

Caption: Pass rate over all 100 items of each family under the benchmark's regular expression on the benchmark's own model (higher is better). ^+Selected after the run: the best of three target blocks, counting an item if it passes at either read layer (per-layer counts and a stricter single-sample recount are in Section app:carriers). Counting only the primary answer string (no aliases) the union cell is .16; on typo, 26 of the 33 passes are items the 27B does not itself correct within 48 tokens (7 of 34 gated items pass). Our raw generations and the published summarised token bags are different readout budgets, so proximity in this table is numerical, not equivalence. The gated and immediate denominators are in Table tab:benchfull and per-carrier counts in Section app:carriers.


### tab:benchfull

Source comment: Sources: runs/w5_27b_full_*, runs/w5_x3c_14b_{basic,typo}_all, runs/w4_e1_14b_basic_cl, runs/w4_e1_14b_multihop_cl, runs/w5_ctrl_14b_basic_typo, runs/w5_8b_x3c_basic.

| llrrrl@{}} model, family | arm | all 100 [95% CI] | gated | immediate | per read layer |
|---|---|---|---|---|---|
| 27B basic | continuation, block 4 (union) | .22 [.14, .30] | .247 (85) | .294 (51) | L52 .17, L56 .15 |
|  | continuation, block 8 | .19 [.11, .27] | .212 | .275 | L52 .14, L56 .14 |
|  | continuation, block 16 | .13 [.06, .20] | .141 | .157 | L52 .05, L56 .12 |
|  | continuation, same layer | .06 [.02, .11] | .071 | .078 | L52 .05, L56 .04 |
|  | copy prompt, block 4 (cut / uncut) | .04 [.01, .08] / .06 | .047 | .059 | L52 .01, L56 .04 |
|  | primary string only, block 4 | .16 | .176 | .235 | L52 .12, L56 .12 |
| 27B typo | continuation, block 4 | .33 [.24, .43] | .206 (34) | -- | L52 .25, L56 .24 |
|  | continuation, block 8 | .28 [.20, .38] | .176 | -- | L52 .21, L56 .17 |
|  | continuation, block 16 | .18 [.11, .26] | .176 | -- | L52 .12, L56 .14 |
|  | copy prompt, block 4 (cut / uncut) | .00 / .09 |  |  | uncut L52 .05, L56 .09 |
| 27B multihop | continuation, block 4 | .06 [.02, .11] | .058 (86) | .060 (83) | L52 .02, L56 .05 |
|  | units: final answer / bridge 1 / bridge 2 | .35 / .06 / .094 |  |  |  |
|  | continuation, block 8 | .03 [.00, .07] | .035 | .036 | L52 .02, L56 .02 |
| 14B basic | continuation, block 4 | .14 [.08, .21] | .190 (63) | .393 (28) | L32 .11, L36 .09, L24 .00 |
|  | continuation, block 8 | .15 [.08, .22] | .222 | .429 | L32 .12, L36 .11, L24 .00 |
|  | continuation, same layer | .03 | .048 | .071 |  |
|  | copy prompt, block 4 | .03 | .048 | .107 |  |
|  | primary string only, L32 -> block 4 |  | .095 | .179 |  |
| 14B typo | continuation, block 8 | .22 [.14, .30] | .360 (25) | -- | L24 .18, L32 .13, L36 .05 |
|  | continuation, block 4 | .09 | .200 | -- |  |
| 14B multihop (gated) | continuation, block 4 |  | .015 (66) |  | final-answer unit .439 |
| 8B basic | continuation, block 8 |  | .122 (49) | .240 (25) | L29 .102, L32 .082 |
|  | continuation, block 4 |  | .020 | .040 |  |
|  | copy prompt, block 8 |  | .061 | .120 |  |

Caption: Benchmark pass rates by denominator (all 100, with a 95% item-bootstrap interval; gated with its n in parentheses; gated-immediate) under the benchmark's regex and any-layer rule; the last column restricts the pass to one read layer. Every J-lens, logit-lens and shuffled-control cell is .00. On the 14B the random-vector and no-patch controls are 0 of 63 (basic) and 0 of 25 (typo); the cross-category control is 0 of 28 decided basic items and 0 of 16 decided typo items, the rest having no cross-category partner. The 14B L32 -> block 4 basic cell (9 of 63, 8 of 28) was reproduced exactly by the all-100 run, a determinism check under greedy decoding rather than a replication.


### tab:seeded

| lrr@{}} carrier variant | gated (63) | immediate (28) |
|---|---|---|
| seeded, logit lens top-5 | 5 | 4 |
| seeded, J-lens top-5 | 3 | 2 |
| copy prompt, long exemplars | 2 | 2 |

Caption: Carrier variants on the WorkspaceBench basic readout, Qwen3-14B at L32 -> block 4: number of gated and immediate items passed under the benchmark's rule. The seeded rows are the continuation prompt seeded with a token lens's five top first tokens; the last row is the copy prompt with three- to four-word exemplars.


### tab:bridges

| llrrrl@{}} arm | read position | n | exact maj. | exact any | benchmark any-of-8 [95% CI] |
|---|---|---|---|---|---|
| copy, same layer | last token | 143 | .007 | .007 |  |
| copy, block 4 | last token | 143 | .028 | .049 |  |
| copy | first-hop token | 30 | 0 | 0 | .133 |
| continuation, block 4 | first-hop token | 30 |  |  | .867 [.733, .967] |
| continuation, block 4 | first-hop token +1 | 30 |  |  | .167 |
| continuation, block 4 | first-hop token -1 | 30 |  |  | .000 |
| continuation, block 4 | last token | 143 |  |  | .196 |
| continuation, shuffled / random | first-hop token | 23 / 30 |  |  | 0 / 0 |
| continuation, neutral sentence | entity token | 60 |  |  | .717 [.600, .817] |
| copy, neutral sentence | entity token | 60 |  |  | .500 |
| fresh bridges, every arm | every position | 123 | 0 |  |  |
| fresh bridges, continuation | end of description | 123 |  |  | .203 [.138, .276] |

Caption: Bridge-entity recovery on Qwen3-14B two-hop items at layer 32. The exact columns use the cut-then-exact rule with majority and any-of-8; the last column uses the benchmark's word-boundary matcher with any-of-8. The first-hop continuation pass is 26 of 30 items. The neutral sentence places the same 30 entities in two templates that ask nothing about the bridge relation, which pass 20 of 30 and 23 of 30. The fresh bridge rate is read at the end of the first-hop description rather than at a curated first-hop token, so it is not like-for-like with the curated items.


## Random Qualitative Samples


### tab:samples

Source comment: Source: runs/w3_B_x3i3_14b_ans_L32/results.json rows; seed-0 random.sample over sorted item ids.

| lll@{}} held-out item | same layer | block 4 |
|---|---|---|
| Edward Jenner | 0/8 `Edward -> ?` | **8/8** |
| Isaac Newton | 2/8 `Isaac ->` | 2/8 |
| one hundred | 0/8 `100 -> ? 1` | 0/8 `100 -> 10` |
| Marilyn Monroe | 2/8 `born in 1980` | 0/8 `born ->` |
| United Kingdom | 5/8 `United States` | **8/8** |
| Nova Scotia | 7/8 | 6/8 `called` |
| Johannes Kepler | 0/8 `Kepler ->` | 0/8 `Kepler` |
| Mother Teresa | 0/8 `Mother ->` | **8/8** |

Caption: Random qualitative samples. Eight held-out two-word items drawn with a fixed seed, Qwen3-14B, layer 32, copy prompt. Each cell gives the number of the eight carriers that emit the exact string and, when fewer than eight, the start of the most common other output.


## Claim-to-Run Provenance


### tab:prov1

| claim (location) | run | cell; carrier; tokens; rule; gate | k/N | controls | registered or post hoc; note |
|---|---|---|---|---|---|
| 14B two-word (orig.), same layer (Table 2, Fig. 2) | `w3_B_x3i3_14b_ans_L32` | 14B, L32 -> block 32; copy prompt, 8 carriers; 8 greedy, cut at ; -> newline; exact majority (any-of-8 in note); greedy-correct answer items | 36/175 | shuffled 0/171; cross-cat. 0/175 (corrected donor position); random 0/175 | registered (copy-prompt grid); any-of-8 73/175 |
| 14B two-word (orig.), early (Table 2, Fig. 2) | `w3_B_x3i3_14b_ans_L32` | 14B, L32 -> block 4; copy prompt, 8 carriers; 8 greedy, cut at ; -> newline; exact majority (any-of-8 in note); greedy-correct answer items | 82/175 | shuffled 0/171; cross-cat. 0/175 (corrected donor position); random 0/175 | registered (copy-prompt grid; best of {same, 4} at L32); any-of-8 124/175 |
| 14B two-word (fresh), same layer (Table 2, Fig. 2) | `w5_fresh_14b_B_x3i3_L32` | 14B, L32 -> block 32; copy prompt, 8 carriers; 8 greedy, cut at ; -> newline; exact majority (any-of-8 in note); greedy-correct answer items | 15/175 | shuffled 0/175; cross-cat. 0/175; random 0/175 | registered (fresh-item test); any-of-8 38/175 |
| 14B two-word (fresh), early (Table 2, Fig. 2) | `w5_fresh_14b_B_x3i3_L32` | 14B, L32 -> block 4; copy prompt, 8 carriers; 8 greedy, cut at ; -> newline; exact majority (any-of-8 in note); greedy-correct answer items | 59/175 | shuffled 0/175; cross-cat. 0/175; random 0/175 | registered (fresh-item test, pre-named); any-of-8 94/175 |
| 14B sub-word (orig.), same layer (Table 2, Fig. 2) | `w3_A_x3i3_14b_ans_L32` | 14B, L32 -> block 32; copy prompt, 8 carriers; 8 greedy, cut at ; -> newline; exact majority (any-of-8 in note); greedy-correct answer items | 44/342 | shuffled 0/339; cross-cat. 0/342 (corrected donor position); random 0/342 | registered (copy-prompt grid); any-of-8 97/342 |
| 14B sub-word (orig.), early (Table 2, Fig. 2) | `w3_A_x3i3_14b_ans_L32` | 14B, L32 -> block 4; copy prompt, 8 carriers; 8 greedy, cut at ; -> newline; exact majority (any-of-8 in note); greedy-correct answer items | 106/342 | shuffled 0/339; cross-cat. 0/342 (corrected donor position); random 0/342 | registered (copy-prompt grid); any-of-8 177/342 |
| 14B sub-word (fresh), same layer (Table 2, Fig. 2) | `w5_fresh_14b_A_x3i3_L32` | 14B, L32 -> block 32; copy prompt, 8 carriers; 8 greedy, cut at ; -> newline; exact majority (any-of-8 in note); greedy-correct answer items | 19/431 | shuffled 0/429; cross-cat. 0/431; random 0/431 | registered (fresh-item test); any-of-8 57/431 |
| 14B sub-word (fresh), early (Table 2, Fig. 2) | `w5_fresh_14b_A_x3i3_L32` | 14B, L32 -> block 4; copy prompt, 8 carriers; 8 greedy, cut at ; -> newline; exact majority (any-of-8 in note); greedy-correct answer items | 73/431 | shuffled 0/429; cross-cat. 0/431; random 0/431 | registered (fresh-item test, pre-named); any-of-8 184/431 |

Caption: Claim-to-run provenance, exact recovery, one-token calibration and the mechanism arms (main paper Sections 4 and 5), part 1 of 3. Every k/N is recomputed from the item rows of the named results file by `paper/analysis/provenance.py`; the controls column gives the numerator and denominator of each control on the same cell (a cross-category control is decided only on items with a partner).


### tab:prov2

| claim (location) | run | cell; carrier; tokens; rule; gate | k/N | controls | registered or post hoc; note |
|---|---|---|---|---|---|
| 8B two-word, same layer (Table 2, Fig. 2) | `w5_8b_B_x3i3_L29` | 8B, L29 -> block 29; copy prompt, 8 carriers; 8 greedy, cut at ; -> newline; exact majority (any-of-8 in note); greedy-correct answer items | 40/160 | shuffled 0/156; cross-cat. 0/160; random 0/160 | registered (8B replication); any-of-8 83/160 |
| 8B two-word, pre-named early (Table 2, Fig. 2) | `w5_8b_B_x3i3_L29` | 8B, L29 -> block 4; copy prompt, 8 carriers; 8 greedy, cut at ; -> newline; exact majority (any-of-8 in note); greedy-correct answer items | 0/160 | shuffled 0/156; cross-cat. 0/160; random 0/160 | registered (8B replication, pre-named; fails); any-of-8 1/160 |
| 8B two-word, early (Table 2, Fig. 2) | `w5_8b_B_x3i3_L29` | 8B, L29 -> block 8; copy prompt, 8 carriers; 8 greedy, cut at ; -> newline; exact majority (any-of-8 in note); greedy-correct answer items | 79/160 | shuffled 0/156; cross-cat. 0/160; random 0/160 | post hoc (in the 8B replication grid, promoted after block 4 failed); any-of-8 119/160 |
| 8B sub-word, same layer (Table 2, Fig. 2) | `w5_8b_A_x3i3_L29` | 8B, L29 -> block 29; copy prompt, 8 carriers; 8 greedy, cut at ; -> newline; exact majority (any-of-8 in note); greedy-correct answer items | 57/340 | shuffled 0/338; cross-cat. 0/340; random 0/340 | registered (8B replication); any-of-8 128/340 |
| 8B sub-word, early (Table 2, Fig. 2) | `w5_8b_A_x3i3_L29` | 8B, L29 -> block 8; copy prompt, 8 carriers; 8 greedy, cut at ; -> newline; exact majority (any-of-8 in note); greedy-correct answer items | 129/340 | shuffled 1/338; cross-cat. 0/340; random 0/340 | post hoc (as above); any-of-8 194/340 |
| 1.7B sub-word, same layer (Table 2, Fig. 2) | `w3_A_x3i3_1p7b_ans_L22` | 1.7B, L22 -> block 22; copy prompt, 8 carriers; 8 greedy, cut at ; -> newline; exact majority (any-of-8 in note); greedy-correct answer items | 32/278 | shuffled 2/275; cross-cat. 0/278 (corrected donor position); random 0/278 | registered (copy-prompt grid); any-of-8 67/278 |
| 1.7B sub-word, early (Table 2, Fig. 2) | `w3_A_x3i3_1p7b_ans_L22` | 1.7B, L22 -> block 8; copy prompt, 8 carriers; 8 greedy, cut at ; -> newline; exact majority (any-of-8 in note); greedy-correct answer items | 61/278 | shuffled 1/275; cross-cat. 0/278 (corrected donor position); random 0/278 | registered (copy-prompt grid; best of {same, 4, 8}); any-of-8 106/278 |
| one-token answers, copy prompt (Sec. 4 text) | `w3_S_x3i3_14b` | 14B, L32 -> block 32; copy prompt, 8 carriers; 8 greedy, cut; exact majority; greedy-correct one-token items | 114/128 | shuffled 0/127; cross-cat. 0/128; random 0/128 | registered (one-token calibration set) |

Caption: Claim-to-run provenance, exact recovery, one-token calibration and the mechanism arms (main paper Sections 4 and 5), part 2 of 3. Every k/N is recomputed from the item rows of the named results file by `paper/analysis/provenance.py`; the controls column gives the numerator and denominator of each control on the same cell (a cross-category control is decided only on items with a partner).


### tab:prov3

| claim (location) | run | cell; carrier; tokens; rule; gate | k/N | controls | registered or post hoc; note |
|---|---|---|---|---|---|
| one-token answers, copy prompt (Sec. 4 text) | `w3_S_x3i3_14b` | 14B, L32 -> block 4; copy prompt, 8 carriers; 8 greedy, cut; exact majority; greedy-correct one-token items | 101/128 | shuffled 0/127; cross-cat. 0/128; random 0/128 | registered (one-token calibration set) |
| placeholder swap (Sec. 5 text) | `w5_ph_14b_B_i3_swap` | 14B, L32 -> block 32, placeholder x; copy prompt, 8 carriers; 8 greedy, cut; exact majority; greedy-correct two-word items | 31/175 | shuffled 0/171 | registered arm (repeated-insertion test, placeholder swap); early-target reading post hoc |
| placeholder swap (Sec. 5 text) | `w5_ph_14b_B_i3_swap` | 14B, L32 -> block 4, placeholder x; copy prompt, 8 carriers; 8 greedy, cut; exact majority; greedy-correct two-word items | 100/175 | shuffled 0/171 | registered arm (repeated-insertion test, placeholder swap); early-target reading post hoc |
| mechanism, copy prompt, every block <= 32 (Sec. 5, Fig. 3) | `w5_ph_14b_B_i3` | 14B, L32 -> embedding + blocks 0..32; copy prompt, 8 carriers; 8 greedy, cut; exact majority; greedy-correct two-word items | 61/175 | shuffled 0/171; random 0/175 | registered (repeated-insertion test, report-only) |
| mechanism, copy prompt, every block <= 4 (Sec. 5, Fig. 3) | `w5_ph_14b_B_i3` | 14B, L32 -> embedding + blocks 0..4; copy prompt, 8 carriers; 8 greedy, cut; exact majority; greedy-correct two-word items | 83/175 | shuffled 0/171; random 0/175 | registered (repeated-insertion test, report-only) |
| mechanism, continuation, same layer (Sec. 5, Fig. 3) | `w5_ph_14b_B_cont` | 14B, L32 -> block 32; continuation prompt, 8 carriers; 12 greedy, uncut; prefix majority; greedy-correct two-word items | 66/175 | shuffled 0/171; random 0/175 | registered (repeated-insertion test, continuation arms) |
| mechanism, continuation, every block <= 32 (Sec. 5, Fig. 3) | `w5_ph_14b_B_cont` | 14B, L32 -> embedding + blocks 0..32; continuation prompt, 8 carriers; 12 greedy, uncut; prefix majority; greedy-correct two-word items | 120/175 | shuffled 0/171; random 0/175 | registered (repeated-insertion test, continuation arms) |
| mechanism, continuation, every block <= 4 (Sec. 5, Fig. 3) | `w5_ph_14b_B_cont` | 14B, L32 -> embedding + blocks 0..4; continuation prompt, 8 carriers; 12 greedy, uncut; prefix majority; greedy-correct two-word items | 129/175 | shuffled 0/171; random 0/175 | registered (repeated-insertion test, continuation arms) |
| mechanism, continuation, block 4 (Sec. 5, Fig. 3) | `w4_B_x3cont_14b_ans_L32` | 14B, L32 -> block 4; continuation prompt, 8 carriers; 12 greedy, uncut; prefix majority; greedy-correct two-word items | 133/175 | shuffled 0/171; random 0/175 | registered (repeated-insertion test, continuation arms) |

Caption: Claim-to-run provenance, exact recovery, one-token calibration and the mechanism arms (main paper Sections 4 and 5), part 3 of 3. Every k/N is recomputed from the item rows of the named results file by `paper/analysis/provenance.py`; the controls column gives the numerator and denominator of each control on the same cell (a cross-category control is decided only on items with a partner).


### tab:prov4

| claim (location) | run | cell; carrier; tokens; rule; gate | k/N | controls | registered or post hoc; note |
|---|---|---|---|---|---|
| step readout, 2nd piece top-1 (Table 1) | `s2_g8_L16` | 14B, L16 (best of L16/24/32); first-order step readout given the true first piece; n/a; top-1 of the true second piece; gated answer+bridge items | 1/285 | 0 when the lens holds the first piece | registered (step-readout family); pool includes 103 bridge items |
| beam decoder, exact recall (Table 1) | `hp_heldout_L18` | 1.7B, L18; J-lens beam decoder (chosen configuration); beam 10, length 5; exact recall at 1 and 10; held-out sub-word items | 0/60 | n/a | registered configuration chosen on the tuning half; 2 configurations x stop rules on this held-out file; the tuning search (hp_screen_L22, 150 items) has one recall@10 hit in one configuration |
| beam decoder, exact recall (Table 1) | `hp_heldout_L22` | 1.7B, L22; J-lens beam decoder (chosen configuration); beam 10, length 5; exact recall at 1 and 10; held-out sub-word items | 0/278 | n/a | registered configuration chosen on the tuning half; 2 configurations x stop rules on this held-out file; the tuning search (hp_screen_L22, 150 items) has one recall@10 hit in one configuration |
| linear probe, sub-word, 2nd piece top-10 (Table 1) | `p2_A_14b` | 14B, L32; ridge probe h_l -> final residual one position ahead (generic text); n/a; top-10 of the true second piece; greedy-correct answer items | 51/342 | shuffled 51/339; paired diff 0.000 [-0.009, 0.009], McNemar b=1 c=1 p=1.0000 | registered (linear-probe test); J-lens second piece top-10 10/342 |
| linear probe, two-word, 2nd piece top-10 (Table 1) | `p2_B_14b` | 14B, L32; ridge probe h_l -> final residual one position ahead (generic text); n/a; top-10 of the true second piece; greedy-correct answer items | 8/175 | shuffled 0/171; paired diff 0.047 [0.018, 0.082], McNemar b=8 c=0 p=0.0078 | registered (linear-probe test); J-lens second piece top-10 76/175 |

Caption: Claim-to-run provenance, linear and first-order readers (main paper Section 3, Table 1), part 1 of 2. Every k/N is recomputed from the item rows of the named results file by `paper/analysis/provenance.py`; the controls column gives the numerator and denominator of each control on the same cell (a cross-category control is decided only on items with a partner).


### tab:prov5

| claim (location) | run | cell; carrier; tokens; rule; gate | k/N | controls | registered or post hoc; note |
|---|---|---|---|---|---|
| linear probe calibration, first answer token top-1 (Table 1) | `p2_S_14b` | 14B, L32; same ridge probe; n/a; top-1 of the next (first answer) token; one-token items | 29/128 | n/a | registered calibration |
| linear probe calibration, first answer token top-1 (Table 1) | `p2_S_14b` | 14B, L36; same ridge probe; n/a; top-1 of the next (first answer) token; one-token items | 78/128 | n/a | registered calibration |
| one-position patch, replacement (h_l - x_{carrier}) (Table 1) | `w2A_x1_14b_L32_s{0,1,2}` | 14B, L32, one position before the first piece; 64 generic carriers, true first piece appended; n/a; second piece top-10 by change in log-probability; greedy-correct sub-word items | 135/342 | same-category 18/339; cross-category 1/342; random 0/342 | registered (one-position patch family) |
| one-position patch, additive (+h_l) (Table 1) | `w2A_x1_14b_L32_s{0,1,2}` | 14B, L32, one position before the first piece; 64 generic carriers, true first piece appended; n/a; second piece top-10 by change in log-probability; greedy-correct sub-word items | 149/342 | same-category 43/339; cross-category 7/342; random 0/342 | registered (one-position patch family) |
| exact tangent of the additive map (Table 1) | `w2A_x1_14b_L32_s{0,1,2}` | 14B, L32, one position before the first piece; 64 generic carriers, true first piece appended; n/a; second piece top-10 by change in log-probability; greedy-correct sub-word items | 25/342 | same-category 9/339; cross-category 0/342; random 0/342 | registered (one-position patch family) |

Caption: Claim-to-run provenance, linear and first-order readers (main paper Section 3, Table 1), part 2 of 2. Every k/N is recomputed from the item rows of the named results file by `paper/analysis/provenance.py`; the controls column gives the numerator and denominator of each control on the same cell (a cross-category control is decided only on items with a partner).


### tab:prov6

| claim (location) | run | cell; carrier; tokens; rule; gate | k/N | controls | registered or post hoc; note |
|---|---|---|---|---|---|
| 27B basic, pre-named cell (Table 4) | `w5_27b_full_basic` | 27B, L52 -> block 8; continuation prompt, 8 carriers; 12 greedy, uncut; bench regex (units may hit different samples at one layer); all 100 | 14/100 | shuffled 0/100 | registered (27B decision rule, pre-named); primary-string-only 14/100 |
| 27B basic, selected two-layer union (Table 4) | `w5_27b_full_basic` | 27B, L52 or L56 -> block 4 (union over layers); continuation prompt, 8 carriers; 12 greedy, uncut; bench regex (units may hit different samples at one layer); all 100 | 22/100 | shuffled 0/100 | post hoc (best of 3 targets x 2 layers); per layer L52 17/100, L56 15/100; strict single-sample recount 21/100; primary-string-only 16/100 |
| 27B basic, same layer (Table 4) | `w5_27b_full_basic` | 27B, L52 or L56 -> same layer; continuation prompt, 8 carriers; 12 greedy, uncut; bench regex (units may hit different samples at one layer); all 100 | 6/100 | shuffled 0/100 | report-only; primary-string-only 4/100 |
| 27B basic, copy prompt (Table 4) | `w5_27b_full_basic` | 27B, L52 or L56 -> block 4; copy prompt, 8 carriers; 8 greedy, cut; bench regex (units may hit different samples at one layer); all 100 | 4/100 | shuffled 0/100 | report-only; primary-string-only 3/100 |
| 27B typo, block 4 (Table 4) | `w5_27b_full_typo` | 27B, L52 or L56 -> block 4; continuation prompt, 8 carriers; 12 greedy, uncut; bench regex (units may hit different samples at one layer); all 100 | 33/100 | shuffled 0/100 | post hoc (report-only family); 7 of 34 gated items pass; 26 passes are ungated; primary-string-only 33/100 |
| 27B typo, block 8 (Table 4) | `w5_27b_full_typo` | 27B, L52 or L56 -> block 8; continuation prompt, 8 carriers; 12 greedy, uncut; bench regex (units may hit different samples at one layer); all 100 | 28/100 | shuffled 0/100 | post hoc (report-only family); primary-string-only 28/100 |
| 27B multihop, block 4 (Table 4) | `w5_27b_full_multihop` | 27B, L52 or L56 -> block 4; continuation prompt, 8 carriers; 12 greedy, uncut; bench regex (units may hit different samples at one layer); all 100 | 6/100 | shuffled 0/100 | post hoc (report-only family); final-answer unit alone hits on 35/100; bridge units are the required ones; primary-string-only 6/100 |

Caption: Claim-to-run provenance, benchmark, bridges and certificate (main paper Sections 6 and 7), part 1 of 2. Every k/N is recomputed from the item rows of the named results file by `paper/analysis/provenance.py`; the controls column gives the numerator and denominator of each control on the same cell (a cross-category control is decided only on items with a partner).


### tab:prov7

| claim (location) | run | cell; carrier; tokens; rule; gate | k/N | controls | registered or post hoc; note |
|---|---|---|---|---|---|
| 14B basic, gated (registered pass) (Sec. 6 text) | `w5_ctrl_14b_basic_typo` | 14B, L32 -> block 4; continuation prompt, 8 carriers; 12 greedy, uncut; bench regex (units may hit different samples at one layer); gated (48-token plain prompt) | 9/63 | shuffled 0/63; cross-cat. 0/28 decided of 63; random 0/63 decided of 63; no patch 0/63 | registered (continuation hypothesis and its four controls, pre-named); cross-category control decided on 28 of 63 (35 items have no partner); primary-string-only 6/63 |
| 14B basic, immediate (registered pass) (Sec. 6 text) | `w5_ctrl_14b_basic_typo` | 14B, L32 -> block 4; continuation prompt, 8 carriers; 12 greedy, uncut; bench regex (units may hit different samples at one layer); gated, immediate | 8/28 | shuffled 0/28; cross-cat. 0/12 decided of 28; random 0/28 decided of 28; no patch 0/28 | registered (continuation hypothesis, pre-named); primary-string-only 5/28 |
| 14B typo, gated (Sec. 6 text) | `w5_ctrl_14b_basic_typo` | 14B, L32 -> block 8; continuation prompt, 8 carriers; 12 greedy, uncut; bench regex (units may hit different samples at one layer); gated (48-token plain prompt) | 8/25 | shuffled 0/25; cross-cat. 0/16 decided of 25; random 0/25 decided of 25; no patch 0/25 | post hoc (typo family report-only under the continuation hypothesis); cross-category control decided on 16 of 25; primary-string-only 8/25 |
| first-hop bridge, continuation prompt (Sec. 7 text) | `w5_br_14b_B_fh` | 14B, L32 -> block 4, read at the first-hop token; continuation prompt, 8 carriers; 8 greedy, uncut; bench regex any-of-8 (exact majority in note); two-hop items with a located first-hop token | 26/30 | shuffled 0/23; random 0/30 (shuffled n 23) | registered (first-hop bridge test, one of four arms; passes); exact majority 0/30; 17 distinct first-hop cues |
| neutral-context control (Sec. 7 text) | `w5_a26_14b_neutral_v2` | 14B, L32 -> block 4, read at the entity token of a neutral sentence; continuation prompt, 8 carriers; 8 greedy, uncut; bench regex any-of-8; same 30 entities x 2 templates | 43/60 | shuffled 0/60; random 0/60 | registered (neutral-context control; verdict unresolved); templates 20/30 and 23/30 |
| bridge at the last token, copy prompt (Sec. 7 text) | `w5_br_14b_B_fh` | 14B, L32 -> block 4, last token; copy prompt, 8 carriers; 8 greedy, cut; exact majority; gated two-hop items | 4/143 | shuffled 0/143; random 0/143 | registered (last-token bridge test; did not pass; no kill line) |
| benchmark multihop position scan, bridge unit maximum (Sec. 7 text) | `w5_br_14b_mh_scan` | 14B, L32 -> block 4, offset 35 of the last 40 positions; two answer-frame carriers; 12 greedy, uncut; bench regex, bridge-1 unit, maximum over offsets; 66 gated multihop items (eligible n varies by offset) | 2/51 | shuffled maximum 0 | registered (benchmark multihop scan; below its .05 line); final-answer unit at the last token 0.258 |
| certificate: background activations at full agreement (Sec. 7 text) | `w5_e5c_14b_L32` | 14B, L32 -> block 4, generic text; copy prompt, 8 carriers; 8 greedy, cut; all 8 carriers agree; 2,304 background activations | 923/2304 | n/a | registered (certificate; passes as registered, zero coverage) |

Caption: Claim-to-run provenance, benchmark, bridges and certificate (main paper Sections 6 and 7), part 2 of 2. Every k/N is recomputed from the item rows of the named results file by `paper/analysis/provenance.py`; the controls column gives the numerator and denominator of each control on the same cell (a cross-category control is decided only on items with a partner).


## Clustered Paired Intervals and Multiplicity


### tab:clustered

Source comment: Source: paper/analysis/paired_clustered.py (paired_clustered.csv).

| llrrrrrllr@{}} item set | cell | n | targets | same | early | gap | cluster CI | item CI | fav./unf. |
|---|---|---|---|---|---|---|---|---|---|
| 14B two-word (original) | L32/4 | 175 | 165 | 36 | 82 | .263 | [.183, .343] | [.189, .337] | 52 / 6 |
| 14B two-word (fresh) | L32/4 | 175 | 152 | 15 | 59 | .251 | [.178, .326] | [.183, .320] | 46 / 2 |
| 14B sub-word (original) | L32/4 | 342 | 316 | 44 | 106 | .181 | [.126, .238] | [.129, .234] | 78 / 16 |
| 14B sub-word (fresh) | L32/4 | 431 | 388 | 19 | 73 | .125 | [.088, .165] | [.088, .162] | 64 / 10 |
| 8B two-word | L29/8 | 160 | 152 | 40 | 79 | .244 | [.154, .335] | [.156, .331] | 49 / 10 |
| 8B sub-word | L29/8 | 340 | 314 | 57 | 129 | .212 | [.159, .266] | [.162, .262] | 83 / 11 |
| 1.7B sub-word | L22/4 | 278 | 263 | 32 | 57 | .090 | [.046, .135] | [.047, .137] | 34 / 9 |
| 1.7B sub-word | L22/8 | 278 | 263 | 32 | 61 | .104 | [.061, .149] | [.061, .151] | 36 / 7 |

Caption: Paired early-minus-same difference in exact majority recovery, with a 95% percentile interval from a bootstrap over target strings and from a bootstrap over items (10,000 resamples each); "targets" is the number of distinct target strings, "same" and "early" are the recovered counts, and "fav./unf." counts items recovered only at the early target and only at the same layer. The 8B block-8 cells were chosen after the pre-named block-4 cell failed; the 1.7B block-8 cell is the best of its registered target grid; "cell" is source layer / target block.


### tab:repins

| lll@{}} | copy prompt | continuation |
|---|---|---|
| same layer | 36 | 66 |
| all <= 32 | 61 | 120 |
| all <= 4 | 83 | 129 |
| block 4 | 82 | 133 |
| all-32 - same | .143 [.086, .203] | .309 [.222, .395] |
| block 4 - all-32 | .120 [.046, .196] | .074 [-.011, .162] |
| all-4 - block 4 | .006 [-.012, .028] | -.023 [-.057, .006] |
| block 4 - same | .263 [.183, .343] | .383 [.295, .474] |
| gap share, all-32 | .54 [.35, .78] | .81 [.62, 1.02] |

Caption: Repeated-insertion arms on the 175 Qwen3-14B two-word items at layer 32; the copy prompt is scored by exact majority and the continuation prompt by prefix majority. Top: items recovered under each arm ("all <= k" writes the state into every block up to k). Middle: paired differences between arms with 95% intervals from a bootstrap over target strings. Bottom: the share of the same-to-early gap closed by the all-32 arm, with a 95% item-bootstrap interval.


### tab:neutral

| lrll@{}} quantity | est. | cue CI | item CI |
|---|---|---|---|
| difference, template 1 | .200 | [.032, .414] | [.067, .367] |
| difference, template 2 | .100 | [.000, .267] | [.000, .233] |
| difference, mean | .150 | [.019, .323] | [.050, .267] |
| neutral rate | .717 | [.500, .897] |  |
| two-hop rate | .867 | [.680, 1.000] |  |

Caption: Neutral-context control on the 30 first-hop bridge items (17 distinct cues), continuation prompt at L32 -> block 4, benchmark rule with any-of-8. "Difference" is the two-hop rate minus the neutral rate, paired by entity, for each of the two neutral templates and for their mean. Intervals are 95% percentile bootstrap intervals over cues and over items.


### tab:binom

Source comment: Source: paper/analysis/paired_clustered.py (binomial_tests.csv).

| llrl@{}} family | cell | k/n | Holm p |
|---|---|---|---|
| 14B two-word grid | L32 -> 4 | 82/175 | 8x10^-6 |
|  | L36 -> 4 | 73/175 | .002 |
|  | L32, L36 same | 36, 34 | 1 |
| 14B sub-word grid | L32 -> 4 | 106/342 | 1 (raw .36) |
|  | L36 -> 4 | 102/342 | 1 |
| fresh two-word | L32 -> 4 | 59/175 | .16 |
| fresh sub-word | L32 -> 4 | 73/431 | 1 |
| 8B two-word | L29, L32 -> 4 | 0, 4 of 160 | 1 |
|  | L29 -> 8^++ | 79/160 | 7x10^-7 |
| first-hop bridge | cont., bench any | 26/30 | 7x10^-10 |
|  | three other arms | <= 4/30 | 1 |

Caption: Exact one-sided binomial tests against the registered pass line of .30, Holm-adjusted within each family of cells declared together. ^++Chosen after the pre-named cell failed.


## Carrier-Wise Counts


### tab:carriers-ours

Source comment: Source: paper/analysis/carriers.py (carriers_ours.csv). Carrier order as listed in Section A.

| llrlrrrr@{}} cell | rule | n | per-carrier hits (8) | maj. | any | uncut maj. | uncut any |
|---|---|---|---|---|---|---|---|
| 14B two-word (orig.), same layer | cut exact | 175 | 47 37 44 46 40 44 42 30 | 36 | 73 | 40 | 77 |
| 14B two-word (orig.), block 4 | cut exact | 175 | 97 64 98 93 87 90 85 64 | 82 | 124 | 84 | 127 |
| 14B two-word (fresh), same layer | cut exact | 175 | 18 22 17 17 16 21 20 11 | 15 | 38 | 16 | 40 |
| 14B two-word (fresh), block 4 | cut exact | 175 | 69 44 67 73 63 69 67 48 | 59 | 94 | 61 | 97 |
| 14B sub-word (orig.), same layer | cut exact | 342 | 51 43 62 44 60 57 50 51 | 44 | 97 | 45 | 97 |
| 14B sub-word (orig.), block 4 | cut exact | 342 | 120 82 114 100 124 128 139 114 | 106 | 177 | 138 | 204 |
| 14B sub-word (fresh), same layer | cut exact | 431 | 20 16 35 27 30 24 24 17 | 19 | 57 | 21 | 60 |
| 14B sub-word (fresh), block 4 | cut exact | 431 | 86 56 103 77 122 94 113 83 | 73 | 184 | 106 | 215 |
| 8B two-word, same layer | cut exact | 160 | 40 34 57 46 42 39 59 36 | 40 | 83 | 42 | 84 |
| 8B two-word, block 8 | cut exact | 160 | 67 61 96 86 91 80 96 81 | 79 | 119 | 82 | 120 |
| 8B sub-word, same layer | cut exact | 340 | 65 56 76 57 73 66 72 62 | 57 | 128 | 58 | 131 |
| 8B sub-word, block 8 | cut exact | 340 | 116 129 138 123 130 111 144 125 | 129 | 194 | 168 | 229 |
| 1.7B sub-word, same layer | cut exact | 278 | 34 35 45 28 45 36 38 37 | 32 | 67 | 34 | 70 |
| 1.7B sub-word, block 8 | cut exact | 278 | 67 67 69 54 68 71 76 65 | 61 | 106 | 101 | 161 |
| 14B two-word (orig.), cont., same layer | prefix | 175 | 72 70 71 50 71 83 73 62 | 66 | 119 | 65 | 117 |
| 14B two-word (orig.), cont., block 4 | prefix | 175 | 129 112 125 112 129 138 134 133 | 133 | 145 | 133 | 150 |
| 14B two-word (fresh), cont., block 4 | prefix | 175 | 106 100 100 82 114 122 113 121 | 112 | 140 | 116 | 147 |

Caption: Our items: hits per carrier under the registered rule (cut-then-exact for the copy prompt, prefix for the continuation prompt), the majority-of-8 and any-of-8 counts recomputed from the stored generations (they reproduce the stored votes exactly), and the same counts under the benchmark's uncut word-boundary matcher. Carrier order is that of Section app:details.


### tab:carriers-bench

Source comment: Source: paper/analysis/carriers.py (carriers_bench.csv).

| lrrlrrrrr@{}} cell | n | pass | per-carrier passes (8) | prose union | frame union | strict | primary only | per layer |
|---|---|---|---|---|---|---|---|---|
| 27B basic, block 4 | 100 | 22 | 3 3 6 7 13 14 10 12 | 12 | 22 | 21 | 16 | L52 17, L56 15 |
| 27B basic, block 8 | 100 | 19 | 1 3 5 5 7 11 8 14 | 10 | 18 | 19 | 14 | L52 14, L56 14 |
| 27B typo, block 4 | 100 | 33 | 0 0 1 1 1 3 3 29 | 1 | 33 | 33 | 33 | L52 25, L56 24 |
| 27B typo, block 8 | 100 | 28 | 1 0 0 2 1 3 2 27 | 3 | 28 | 28 | 28 | L52 21, L56 17 |
| 27B multihop, block 4 | 100 | 6 | 0 3 2 0 1 3 2 1 | 4 | 3 | 6 | 6 | L52 2, L56 5 |
| 14B basic (gated), block 4 | 63 | 9 | 3 1 2 0 7 4 4 2 | 3 | 9 | 9 | 6 | L32 9 |
| 14B typo (gated), block 8 | 25 | 8 | 0 1 1 1 3 0 3 7 | 1 | 8 | 8 | 8 | L32 8 |

Caption: Benchmark cells, continuation prompt: passes when each carrier's output is scored alone under the implemented contract (any tested layer), the unions of the four prose prefixes and of the four answer frames (units may combine across carriers, as in the contract), the implemented count ("pass"), the strict recount that requires every required unit inside one sample at one layer, the primary-string-only count (no aliases), and the count at each read layer alone. The strict recount removes one basic-readout item (`brh-id-chengho`, whose two required units hit different samples at L52).

