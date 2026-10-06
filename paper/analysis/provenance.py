"""Claim-to-run provenance table: every headline number in the main text, with the run tag, model, source layer,
target block, carrier, token budget, scoring rule (cut / uncut / prefix / bench regex), gate, exact numerator and
denominator recomputed from the stored rows, control coverage, and whether the cell was registered before the run
or chosen after it. Writes provenance.json / .csv and a LaTeX fragment (provenance_table.tex) for the supplement.
Usage: python paper/analysis/provenance.py"""
import json
from common import ROOT, load, block, rows, bench_hits, write_json, write_csv

C = []  # claim rows


def add(claim, where, run, cell, carrier, tokens, rule, gate, k, n, controls, status, note=""):
    C.append({"claim": claim, "where": where, "run": run, "cell": cell, "carrier": carrier, "tokens": tokens, "rule": rule, "gate": gate,
              "k": k, "n": n, "rate": (k / n if n else None), "controls": controls, "status": status, "note": note})


def cnt(tag, arm, metric="vote_exact", kind="answer", control="none", position_rule=None):
    rs = rows(tag, arm, control=control, kind=kind, position_rule=position_rule)
    return sum(int(r[metric]) for r in rs), len(rs)


def ctrl_str(tag, arm, metric="vote_exact", kind="answer", ctrls=("shuffled", "crosscat", "random"), position_rule=None):
    parts = []
    for c in ctrls:
        k, n = cnt(tag, arm, metric, kind, c, position_rule)
        if n:
            parts.append(f"{c} {k}/{n}")
    return "; ".join(parts)


# ---- Section 4, Table 2 / Figure 2: exact recovery -------------------------------------------------------------
ID = [("14B two-word (orig.), same layer", "w3_B_x3i3_14b_ans_L32", "identity3:replace", "14B, L32 -> block 32", "registered (R3 grid)", "w3c_B_x3i3_14b_ans_L32"),
      ("14B two-word (orig.), early", "w3_B_x3i3_14b_ans_L32", "identity3:replace:t4", "14B, L32 -> block 4", "registered (R3 grid; best of {same, 4} at L32)", "w3c_B_x3i3_14b_ans_L32"),
      ("14B two-word (fresh), same layer", "w5_fresh_14b_B_x3i3_L32", "identity3:replace", "14B, L32 -> block 32", "registered (A24)", None),
      ("14B two-word (fresh), early", "w5_fresh_14b_B_x3i3_L32", "identity3:replace:t4", "14B, L32 -> block 4", "registered (A24, pre-named)", None),
      ("14B sub-word (orig.), same layer", "w3_A_x3i3_14b_ans_L32", "identity3:replace", "14B, L32 -> block 32", "registered (R3 grid)", "w3c_A_x3i3_14b_ans_L32"),
      ("14B sub-word (orig.), early", "w3_A_x3i3_14b_ans_L32", "identity3:replace:t4", "14B, L32 -> block 4", "registered (R3 grid)", "w3c_A_x3i3_14b_ans_L32"),
      ("14B sub-word (fresh), same layer", "w5_fresh_14b_A_x3i3_L32", "identity3:replace", "14B, L32 -> block 32", "registered (A24)", None),
      ("14B sub-word (fresh), early", "w5_fresh_14b_A_x3i3_L32", "identity3:replace:t4", "14B, L32 -> block 4", "registered (A24, pre-named)", None),
      ("8B two-word, same layer", "w5_8b_B_x3i3_L29", "identity3:replace", "8B, L29 -> block 29", "registered (A23)", None),
      ("8B two-word, pre-named early", "w5_8b_B_x3i3_L29", "identity3:replace:t4", "8B, L29 -> block 4", "registered (A23, pre-named; fails)", None),
      ("8B two-word, early", "w5_8b_B_x3i3_L29", "identity3:replace:t8", "8B, L29 -> block 8", "post hoc (in the A23 grid, promoted after t4 failed)", None),
      ("8B sub-word, same layer", "w5_8b_A_x3i3_L29", "identity3:replace", "8B, L29 -> block 29", "registered (A23)", None),
      ("8B sub-word, early", "w5_8b_A_x3i3_L29", "identity3:replace:t8", "8B, L29 -> block 8", "post hoc (as above)", None),
      ("1.7B sub-word, same layer", "w3_A_x3i3_1p7b_ans_L22", "identity3:replace", "1.7B, L22 -> block 22", "registered (R3 grid)", "w3c_A_x3i3_1p7b_ans_L22"),
      ("1.7B sub-word, early", "w3_A_x3i3_1p7b_ans_L22", "identity3:replace:t8", "1.7B, L22 -> block 8", "registered (R3 grid; best of {same, 4, 8})", "w3c_A_x3i3_1p7b_ans_L22")]
for claim, tag, arm, cell, status, ctag in ID:
    k, n = cnt(tag, arm)
    ka, _ = cnt(tag, arm, "any_exact")
    ctrls = ctrl_str(ctag or tag, arm)
    if ctag:  # the corrected shuffled / cross-category rows live in the w3c_* rerun; the random rows were never affected
        ctrls = ctrl_str(ctag, arm, ctrls=("shuffled", "crosscat")) + " (corrected donor position); " + ctrl_str(tag, arm, ctrls=("random",))
    add(claim, "Table 2, Fig. 2", tag, cell, "copy prompt, 8 carriers", "8 greedy, cut at ; -> newline", "exact majority (any-of-8 in note)", "greedy-correct answer items", k, n, ctrls, status, f"any-of-8 {ka}/{n}")

# ---- one-token answers, hard split, placeholder swap ----------------------------------------------------------
for arm, cell in (("identity3:replace", "14B, L32 -> block 32"), ("identity3:replace:t4", "14B, L32 -> block 4")):
    k, n = cnt("w3_S_x3i3_14b", arm)
    add("one-token answers, copy prompt", "Sec. 4 text", "w3_S_x3i3_14b", cell, "copy prompt, 8 carriers", "8 greedy, cut", "exact majority", "greedy-correct one-token items", k, n, ctrl_str("w3_S_x3i3_14b", arm), "registered (R3 calibration set)")
for arm, cell in (("identity3:replace:ph=x", "14B, L32 -> block 32, placeholder x"), ("identity3:replace:t4:ph=x", "14B, L32 -> block 4, placeholder x")):
    k, n = cnt("w5_ph_14b_B_i3_swap", arm)
    add("placeholder swap", "Sec. 5 text", "w5_ph_14b_B_i3_swap", cell, "copy prompt, 8 carriers", "8 greedy, cut", "exact majority", "greedy-correct two-word items", k, n, ctrl_str("w5_ph_14b_B_i3_swap", arm, ctrls=("shuffled", "random")), "registered arm (A21 e); early-target reading post hoc")

# ---- Section 5 mechanism --------------------------------------------------------------------------------------
for claim, tag, arm, cell in (("mechanism, copy prompt, every block <= 32", "w5_ph_14b_B_i3", "identity3:replace:all32", "14B, L32 -> embedding + blocks 0..32"),
                              ("mechanism, copy prompt, every block <= 4", "w5_ph_14b_B_i3", "identity3:replace:all4", "14B, L32 -> embedding + blocks 0..4")):
    k, n = cnt(tag, arm)
    add(claim, "Sec. 5, Fig. 3", tag, cell, "copy prompt, 8 carriers", "8 greedy, cut", "exact majority", "greedy-correct two-word items", k, n, ctrl_str(tag, arm, ctrls=("shuffled", "random")), "registered (A21; report-only mechanism test)")
for claim, tag, arm, cell in (("mechanism, continuation, same layer", "w5_ph_14b_B_cont", "continuation:replace", "14B, L32 -> block 32"),
                              ("mechanism, continuation, every block <= 32", "w5_ph_14b_B_cont", "continuation:replace:all32", "14B, L32 -> embedding + blocks 0..32"),
                              ("mechanism, continuation, every block <= 4", "w5_ph_14b_B_cont", "continuation:replace:all4", "14B, L32 -> embedding + blocks 0..4"),
                              ("mechanism, continuation, block 4", "w4_B_x3cont_14b_ans_L32", "continuation:replace:t4", "14B, L32 -> block 4")):
    k, n = cnt(tag, arm, "vote_prefix")
    add(claim, "Sec. 5, Fig. 3", tag, cell, "continuation prompt, 8 carriers", "12 greedy, uncut", "prefix majority", "greedy-correct two-word items", k, n, ctrl_str(tag, arm, "vote_prefix", ctrls=("shuffled", "random")), "registered (A21 continuation arms)")

# ---- Section 3 linear readers ---------------------------------------------------------------------------------
s2 = block("s2_g8_L16")
add("step readout, 2nd piece top-1", "Table 1", "s2_g8_L16", "14B, L16 (best of L16/24/32)", "first-order step readout given the true first piece", "n/a", "top-1 of the true second piece", "gated answer+bridge items", round(s2["p2_top1_z"] * s2["n"]), s2["n"], "0 when the lens holds the first piece", "registered (R2/step family); pool includes 103 bridge items")
for tag, L in (("hp_heldout_L18", 18), ("hp_heldout_L22", 22)):
    b = block(tag)["beam_summary"]["chosen_eb16dnorm_hc_lens"]["as_run"]
    add("beam decoder, exact recall", "Table 1", tag, f"1.7B, L{L}", "J-lens beam decoder (chosen configuration)", "beam 10, length 5", "exact recall at 1 and 10", "held-out sub-word items", round(b["recall10"] * b["n"]), b["n"], "n/a", "registered configuration chosen on the tuning half", f"{len(block(tag)['beam_summary'])} configurations x stop rules on this held-out file; the tuning search (hp_screen_L22, 150 items) has one recall@10 hit in one configuration")
for tag, lab in (("p2_A_14b", "sub-word"), ("p2_B_14b", "two-word")):
    s = block(tag, 32)["summary"]
    n = s["none"]["n"]
    k = round(s["none"]["rank_s2_probe2"]["top10"] * n)
    ns = s["shuffled"]["n"]
    ks = round(s["shuffled"]["rank_s2_probe2"]["top10"] * ns)
    pr = s["shuffled"]["paired_s2_probe2_top10"]
    add(f"linear probe, {lab}, 2nd piece top-10", "Table 1", tag, "14B, L32", "ridge probe h_l -> final residual one position ahead (generic text)", "n/a", "top-10 of the true second piece", "greedy-correct answer items", k, n, f"shuffled {ks}/{ns}; paired diff {pr['diff']:.3f} [{pr['lo']:.3f}, {pr['hi']:.3f}], McNemar b={pr['mcnemar']['b']} c={pr['mcnemar']['c']} p={pr['mcnemar']['p']:.4f}", "registered (A11 P2)", f"J-lens second piece top-10 {round(s['none']['rank_s2_jlens']['top10']*n)}/{n}")
for L in (32, 36):
    s = block("p2_S_14b", L)["summary"]["none"]
    add("linear probe calibration, first answer token top-1", "Table 1", "p2_S_14b", f"14B, L{L}", "same ridge probe", "n/a", "top-1 of the next (first answer) token", "one-token items", round(s["rank_s1_probe1"]["top1"] * s["n"]), s["n"], "n/a", "registered calibration")
tot = {}
for sh in ("s0", "s1", "s2"):
    for r in block(f"w2A_x1_14b_L32_{sh}")["rows"]:
        key = (r["arm"], r["control"])
        d = tot.setdefault(key, [0, 0])
        d[0] += int(r["rank_s2"] <= 10)
        d[1] += 1
for arm, lab in (("finite:replace:rep:plain", "one-position patch, replacement (h - x_carrier)"), ("finite:add:a1:plain", "one-position patch, additive (+h)"), ("linear:lp:plain", "exact tangent of the additive map")):
    k, n = tot[(arm, "none")]
    ks, ns = tot[(arm, "shuffled")]
    add(lab, "Table 1", "w2A_x1_14b_L32_s{0,1,2}", "14B, L32, one position before the first piece", "64 generic carriers, true first piece appended", "n/a", "second piece top-10 by change in log-probability", "greedy-correct sub-word items", k, n, f"same-category {ks}/{ns}; cross-category {tot[(arm,'crosscat')][0]}/{tot[(arm,'crosscat')][1]}; random {tot[(arm,'random')][0]}/{tot[(arm,'random')][1]}", "registered (R1/R2)")

# ---- Section 6 benchmark --------------------------------------------------------------------------------------
def bench_row(claim, where, tag, fam, arm, cell, status, note="", layer=None, denom="all"):
    f = load(tag)["families"][fam]
    a = f["arms"][arm]
    d = a["denominators"][denom]
    n = d["n_decided"]
    rate = d["pass"]
    if layer is not None:
        rate = a["pass_rate_by_layer"][str(layer)]
    k = round(rate * n)
    sh = f["arms"].get(arm + "__shuffled")
    ctr = f"shuffled {round(sh['denominators'][denom]['pass'] * sh['denominators'][denom]['n_decided'])}/{sh['denominators'][denom]['n_decided']}" if sh else "n/a"
    for extra in ("crosscat", "random"):
        e = f["arms"].get(arm + "__" + extra)
        if e:
            ctr += f"; {extra} {round(e['denominators'][denom]['pass'] * e['denominators'][denom]['n_decided'])}/{e['denominators'][denom]['n_decided']} decided of {e['denominators'][denom]['n']}"
    np_ = f["arms"].get("x3c_nopatch")
    if np_ and arm.startswith("x3c"):
        ctr += f"; no patch {round(np_['denominators'][denom]['pass'] * np_['denominators'][denom]['n_decided'])}/{np_['denominators'][denom]['n_decided']}"
    prim = a["primary_only"]
    pk = round(prim["denominators"][denom]["pass"] * prim["denominators"][denom]["n_decided"]) if denom in prim["denominators"] and prim["denominators"][denom]["n_decided"] else None
    add(claim, where, tag, cell, "continuation prompt, 8 carriers" if arm.startswith("x3c") else "copy prompt, 8 carriers", "12 greedy, uncut" if arm.startswith("x3c") else "8 greedy, cut", "bench regex (units may hit different samples at one layer)",
        {"all": "all 100", "gated": "gated (48-token plain prompt)", "gated_immediate": "gated, immediate"}[denom], k, n, ctr, status, note + (f"; primary-string-only {pk}/{n}" if pk is not None else ""))


bench_row("27B basic, pre-named cell", "Table 4", "w5_27b_full_basic", "basic_readout_mt", "x3c_t8", "27B, L52 -> block 8", "registered (A19 pre-named)", layer=52)
bench_row("27B basic, selected two-layer union", "Table 4", "w5_27b_full_basic", "basic_readout_mt", "x3c_t4", "27B, L52 or L56 -> block 4 (union over layers)", "post hoc (best of 3 targets x 2 layers)", note="per layer L52 17/100, L56 15/100; strict single-sample recount 21/100")
bench_row("27B basic, same layer", "Table 4", "w5_27b_full_basic", "basic_readout_mt", "x3c", "27B, L52 or L56 -> same layer", "report-only")
bench_row("27B basic, copy prompt", "Table 4", "w5_27b_full_basic", "basic_readout_mt", "x3_identity3_t4", "27B, L52 or L56 -> block 4", "report-only")
bench_row("27B typo, block 4", "Table 4", "w5_27b_full_typo", "typo_mt", "x3c_t4", "27B, L52 or L56 -> block 4", "post hoc (report-only family)", note="7 of 34 gated items pass; 26 passes are ungated")
bench_row("27B typo, block 8", "Table 4", "w5_27b_full_typo", "typo_mt", "x3c_t8", "27B, L52 or L56 -> block 8", "post hoc (report-only family)")
bench_row("27B multihop, block 4", "Table 4", "w5_27b_full_multihop", "multihop_mt", "x3c_t4", "27B, L52 or L56 -> block 4", "post hoc (report-only family)", note="final-answer unit alone hits on 35/100; bridge units are the required ones")
bench_row("14B basic, gated (A15 pass)", "Sec. 6 text", "w5_ctrl_14b_basic_typo", "basic_readout_mt", "x3c_t4", "14B, L32 -> block 4", "registered (A15/A20 pre-named)", denom="gated", note="cross-category control decided on 28 of 63 (35 items have no partner)")
bench_row("14B basic, immediate (A15 pass)", "Sec. 6 text", "w5_ctrl_14b_basic_typo", "basic_readout_mt", "x3c_t4", "14B, L32 -> block 4", "registered (A15 pre-named)", denom="gated_immediate")
bench_row("14B typo, gated", "Sec. 6 text", "w5_ctrl_14b_basic_typo", "typo_mt", "x3c_t8", "14B, L32 -> block 8", "post hoc (typo report-only under A15)", denom="gated", note="cross-category control decided on 16 of 25")

# ---- Section 7 bridges and certificate -------------------------------------------------------------------------
k, n = cnt("w5_br_14b_B_fh", "continuation:replace:t4:firsthop", kind="bridge", position_rule="firsthop")
rs = rows("w5_br_14b_B_fh", "continuation:replace:t4:firsthop", position_rule="firsthop")
kb = sum(int(any(bench_hits(r))) for r in rs)
add("first-hop bridge, continuation prompt", "Sec. 7 text", "w5_br_14b_B_fh", "14B, L32 -> block 4, read at the first-hop token", "continuation prompt, 8 carriers", "8 greedy, uncut", "bench regex any-of-8 (exact majority in note)", "two-hop items with a located first-hop token", kb, len(rs), ctrl_str("w5_br_14b_B_fh", "continuation:replace:t4:firsthop", "any_exact", "bridge", ("shuffled", "random"), "firsthop") + " (shuffled n 23)", "registered (A22 (i), one of four arms; passes)", f"exact majority {k}/{n}; 17 distinct first-hop cues")
rs = rows("w5_a26_14b_neutral_v2", "continuation:replace:t4:firsthop", position_rule="firsthop")
kb = sum(int(any(bench_hits(r))) for r in rs)
add("neutral-context control", "Sec. 7 text", "w5_a26_14b_neutral_v2", "14B, L32 -> block 4, read at the entity token of a neutral sentence", "continuation prompt, 8 carriers", "8 greedy, uncut", "bench regex any-of-8", "same 30 entities x 2 templates", kb, len(rs), ctrl_str("w5_a26_14b_neutral_v2", "continuation:replace:t4:firsthop", "any_exact", "bridge", ("shuffled", "random"), "firsthop"), "registered (A26; verdict unresolved)", "templates 20/30 and 23/30")
for arm, cell in (("identity3:replace:t4:last", "14B, L32 -> block 4, last token"),):
    k, n = cnt("w5_br_14b_B_fh", arm, kind="bridge", position_rule="last")
    add("bridge at the last token, copy prompt", "Sec. 7 text", "w5_br_14b_B_fh", cell, "copy prompt, 8 carriers", "8 greedy, cut", "exact majority", "gated two-hop items", k, n, ctrl_str("w5_br_14b_B_fh", arm, "vote_exact", "bridge", ("shuffled", "random"), "last"), "registered (R5; did not pass; no kill line in R5)")
pp = load("w5_br_14b_mh_scan")["families"]["multihop_mt"]["position_profile"]["x3c_t4"]["offsets"]
best = max(pp.items(), key=lambda kv: kv[1]["unit"]["bridge1"])
add("benchmark multihop position scan, bridge unit maximum", "Sec. 7 text", "w5_br_14b_mh_scan", f"14B, L32 -> block 4, offset {best[0]} of the last 40 positions", "two answer-frame carriers", "12 greedy, uncut", "bench regex, bridge-1 unit, maximum over offsets", "66 gated multihop items (eligible n varies by offset)", round(best[1]["unit"]["bridge1"] * best[1]["n"]), best[1]["n"], "shuffled maximum 0", "registered (A22 (ii); below its .05 line)", f"final-answer unit at the last token {pp['1']['unit']['target']:.3f}")
cert = load("w5_e5c_14b_L32")["layers"]["32"]["targets"]["t4"]
add("certificate: background activations at full agreement", "Sec. 7 text", "w5_e5c_14b_L32", "14B, L32 -> block 4, generic text", "copy prompt, 8 carriers", "8 greedy, cut", "k_agree = 8", "2,304 background activations", cert["k_hist_all"]["8"], sum(cert["k_hist_all"].values()), "n/a", "registered (A25; passes as registered, zero coverage)")

write_json("provenance.json", {"what": "paper/analysis/provenance.py: claim-to-run provenance", "claims": C})
write_csv("provenance.csv", C, ["claim", "where", "run", "cell", "carrier", "tokens", "rule", "gate", "k", "n", "rate", "controls", "status", "note"])


def tex(s):
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%").replace("->", "$\\to$").replace("<=", "$\\le$").replace("{", "\\{").replace("}", "\\}")


GROUPS = [("Exact recovery, one-token calibration and the mechanism arms (Sections 4 and 5)", lambda c: c["where"].startswith(("Table 2", "Sec. 4", "Sec. 5"))),
          ("Linear and first-order readers (Section 3, Table 1)", lambda c: c["where"].startswith("Table 1")),
          ("Benchmark, bridges and certificate (Sections 6 and 7)", lambda c: c["where"].startswith(("Table 4", "Sec. 6", "Sec. 7")))]
ROWS_PER_TABLE = 9  # a table* cannot break across pages, so each group is emitted in page-sized chunks
HEAD = ("\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{1mm}\n"
        "\\begin{tabular}{@{}p{2.5cm}p{2.3cm}p{4.5cm}p{1.3cm}p{3.0cm}p{2.9cm}@{}}\n\\toprule\n"
        "claim (location) & run & cell; carrier; tokens; rule; gate & $k/N$ & controls & registered or post hoc; note \\\\\n\\midrule\n")
CAP = (". Every $k/N$ is recomputed from the item rows of the named results file by "
       "\\texttt{paper/analysis/provenance.py}; \\texttt{controls} gives the numerator and denominator of each control "
       "on the same cell (a cross-category control is decided only on items with a partner).}\n\\end{table*}")
frag = []
for title, sel in GROUPS:
    body = []
    for c in C:
        if not sel(c):
            continue
        body.append(f"{tex(c['claim'])} ({tex(c['where'])}) & \\texttt{{{tex(c['run']).replace(chr(92)+'_', chr(92)+'_'+chr(92)+'allowbreak{{}}')}}} & {tex(c['cell'])}; {tex(c['carrier'])}; {tex(c['tokens'])}; {tex(c['rule'])}; {tex(c['gate'])} & {c['k']}/{c['n']} & {tex(c['controls'])} & {tex(c['status'])}{('; ' + tex(c['note'])) if c['note'] else ''} \\\\")
    n_chunks = max(1, -(-len(body) // ROWS_PER_TABLE))  # evenly sized chunks, so no table holds a single stray row
    cut = [len(body) * i // n_chunks for i in range(n_chunks + 1)]
    chunks = [body[cut[i]:cut[i + 1]] for i in range(n_chunks)]
    for j, ch in enumerate(chunks):
        part = title if len(chunks) == 1 else f"{title}, part {j + 1} of {len(chunks)}"
        frag.append(HEAD + "\n".join(ch) + "\n\\bottomrule\n\\end{tabular}\n\\caption{Claim-to-run provenance, " + part + CAP)
open(ROOT / "paper" / "analysis" / "provenance_table.tex", "w", encoding="utf-8").write("\n\n".join(frag) + "\n")
for c in C:
    print(f"{c['claim'][:44]:44s} {c['run'][:26]:26s} {c['k']:5d}/{c['n']:<5d} {c['status'][:40]}")
print(f"-> provenance.json, provenance.csv, provenance_table.tex ({len(C)} rows)")
