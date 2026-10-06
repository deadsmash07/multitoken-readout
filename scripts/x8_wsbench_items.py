"""X8 / P6: WorkspaceBench's own multi-token items, prompts, read sites and regex scorer, on OUR models (Qwen3-1.7B /
14B; Qwen3.6-27B through --backend hf), with the ORIGINAL J-lens and the logit lens as the baselines beside our
readers, everything written in the bench's readouts contract so `wsbench judge` can re-score it.

Bank / plan fidelity (camilablank/workspace-bench, commit 92d763e, 2026-09-23):
  items      evals/<family>/items.json for basic_readout_mt, multihop_mt, multilingual_mt, typo_mt, multilingual_typo,
             multilingual_multihop (vendored under data/wsbench/; --wsbench-root <checkout> uses the checkout's copy and
             puts <checkout>/src on sys.path). id = label_of(name) (banks.py).
  render     "plain": tokenizer(prompt, add_special_tokens=False) (readplan / produce/render.py).
  read site  readout.offsets [-1] -> the FINAL prompt token (readplan._final_token_families); the row's `pos` is that
             absolute index and `token` its decoded string, as the bench's producer writes them.
  gate       the bench gated every item on Qwen3.6-27B (greedy-correct AND >= 8/10 at T 0.7 through a chat-templated
             capability question; multihop families also gate every bridge question). Our gate on OUR model, the
             documented tolerance: (a) surface leg = the greedy PLAIN continuation of the prompt (--gate-tokens) must
             contain the target or a target_alt under the bench's own matcher (unicode_word_matcher, word boundary /
             answer position for numbers); (b) bridge leg (multihop_mt, multilingual_multihop) = every bridges[].question
             asked through the tokenizer's chat template (thinking off; "Question: ...\\nAnswer:" without a template),
             greedy --gate-tokens, must contain one of bridges[].answers; (c) optional --gate-draws N sampled continuations
             at --gate-temp must reach --gate-rate (default 0 draws = greedy only). Items failing the gate are reported,
             never scored ("a lens that says nothing about an intermediate the model never computed is not wrong") -
             unless --no-gate (A16): then EVERY item is read and scored and the gate is only recorded.
  immediate  (A15 / A16; ANALYSIS_WAVE3 4.4) every gate row also records `immediate`: the folded greedy continuation
             STARTS with the target / an alias (after optional quotes and an article) - the honest subset for a
             final-token reader, where the read position predicts the target next.
  denominators  every arm reports its pass rate over three item sets (results.json["table"][arm][family]):
             pass_all (all items read: the bench-comparable number, only with --no-gate), pass_gated (the 48-token
             gate), pass_immediate (gated AND immediate); with n for each.
  scoring    the regex contract of wsbench.multitoken.regex (the bench's module when importable, else the vendored
             port sjlens/wsbench_regex.py, pinned to the bench's goldens): a form hits ONE sample (a prose sample, or ONE
             top-k token string); a layer passes when every required unit hits; an item passes at any layer of the
             arm's layer grid. Token bags are scored RAW here (no LLM summarizer; the bench's token-arm headline
             `mt-regex-summarized` needs Gemini): that is the bench's own `regex-raw` column, where the released J-lens
             scores 0.00 on every mt family by construction. runs/<tag>/judge_cmds.sh lists the `wsbench judge`
             commands (prose arms need no key; token arms need OPENROUTER_API_KEY for the summarized number) and the
             direct-Gemini equivalent (scripts/wsbench_judge_gemini.py, the bench's summarize-then-regex protocol).
Arms (each its own readouts file, rows in the contract; controls = shuffled-h within the family's subfamily, A1):
  jlens            the bench's J-lens arm: the released lens, cosine readout (W_U J h)/||J^T W_U[t]||, top-10 tokens with
                   scores (produce/methods.py::JLens; the bench's class when importable, else the same formula).
  logitlens        the bench's logit_lens arm: W_U norm(h), top-10.
  jlens_chain / logitlens_chain   the plain-lens ordered-bag chains of X7 (prose samples: 10 prefix strings).
  p3               the X7 chain reader (N = 10 samples under --rank, A11c default `last`) at the read layers with the
                   first piece from --first at --first-layer; p3_top1 = its N = 1 sample (under tau when --tau /
                   --tau-from is given, else the best prefix under --rank: tau is OPTIONAL). Every P3 / chain prefix is
                   written to runs/<tag>/prefixes/<arm>__<ctrl>/<family>.jsonl so the samples can be re-ranked and
                   re-scored offline (--rescore --rank R [--tau t]: CPU, no model).
  x3_<kind>[_t<l>] the Patchscope (x3_patchscope.py) with --x3-carriers-kind carriers (comma list; default identity3,
                   the item-free exemplars of A12; identity3_long / spellfix are the A15 / A17 kinds): h read at the read
                   layer l is patched into the carrier's "?" position AFTER block t for every t of --x3-target-layer
                   (comma list; `same` = the read layer -> arm x3_<kind>, an integer t -> x3_<kind>_t<t>;
                   x3_patchscope's target-layer path, x3_generate below == x3.patched_generate(mode replace)), greedy
                   --x3-gen tokens, cut at the carrier stops, ONE sample per carrier (8 carriers = 8 prose samples of
                   one cell; the bench's any-sample rule = X3's "any of 8"). The UNCUT generations are written as a
                   diagnostic arm x3_<kind>[_t<l>]_uncut (no extra generation; the bench scores prose as it comes, so
                   this is the bench-rule analogue of E5a; ANALYSIS_WAVE3 4.8 early close ' sodium -> sodium chloride').
  x3c[_t<l>]       (A15) the CONTINUATION carrier: h REPLACES the residual at the LAST token of each of the 8 neutral
                   prefixes (x3.CONTINUATION) after block t; greedy --x3c-gen (12) tokens; the raw generation is one
                   prose sample (no stop cut). A prediction-site reader: the model continues from the patched state.
  x3s_ll[_t<l>] / x3s_jl[_t<l>]   (A15) lens-SEEDED teacher-forced continuation: the first-piece candidates are the
                   top-5 tokens of the LOGIT LENS (x3s_ll) or of the bench's cosine J-lens (x3s_jl) at the read layer;
                   for each candidate and each of the 2 answer frames (x3.CONTINUATION_SEED) h is patched at the frame's
                   last token after block t, the candidate is appended, and the model continues greedy --x3s-gen (10)
                   tokens; sample = candidate + continuation; 5 x 2 = 10 samples per cell (the bench's token top-10
                   budget). Single vector, single layer: bench-legal.
  --x3-positions span   (A17 secondary, typo families) every X3-type arm is also read at every earlier piece of the
                   typo span (the prompt ends with typo_word): arm suffix _off<k> = k tokens from the end (k >= 2; the
                   base arm is the final token); rows carry the absolute `pos`. A per-position profile, labelled post hoc.
Layers: --layers is the grid of the cheap token / bag arms (any-layer pass over it); --p3-layers the read layers of
p3 and, unless --x3-layers is given (A19: the two were one flag before wave 5, which is why the 27B target scan read
X3 at L48 only), of the x3 arms (their any-layer pass is over those). The token / bag grid is extended to include the
read layers so that the J-lens and the logit lens are always read on the SAME cells as X3 / P3; every token / bag arm
also reports its pass rate restricted to the read layers (`on_read_layers`). A11 registers this as a pilot: report only.
Wave 5 (A20 / A22): --controls crosscat (cross-subfamily partner), random (norm-matched Gaussian h) and nopatch (the
bare carriers generated once with no h; arm <prefix>_nopatch, and results.json["families"][f]["nopatch_withdrawal"]
lists the items each real arm loses to it); every arm carries a `primary_only` block (first creditable form per unit,
no aliases; table columns pass_primary_*); --x3-positions all reads every prompt position (arms _off<k>) and
results.json["families"][f]["position_profile"] gives the per-offset unit rates and the any-position union (a
selection); --rescore --rescore-out <tag> re-scores an old run into runs/<tag>/ without touching it."""
import argparse, importlib.util, json, os, re, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.lens import jlens
from sjlens import wsbench_regex as R
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
_load = lambda n, f: (lambda s: (lambda mod: (s.loader.exec_module(mod), mod)[1])(importlib.util.module_from_spec(s)))(importlib.util.spec_from_file_location(n, os.path.join(HERE, f)))
e12 = _load("e12", "e12_second_token.py"); x7 = _load("x7", "x7_chain_reader.py"); x3 = _load("x3", "x3_patchscope.py"); x1 = x7.x1
PREFIX_ARMS = ("p3", "jlens_chain", "logitlens_chain")  # arms whose prefixes are stored and re-rankable offline
X3_READERS = ("x3", "x3c", "x3s", "x3long")  # x3long = x3 with the identity3_long kind (alias)
SEED_K = 5  # x3s: first-piece candidates per lens (5 x 2 frames = 10 samples, the bench's token budget)
_ARTICLE = re.compile(r"^(?:the|a|an)\s+")
_LEAD = " \t\r\n\"'“”‘’(*[`"


def bench_modules(root, log=print):
    """(regex module, produce modules or None): the bench's own code when a checkout is given and importable."""
    if root:
        sys.path.insert(0, os.path.join(root, "src"))
        try:
            import wsbench.multitoken.regex as breg; log(f"using the bench's scorer from {root} ({breg.SCORER_VERSION})")
        except Exception as e:
            log(f"bench scorer not importable ({e!r}); using the vendored port"); breg = R
        try:
            from wsbench.produce import methods as bmeth, backend as bback; prod = (bmeth, bback); log("using the bench's Producer methods (JLens / LogitLens) for the token arms")
        except Exception as e:
            log(f"bench produce modules not importable ({e!r}); the token arms use the vendored formulas"); prod = None
        return breg, prod
    return R, None


def chat_question(tok, q):
    """the bench asks the gate questions through the chat template with thinking off; plain fallback."""
    if getattr(tok, "chat_template", None):
        try:
            out = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            return list(out["input_ids"] if hasattr(out, "keys") else out)
        except Exception: pass
    return tok.encode(f"Question: {q}\nAnswer:", add_special_tokens=False)


def matches_any(text, forms):
    return any(R.unicode_word_matcher(f)(text) for f in forms if f)


def immediate_hit(text, forms):
    """the `immediate` flag (A15 / A16): the folded continuation STARTS with a form - after optional whitespace /
    quotes / brackets and an optional English article - and the form ends at a word boundary (CJK / Hangul forms:
    plain prefix). 'Aghlabids. The Aghlabids were ...' -> True for 'Aghlabids'; ' a dynasty called the Aghlabids' -> False."""
    t = R.fold(text).lstrip(_LEAD); t2 = _ARTICLE.sub("", t).lstrip(_LEAD)
    for f in forms:
        ff = R.fold(str(f)).strip()
        if not ff: continue
        cjk = bool(R._CJK_HANGUL.search(str(f)))
        for cand in (t, t2):
            if cand.startswith(ff) and (cjk or len(cand) == len(ff) or not re.match(r"[^\W_]", cand[len(ff)])): return True
    return False


def target_forms(it):
    return [str(it.get("target") or "")] + [str(x) for x in it.get("target_alts", [])]


def immediate_of_row(row, it):
    """the gate row's immediate flag; rows of runs before A15 / A16 lack it and are re-derived from the stored greedy text."""
    if "immediate" in row: return bool(row["immediate"])
    s = row.get("surface") or {}
    return bool(row.get("ok")) and immediate_hit(str(s.get("greedy") or ""), target_forms(it))


def gate_item(backend, tok, m, it, ids, n_tok, draws=0, temp=0.7, rate=0.8):
    """the documented gate (module docstring): surface leg, bridge legs, optional sampled consistency; plus the
    `immediate` flag of the surface leg (never part of the gate decision)."""
    dev = backend.device; out = {"surface": None, "bridges": [], "sampled": None}
    g = backend.greedy(ids, n_tok); txt = tok.decode(g); forms = target_forms(it)
    out["surface"] = {"greedy": txt, "ok": bool(matches_any(txt, forms)), "immediate": bool(immediate_hit(txt, forms))}
    ok = out["surface"]["ok"]
    for br in it.get("bridges") or []:
        qids = torch.tensor([chat_question(tok, br["question"])], device=dev); gb = backend.greedy(qids, n_tok); tb = tok.decode(gb)
        okb = bool(matches_any(tb, [str(x) for x in br.get("answers", [])])); out["bridges"].append({"question": br["question"], "greedy": tb, "ok": okb}); ok = ok and okb
    if ok and draws > 0:
        with torch.no_grad():
            gen = m.generate(ids, do_sample=True, temperature=temp, top_p=0.95, top_k=64, max_new_tokens=n_tok, num_return_sequences=draws, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        texts = tok.batch_decode(gen[:, ids.shape[1]:], skip_special_tokens=True); r = sum(matches_any(t, forms) for t in texts) / draws
        out["sampled"] = {"rate": r, "ok": bool(r >= rate), "n": draws}; ok = ok and out["sampled"]["ok"]
    out["ok"] = bool(ok); out["immediate"] = bool(ok and out["surface"]["immediate"]); return out


def token_row(tok, item_id, layer, pos, token, ids, scores):
    return {"id": item_id, "layer": layer, "pos": pos, "token": token, "tokens": R.display_tokens(tok, ids), "scores": [round(float(v), 4) for v in scores]}


def prose_row(item_id, layer, pos, token, samples):
    return {"id": item_id, "layer": layer, "pos": pos, "token": token, "samples": [str(s) for s in samples]}


def x3_generate(backend, cid, cp, h, target_layer, n_gen, xclean=None):
    """the X3 identity Patchscope through a backend: h (read at the SOURCE layer) REPLACES the carrier residual at
    position cp after block `target_layer` (delta = h - x_clean, baked into the prefill cache), then greedy n_gen
    tokens. == x3.patched_generate(q, cid, source_layer, cp, h, n_gen, "replace", 1.0, None, target_layer) on the
    Qwen3Min path (tests/test_n0_tiny.py). xclean = the clean carrier residual at (target_layer, cp), cacheable."""
    x = backend.resid(cid, target_layer)[0, cp].float() if xclean is None else xclean
    mask = torch.zeros(1, cid.shape[1], dtype=torch.bool, device=cid.device); mask[0, cp] = True
    return backend.greedy(cid, n_gen, (target_layer, mask, (h.float() - x).view(1, 1, -1).expand(1, cid.shape[1], -1)))


def seeded_generate(backend, cid, cp, h, target_layer, seed_ids, n_gen, xclean=None):
    """x3s (A15): as x3_generate, but the seed token(s) are APPENDED to the carrier before generation (teacher-forced
    first piece), so the model continues from "<frame> <seed>" with h patched at the frame's last token cp after block
    `target_layer`. The clean residual at cp does not depend on the appended seed (causal), so xclean is shared with
    x3_generate. Returns seed_ids + the n_gen generated ids. With an empty seed == x3_generate."""
    seed = [int(t) for t in seed_ids]
    ids = torch.cat([cid, torch.tensor([seed], dtype=cid.dtype, device=cid.device)], 1) if seed else cid
    x = backend.resid(cid, target_layer)[0, cp].float() if xclean is None else xclean
    mask = torch.zeros(1, ids.shape[1], dtype=torch.bool, device=cid.device); mask[0, cp] = True
    return seed + backend.greedy(ids, n_gen, (target_layer, mask, (h.float() - x).view(1, 1, -1).expand(1, ids.shape[1], -1)))


def lens_seeds(z, k, special_ids=()):
    """the top-k token ids of a [V] lens score vector, special tokens excluded (x3s first-piece candidates)."""
    z = z.detach().float().clone()
    for t in special_ids:
        if 0 <= int(t) < z.numel(): z[int(t)] = float("-inf")
    return torch.topk(z, min(k, z.numel())).indices.tolist()


def arm_of(prefix, target, read_layer):
    """<prefix> at the read layer ('same'), <prefix>_t<l> for an early target block l."""
    return prefix if target == "same" or target == read_layer else f"{prefix}_t{target}"


def x3_arm_name(kind, target, read_layer):
    """x3_<kind> at the read layer ('same'), x3_<kind>_t<l> for an early target block l."""
    return arm_of(f"x3_{kind}", target, read_layer)


def typo_span(tok, ids, typo_word):
    """the positions of the typo span: the smallest k such that the decoded last k tokens end with typo_word (the six
    typo prompts all END with typo_word, ANALYSIS_WAVE3 3.6); [] when the word is not found in the last 12 tokens."""
    if not typo_word: return []
    w = str(typo_word).strip().casefold(); n = ids.shape[1]
    for k in range(1, min(12, n) + 1):
        if tok.decode([int(t) for t in ids[0, n - k:]]).strip().casefold().endswith(w): return list(range(n - k, n))
    return []


def prefix_row(item_id, layer, pos, token, target, cands, prefixes):
    """one stored-prefix row (runs/<tag>/prefixes/...): everything needed to rebuild the samples under any rank / tau."""
    return {"id": item_id, "layer": layer, "pos": pos, "token": token, "target": target, "cands": cands,
            "prefixes": [{"ids": list(p["ids"]), "t": p["t"], "S": p["S"], "d": list(p["d"]), "fr": p["fr"]} for p in prefixes]}


def samples_of(arm, prefixes, rank, tau, n=10):
    """the prose samples of a prefix arm under (rank, tau): p3 / chains = the N = 10 set; p3_top1 = the N = 1 sample."""
    if arm == "p3_top1": return [x7.sample_top1(prefixes, tau, rank)]
    return x7.samples_topn(prefixes, n, rank)


def write_readouts(out_dir, arm, ctrl, fam, rs):
    sub = arm + ("" if ctrl == "none" else "__" + ctrl); d = os.path.join(out_dir, "readouts", sub); os.makedirs(d, exist_ok=True); f = os.path.join(d, fam + ".jsonl")
    with open(f, "w", encoding="utf-8") as fh:
        for r in rs: fh.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
    return sub, f


def subset_rates(rows, ids, seed):
    """pass / any-hit rates (with the item-level bootstrap CI) of the scored rows restricted to `ids` (one denominator)."""
    rs = [r for r in rows if r["id"] in ids]; dec = [r for r in rs if r["pass"] is not None]
    pr = [float(bool(r["pass"])) for r in dec]; roles = sorted({x for r in rs for x in r["roles"] + r["optional_roles"]})
    rate = lambda xs: (sum(xs) / len(xs)) if xs else None
    return {"n": len(rs), "n_decided": len(dec), "pass": rate(pr), "any_hit": rate([float(r["any_hit"]) for r in dec]),
            "pass_ci": x1.boot(pr, lambda t: t.mean(), seed=seed)[1:] if pr else None,
            "unit_any_layer": {x: rate([float(r["unit_hit"][x]) for r in dec if x in r["unit_hit"]]) for x in roles}}


def score_arm(f, sub, arm, scored, contract, arm_layers, read_layers, seed, gated_ids=None, immediate_ids=None, no_gate=False):
    """score one readouts file over its layer grid on the SCORED items (gated, or every item with --no-gate); token /
    bag arms also over the read layers only (same cells as X3 / P3); the three denominators (A16) beside it."""
    cells = [R.parse_readout_row(json.loads(line)) for line in open(f, encoding="utf-8")]
    assert all(c is not None for c in cells), f"malformed readouts row written for {sub}"
    sc = R.score_cells(cells, scored, contract, arm_layers)
    pr = [float(bool(r["pass"])) for r in sc["rows"] if r["pass"] is not None]
    sc["pass_rate_ci"] = x1.boot(pr, lambda t: t.mean(), seed=seed)[1:] if pr else None
    e = {"file": f, "layers": arm_layers, "kind": "tokens" if arm in ("jlens", "logitlens") else "prose", **sc}
    rl = [l for l in read_layers if l in arm_layers]
    if rl and sorted(rl) != sorted(arm_layers):
        s2 = R.score_cells(cells, scored, contract, rl); e["on_read_layers"] = {"layers": rl, "pass_rate": s2["pass_rate"], "any_hit_rate": s2["any_hit_rate"], "n_items_decided": s2["n_items_decided"], "pass_rate_by_layer": s2["pass_rate_by_layer"]}
    ids = {it["id"] for it in scored}; g = set(gated_ids if gated_ids is not None else ids) & ids; im = set(immediate_ids or ()) & g
    dens = lambda rows: {"all": subset_rates(rows, ids, seed) if no_gate else {"n": None, "pass": None, "any_hit": None, "note": "gated run: items failing the gate were not read (not bench-comparable)"},
                         "gated": subset_rates(rows, g, seed), "gated_immediate": subset_rates(rows, im, seed)}
    e["denominators"] = dens(sc["rows"])
    # A20: the primary-string-only column (first creditable form per unit and language; no aliases), beside the bench rule
    prim = {it["id"]: R.primary_units(R.scored_units(it, contract)) for it in scored}
    sp = R.score_cells(cells, scored, contract, arm_layers, units_of=prim)
    e["primary_only"] = {"pass_rate": sp["pass_rate"], "any_hit_rate": sp["any_hit_rate"], "n_items_decided": sp["n_items_decided"], "pass_rate_by_layer": sp["pass_rate_by_layer"], "unit_any_layer": sp["unit_any_layer"],
                         "pass_ids": sorted(r["id"] for r in sp["rows"] if r["pass"]), "alias_only_ids": sorted({r["id"] for r in sc["rows"] if r["pass"]} - {r["id"] for r in sp["rows"] if r["pass"]}),
                         "passing_layers_by_id": {r["id"]: r["passing_layers"] for r in sp["rows"] if r["passing_layers"]}, "denominators": dens(sp["rows"])}
    if rl and sorted(rl) != sorted(arm_layers):
        s3 = R.score_cells(cells, scored, contract, rl, units_of=prim); e["primary_only"]["on_read_layers"] = {"layers": rl, "pass_rate": s3["pass_rate"], "pass_rate_by_layer": s3["pass_rate_by_layer"]}
    return e


_OFF = re.compile(r"^(.*)_off(\d+)$")


def position_profile(arms):
    """A22: for every arm family read at several positions (base arm = offset 1 from the end, _off<k> = k from the
    end): per-offset pass / unit rates (one denominator: the items decided at every offset) and the per-item
    any-position union (a per-item SELECTION over positions, labelled as such). Controls are profiled the same way."""
    fam = {}
    for sub, e in arms.items():
        base, ctrl = (sub.split("__") + ["none"])[:2]; m = _OFF.match(base)
        key, off = (m.group(1), int(m.group(2))) if m else (base, 1)
        fam.setdefault((key, ctrl), {})[off] = e
    out = {}
    for (key, ctrl), by_off in fam.items():
        if len(by_off) < 2: continue
        roles = sorted({r for e in by_off.values() for r in (e.get("unit_any_layer") or {})})
        prof = {}; union = {}; ids_all = None
        for off, e in sorted(by_off.items()):
            dec = {r["id"]: r for r in e["rows"] if r["pass"] is not None}
            ids_all = set(dec) if ids_all is None else ids_all & set(dec)
            prof[str(off)] = {"n": len(dec), "pass": e["pass_rate"], "unit": {r: e["unit_any_layer"].get(r) for r in roles}}
            for i, r in dec.items():
                u = union.setdefault(i, {"pass": False, "unit": {x: False for x in roles}}); u["pass"] = u["pass"] or bool(r["pass"])
                for x in roles: u["unit"][x] = u["unit"][x] or bool(r["unit_hit"].get(x))
        n = len(ids_all or ()); rate = lambda xs: (sum(xs) / len(xs)) if xs else None
        best = {"n": n, "pass": rate([float(union[i]["pass"]) for i in ids_all or ()]), "unit": {x: rate([float(union[i]["unit"][x]) for i in ids_all or ()]) for x in roles}}
        out[key + ("" if ctrl == "none" else "__" + ctrl)] = {"offsets": prof, "any_position": {**best, "note": "union over read positions per item: a per-item selection, not a per-position statistic"}}
    return out


def nopatch_withdrawal(arms):
    """A20: for every real x3-type arm whose carrier family has a _nopatch arm, the items the BARE carrier already
    passes are withdrawn from that arm's pass set; the withdrawn ids and the pass rate after withdrawal are recorded."""
    out = {}
    for sub, e in arms.items():
        base, ctrl = (sub.split("__") + ["none"])[:2]
        if ctrl != "none" or "_nopatch" in base: continue
        m = _OFF.match(base); base0 = m.group(1) if m else base
        uncut = base0.endswith("_uncut"); core = base0[: -len("_uncut")] if uncut else base0
        np_name = re.sub(r"_t\d+$", "", core) + "_nopatch" + ("_uncut" if uncut else ""); np_ = arms.get(np_name)
        if np_ is None or not base.startswith("x3"): continue
        np_ids = {r["id"] for r in np_["rows"] if r["pass"]}; real = {r["id"] for r in e["rows"] if r["pass"]}; dec = [r for r in e["rows"] if r["pass"] is not None]
        out[sub] = {"nopatch_arm": np_name, "n_nopatch_pass": len(np_ids), "withdrawn_ids": sorted(real & np_ids), "pass_rate_after_withdrawal": (len(real - np_ids) / len(dec)) if dec else None}
    return out


def is_read_layer_arm(arm):
    return arm in ("p3", "p3_top1") or arm.startswith("x3")


def gate_sets(fres, items):
    """(scored items, gated ids, immediate ids) of one family's results block (old runs: immediate re-derived)."""
    by_id = {it["id"]: it for it in items}; ok_ids = {r["id"] for r in fres["gate_rows"] if r["ok"]}
    im_ids = {r["id"] for r in fres["gate_rows"] if r["id"] in by_id and immediate_of_row(r, by_id[r["id"]])}
    read_ids = {r["id"] for r in fres["gate_rows"]} if fres.get("no_gate") else ok_ids
    return [it for it in items if it["id"] in read_ids], ok_ids, im_ids


def rescore(out_dir, rank, tau, seed, log, dest=None):
    """CPU: rebuild the p3 / p3_top1 / chain readouts from the stored prefixes under (rank, tau), re-score every arm
    from its readouts file (three denominators; old runs gain the immediate flag; A20 primary-only column), rewrite
    results.json - in place, or under `dest` (--rescore-out: the run itself is left untouched)."""
    src_f = os.path.join(out_dir, "results.json"); results = json.load(open(src_f, encoding="utf-8"))
    if dest: os.makedirs(dest, exist_ok=True); results["rescored_from"] = os.path.relpath(out_dir, os.path.dirname(HERE))
    out_f = os.path.join(dest or out_dir, "results.json")
    breg = R; read_layers = sorted(set(results["p3_layers"]) | set(results.get("x3_layers") or []))
    for fam, fres in results["families"].items():
        if not fres.get("arms"): continue
        bank = fres["bank"] if os.path.exists(fres["bank"]) else os.path.join(os.path.dirname(HERE), "data", "wsbench", fam + ".json")  # a run from another machine: the vendored bank
        header, items = R.load_bank(bank); contract = breg.contract_for(header, header.get("family", fam)); scored, ok_ids, im_ids = gate_sets(fres, items); no_gate = bool(fres.get("no_gate"))
        for r in fres["gate_rows"]: r.setdefault("immediate", r["id"] in im_ids)
        fres["n_gated"] = len(ok_ids); fres["n_immediate"] = len(im_ids); fres["n_scored"] = len(scored)
        kw = dict(gated_ids=ok_ids, immediate_ids=im_ids, no_gate=no_gate)
        pdir = os.path.join(out_dir, "prefixes")
        for sub in sorted(os.listdir(pdir)) if os.path.isdir(pdir) else []:
            f = os.path.join(pdir, sub, fam + ".jsonl")
            if not os.path.exists(f): continue
            arm, ctrl = (sub.split("__") + ["none"])[:2]; prows = [json.loads(l) for l in open(f, encoding="utf-8")]
            for a2 in ((arm, "p3_top1") if arm == "p3" else (arm,)):
                rs = [prose_row(r["id"], r["layer"], r["pos"], r["token"], samples_of(a2, r["prefixes"], rank, tau)) for r in prows]
                sub2, f2 = write_readouts(dest or out_dir, a2, ctrl, fam, rs); arm_layers = fres["arms"].get(sub2, {}).get("layers") or (read_layers if is_read_layer_arm(a2) else sorted({r["layer"] for r in prows}))
                fres["arms"][sub2] = score_arm(f2, sub2, a2, scored, contract, arm_layers, read_layers, seed, **kw); fres["arms"][sub2]["file"] = os.path.relpath(f2, os.path.dirname(HERE))
        for sub, e in fres["arms"].items():  # every other arm: re-score from its readouts file (unchanged samples)
            if sub.split("__")[0] in PREFIX_ARMS or sub.split("__")[0] == "p3_top1": continue
            arm = sub.split("__")[0]; f = os.path.join(os.path.dirname(HERE), e["file"])
            if not os.path.exists(f) and os.path.exists(os.path.join(out_dir, "readouts", sub, fam + ".jsonl")): f = os.path.join(out_dir, "readouts", sub, fam + ".jsonl")  # a run pulled from another machine
            e2 = score_arm(f, sub, arm, scored, contract, e["layers"], read_layers, seed, **kw); e2["file"] = e["file"]; fres["arms"][sub] = e2
        fres["position_profile"] = position_profile(fres["arms"]); fres["nopatch_withdrawal"] = nopatch_withdrawal(fres["arms"])
        for sub, e in fres["arms"].items(): log(f"{fam} {sub:24s}: pass (any layer, raw regex) {e['pass_rate']:.3f} (n {e['n_items_decided']}), any_hit {e['any_hit_rate']:.3f}" + (f"; on read layers {e['on_read_layers']['pass_rate']:.3f}" if e.get("on_read_layers") else "") + f"; primary-only {_fmt((e.get('primary_only') or {}).get('pass_rate'))}")
    results["rank"] = rank; results["tau"] = tau; results["rescored"] = True
    finish(results, out_f, log)


def _fmt(v):
    return "-" if v is None else f"{v:.3f}"


def finish(results, out_f, log):
    table = {}
    for fam, fr in results["families"].items():
        for sub, e in fr.get("arms", {}).items():
            d = e.get("denominators") or {}; al, g, im = d.get("all") or {}, d.get("gated") or {}, d.get("gated_immediate") or {}
            po = e.get("primary_only") or {}; pd = po.get("denominators") or {}
            table.setdefault(sub, {})[fam] = {"pass": e["pass_rate"], "any_hit": e["any_hit_rate"], "n": e["n_items_decided"], "pass_on_read_layers": (e.get("on_read_layers") or {}).get("pass_rate", e["pass_rate"]),
                                              "pass_all": al.get("pass"), "n_all": al.get("n"), "pass_gated": g.get("pass"), "n_gated": g.get("n"), "pass_immediate": im.get("pass"), "n_immediate": im.get("n"),
                                              "pass_primary": po.get("pass_rate"), "pass_primary_all": (pd.get("all") or {}).get("pass"), "pass_primary_gated": (pd.get("gated") or {}).get("pass"), "pass_primary_immediate": (pd.get("gated_immediate") or {}).get("pass"),
                                              "pass_after_nopatch_withdrawal": ((fr.get("nopatch_withdrawal") or {}).get(sub) or {}).get("pass_rate_after_withdrawal")}
    results["table"] = table; json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
    for sub, d in table.items(): log(f"TABLE {sub:24s}: " + "  ".join(f"{fam} {_fmt(v['pass'])} (n {v['n']}) [all {_fmt(v['pass_all'])}/{v['n_all']} gated {_fmt(v['pass_gated'])}/{v['n_gated']} immediate {_fmt(v['pass_immediate'])}/{v['n_immediate']}; primary-only {_fmt(v['pass_primary'])} (gated {_fmt(v['pass_primary_gated'])}, imm {_fmt(v['pass_primary_immediate'])})]" for fam, v in d.items()))
    log("done")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-14B"); p.add_argument("--lens-file", default="qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt")
    p.add_argument("--families", default=",".join(R.MT_FAMILIES)); p.add_argument("--bank-root", default=os.path.join(os.path.dirname(HERE), "data", "wsbench")); p.add_argument("--wsbench-root", default="", help="a workspace-bench checkout: its banks and its scorer / Producer modules are used")
    p.add_argument("--layers", default="16,24,32,36,38", help="grid of the token / bag arms (extended by the read layers)"); p.add_argument("--p3-layers", dest="p3_layers", default="32,36", help="READ layers of p3 (and of the x3 arms unless --x3-layers is given)"); p.add_argument("--first", default="logitlens"); p.add_argument("--first-layer", default="38")
    p.add_argument("--x3-layers", default="", help="READ layers of the x3 / x3c / x3s / x3long arms (A19; default: the --p3-layers list, the pre-A19 coupling)")
    p.add_argument("--readers", default="jlens,logitlens,jlens_chain,logitlens_chain,x3", help="comma list of jlens, logitlens, jlens_chain, logitlens_chain, p3, x3, x3c, x3s, x3long (p3 is optional and expensive)")
    p.add_argument("--controls", default="shuffled", help="comma list of shuffled (within subfamily, A1), crosscat (cross-subfamily derangement, A2), random (norm-matched Gaussian h, A1), nopatch (A20: every x3-type carrier generated once with NO h, arm <prefix>_nopatch), or empty")
    p.add_argument("--x3-carrier-ids", default="", help="comma list of carrier indices to use (every x3 kind and x3c); default: the first --x3-carriers")
    p.add_argument("--x3-pos-max", type=int, default=0, help="with --x3-positions all: read at most this many positions from the end (0 = every prompt position)")
    p.add_argument("--rescore-out", default="", help="with --rescore: write the re-scored results.json (and rebuilt readouts) under runs/<this tag> instead of overwriting the run")
    p.add_argument("--gate-tokens", type=int, default=16); p.add_argument("--gate-draws", type=int, default=0); p.add_argument("--gate-temp", type=float, default=0.7); p.add_argument("--gate-rate", type=float, default=0.8)
    p.add_argument("--no-gate", action="store_true", help="A16: read and score EVERY item (the gate is still run and recorded; three denominators in the table)")
    p.add_argument("--limit", type=int, default=0, help="items per family BEFORE gating (smokes)"); p.add_argument("--n-ctx", type=int, default=64); p.add_argument("--chunk", type=int, default=8); p.add_argument("--T", type=int, default=128)
    p.add_argument("--contexts", default="runs/contexts/qwen3_T128_256.pt"); p.add_argument("--k1", type=int, default=5); p.add_argument("--beam", type=int, default=5); p.add_argument("--max-pieces", type=int, default=4)
    p.add_argument("--tau", type=float, default=None, help="P3 N = 1 stop threshold (OPTIONAL)"); p.add_argument("--tau-from", default="", help="tau.json of a tuning run (optional)"); p.add_argument("--rank", default=x7.RANK_DEFAULT, choices=x7.RANKS, help="P3 / chain sample ranking (A11c default last)")
    p.add_argument("--x3-carriers", type=int, default=8); p.add_argument("--x3-gen", type=int, default=8); p.add_argument("--x3-carriers-kind", default="identity3", help="comma list of identity3 (A12, item-free exemplars; default) | identity3_long (A15) | spellfix (A17) | identity2 (wave-2, overlapping exemplars) | identity")
    p.add_argument("--x3-target-layer", default="same", help="comma list: same (= the read layer) and/or block indices after which h is patched into the carrier (e.g. same,4,8)")
    p.add_argument("--x3c-gen", type=int, default=12, help="x3c continuation tokens (A15)"); p.add_argument("--x3s-gen", type=int, default=10, help="x3s tokens after the seed (A15)"); p.add_argument("--x3s-seeds", default="ll,jl", help="x3s seed lenses: ll (logit lens) and/or jl (bench cosine J-lens)")
    p.add_argument("--x3-positions", default="last", help="last | span (typo families: every piece of the typo span as an extra read position, arms _off<k>; A17 secondary) | all (A22: every prompt position, arms _off<k>, per-position profile; controls at the partner's same offset from its end)")
    p.add_argument("--no-uncut-arms", action="store_true", help="do not write the x3_<kind>*_uncut diagnostic arms (raw generations of the identity kinds)")
    p.add_argument("--rescore", action="store_true", help="CPU: rebuild the p3 / chain samples from runs/<tag>/prefixes under --rank / --tau and re-score every arm")
    p.add_argument("--seed", type=int, default=0); p.add_argument("--dtype", default="bf16-mixed"); p.add_argument("--device", default="cuda"); p.add_argument("--backend", default="min"); p.add_argument("--tag", default="x8_wsbench"); a = p.parse_args()
    out_dir = os.path.join(os.path.dirname(HERE), "runs", a.tag); os.makedirs(os.path.join(out_dir, "readouts"), exist_ok=True); out_f = os.path.join(out_dir, "results.json")
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True); log(" ".join(sys.argv))
    tau = a.tau if a.tau is not None else (json.load(open(a.tau_from))["tau"] if a.tau_from else None)  # None = no threshold (tau optional)
    log(f"rank {a.rank}; tau {tau}")
    if a.rescore: return rescore(out_dir, a.rank, tau, a.seed, log, dest=os.path.join(os.path.dirname(HERE), "runs", a.rescore_out) if a.rescore_out else None)
    breg, prod = bench_modules(a.wsbench_root, log)
    tok, m, backend = x7.load_backend(a.backend, a.model, a.dtype, a.device); dev = backend.device
    readers = [r for r in a.readers.split(",") if r]; p3_layers = [int(x) for x in a.p3_layers.split(",")] if a.p3_layers else []
    x3_layers = [int(x) for x in a.x3_layers.split(",")] if a.x3_layers else list(p3_layers)  # A19: x3 read layers decoupled from p3's (default = the old coupling)
    read_layers = sorted(set(p3_layers) | set(x3_layers))
    layers0 = [int(x) for x in a.layers.split(",")]; layers = sorted(set(layers0) | set(read_layers))
    if layers != sorted(layers0): log(f"token / bag grid {layers0} extended by the read layers to {layers} (same cells as x3 / p3)")
    lf = int(a.first_layer) if a.first_layer != "same" else None
    need = sorted(set(layers) | set(read_layers) | ({lf} if lf is not None and "p3" in readers else set()))
    ctrl_names = [c for c in a.controls.split(",") if c]
    assert all(c in ("shuffled", "crosscat", "random", "nopatch") for c in ctrl_names), f"--controls {a.controls!r}"
    car_ids = [int(x) for x in a.x3_carrier_ids.split(",")] if a.x3_carrier_ids else list(range(a.x3_carriers))
    pick = lambda lst: [lst[i] for i in car_ids if i < len(lst)]
    seed_lenses = [s for s in a.x3s_seeds.split(",") if s] if "x3s" in readers else []
    assert all(s in ("ll", "jl") for s in seed_lenses), f"--x3s-seeds {a.x3s_seeds!r}: ll and/or jl"
    need_J = any(r.startswith("jlens") or (r == "p3" and a.first != "logitlens") for r in readers) or "jl" in seed_lenses
    J, _ = jlens.load(hf_hub_download("neuronpedia/jacobian-lens", a.lens_file)) if need_J else ({}, {})
    W_U = backend.unembed(); D = {}
    jl_layers = [l for l in layers if l in J] if J else []  # the released lens file may stop before the deepest block
    if J and jl_layers != layers: log(f"lens file has layers {min(J)}..{max(J)}: the jlens arms run at {jl_layers} only")
    assert not ("p3" in readers and a.first != "logitlens" and lf is not None and lf not in J), f"--first {a.first} at layer {lf}: not in the lens file"
    def Dl(l):
        if l not in D: D[l] = W_U @ J[l].float().to(W_U.device)
        return D[l]
    bench_lens = {}
    if prod is not None and any(r in ("jlens", "logitlens") for r in readers):  # the bench's own reader classes on the same loaded model
        bmeth, bback = prod; bb = bback.Backend(model=m, tokenizer=tok, device=str(dev), model_id=a.model)
        if "jlens" in readers: bench_lens["jlens"] = bmeth.JLens(filename=a.lens_file); bench_lens["jlens"].bind(bb)
        if "logitlens" in readers: bench_lens["logitlens"] = bmeth.LogitLens(); bench_lens["logitlens"].bind(bb)
    ctx = ids_ctx = xlast = None; ok = x7.allowed_tokens(tok, backend.V) if any(r in ("p3", "jlens_chain", "logitlens_chain") for r in readers) else None
    if "p3" in readers:
        from sjlens.eval.phase_a import contexts
        ctx, _bg = contexts(tok, a.n_ctx, 8, a.T, dev, path=a.contexts if os.path.exists(a.contexts) else None); ids_ctx = torch.stack(ctx)
        xlast = {l: backend.resid(ids_ctx, l)[:, a.T - 1].clone() for l in p3_layers}
        cache = x7.CleanCache(backend, ids_ctx, a.chunk)  # clean carrier + suffix distributions, shared across families / items / controls
    # ---- the X3-type arms: (arm prefix, carrier kind, carrier ids, patch positions, n_gen, cut kind or None) ----
    x3_targets = x3.parse_targets(a.x3_target_layer); xclean = {}; specs = []
    kinds = [k for k in a.x3_carriers_kind.split(",") if k] if "x3" in readers else []
    if "x3long" in readers and "identity3_long" not in kinds: kinds.append("identity3_long")
    for kind in kinds:
        assert kind in x3.CARRIERS and kind in x3.IDENTITY_KINDS, f"--x3-carriers-kind {kind!r}: one of {x3.IDENTITY_KINDS}"
        if kind == "identity3" or kind in x3.WAVE4_KINDS: x3.assert_no_overlap(kind)  # A12 / A15 / A17
        ov = x3.carrier_overlap(kind)
        if ov: log(f"WARNING carrier kind {kind}: {len(ov)} exemplar / item overlaps (A12): {sorted({e for e, _, _ in ov})}")
        car = [tok(s, return_tensors="pt").input_ids.to(dev) for s in pick(x3.CARRIERS[kind])]
        specs.append((f"x3_{kind}", kind, car, [c.shape[1] - 1 for c in car], a.x3_gen, kind))
    if "x3c" in readers:
        x3.assert_no_overlap("continuation"); car = [tok(s, return_tensors="pt").input_ids.to(dev) for s in pick(x3.CONTINUATION)]
        specs.append(("x3c", "continuation", car, [c.shape[1] - 1 for c in car], a.x3c_gen, None))
    seed_car = []
    if "x3s" in readers:
        x3.assert_no_overlap("continuation_seed"); seed_car = [tok(s, return_tensors="pt").input_ids.to(dev) for s in x3.CONTINUATION_SEED]
        for sl in seed_lenses: specs.append((f"x3s_{sl}", "continuation_seed", seed_car, [c.shape[1] - 1 for c in seed_car], a.x3s_gen, None))
    for prefix, kind, car, cpos, n_gen, cutk in specs:
        for l in x3_layers:
            for t in x3_targets:
                tl = l if t == "same" else int(t)
                for ci, (cid, cp) in enumerate(zip(car, cpos)):
                    if (kind, ci, tl) not in xclean: xclean[(kind, ci, tl)] = backend.resid(cid, tl)[0, cp].float().clone()
        log(f"{prefix}: {len(car)} {kind} carriers, targets {x3_targets} for read layers {x3_layers}, {n_gen} tokens; stops {x3.STOPS.get(kind)}; patch tokens {[tok.decode([int(c[0, p_])]) for c, p_ in zip(car, cpos)][:3]}")
    # A20 no-patch baseline: every carrier generated ONCE with no h (item-independent); the same samples are written for
    # every item so the bench scorer says which items' targets the bare carrier produces (arm <prefix>_nopatch[_uncut]).
    nopatch = {}
    if "nopatch" in ctrl_names:
        for prefix, kind, car, cpos, n_gen, cutk in specs:
            if prefix.startswith("x3s"): continue  # the seeded arm's samples depend on h through the seeds
            raw = [tok.decode(backend.greedy(cid, n_gen)) for cid in car]
            nopatch[prefix] = ([x3.cut(t, cutk) if cutk else t for t in raw], raw if cutk else None)
            log(f"{prefix}_nopatch (A20): bare carriers -> {raw[:3]}")
    special_ids = list(getattr(tok, "all_special_ids", []) or [])
    results = {"model": a.model, "backend": backend.name, "dtype": a.dtype, "bench_commit": R.BENCH_COMMIT, "scorer": getattr(breg, "SCORER_VERSION", R.SCORER_VERSION), "token_bags": "raw regex only (no summarizer)",
               "producer": "wsbench" if bench_lens else "vendored formulas", "layers": layers, "layers_requested": layers0, "p3_layers": p3_layers, "x3_layers": x3_layers, "first": a.first, "first_layer": a.first_layer, "tau": tau, "rank": a.rank, "readers": readers, "controls": a.controls, "no_gate": bool(a.no_gate),
               "x3": {"kind": a.x3_carriers_kind, "kinds": kinds, "targets": [t for t in x3_targets], "n_carriers": len(car_ids), "carrier_ids": car_ids, "gen": a.x3_gen, "carriers": pick(x3.CARRIERS[kinds[0]]), "stops": x3.STOPS[kinds[0]], "carriers_by_kind": {k: pick(x3.CARRIERS[k]) for k in kinds}, "stops_by_kind": {k: x3.STOPS[k] for k in kinds}, "uncut_arms": not a.no_uncut_arms, "positions": a.x3_positions} if kinds else None,
               "x3c": {"carriers": pick(x3.CONTINUATION), "carrier_ids": car_ids, "gen": a.x3c_gen, "targets": [t for t in x3_targets]} if "x3c" in readers else None,
               "x3s": {"frames": x3.CONTINUATION_SEED, "seed_lenses": seed_lenses, "k": SEED_K, "gen": a.x3s_gen, "targets": [t for t in x3_targets]} if "x3s" in readers else None,
               "nopatch": {k: {"samples": v[0], "raw": v[1]} for k, v in nopatch.items()} if nopatch else None,
               "gate": {"surface": "greedy plain continuation contains target / target_alts (bench matcher)", "immediate": "folded greedy continuation starts with target / alias after an optional article (A15 / A16; not a gate condition)", "bridges": "each bridge question (chat template, thinking off) greedy contains an answer", "draws": a.gate_draws, "temp": a.gate_temp, "rate": a.gate_rate, "tokens": a.gate_tokens},
               "prereg": "docs/prereg_R.yaml amendments A11 (P6 pilot; report only), A11c (rank), A12 (identity3), A15 (x3c / x3s / identity3_long), A16 (--no-gate; three denominators), A17 (spellfix; span positions), A19 (--x3-layers), A20 (crosscat / random / nopatch controls; primary-only column), A22 (--x3-positions all)", "families": {}}
    cmds = ["#!/bin/bash", "# `wsbench judge` on the readouts files below (run inside the workspace-bench checkout, PYTHONPATH=src or uv run).",
            "# Prose arms (chains, p3, x3*) need no API key. Token arms (jlens, logitlens) are summarized by Gemini first (OPENROUTER_API_KEY through the bench;",
            "# with a DIRECT Gemini key use scripts/wsbench_judge_gemini.py, the bench's summarize-then-regex protocol: the last lines below; GEMINI_KEY_FILE=<path>)."]
    gcmds = []
    for fam in [f for f in a.families.split(",") if f]:
        t0 = time.time(); bank = os.path.join(a.wsbench_root, "evals", fam, "items.json") if a.wsbench_root and os.path.exists(os.path.join(a.wsbench_root, "evals", fam, "items.json")) else os.path.join(a.bank_root, fam + ".json")
        header, items = R.load_bank(bank); contract = breg.contract_for(header, header.get("family", fam))
        units_of = {it["id"]: breg.scored_units(it, contract) for it in items}  # validates the bank against its contract
        if a.limit: items = items[: a.limit]
        gated, gate_rows = [], []
        for it in items:
            ids = torch.tensor([tok(it["prompt"], add_special_tokens=False)["input_ids"]], device=dev); it["_ids"] = ids; it["_pos"] = ids.shape[1] - R.read_offset(it)
            gr = gate_item(backend, tok, m, it, ids, a.gate_tokens, a.gate_draws, a.gate_temp, a.gate_rate); gate_rows.append({"id": it["id"], **gr, "subfamily": it.get("subfamily"), "target": it.get("target")})
            if gr["ok"]: gated.append(it)
        gated_ids = {it["id"] for it in gated}; immediate_ids = {r["id"] for r in gate_rows if r["immediate"]}
        scored = items if a.no_gate else gated
        log(f"{fam}: {len(items)} items, {len(gated)} pass our gate ({sum(r['surface']['ok'] for r in gate_rows)} surface, {len(immediate_ids)} immediate, {sum(all(b['ok'] for b in r['bridges']) for r in gate_rows if r['bridges'])} all-bridges of {sum(bool(r['bridges']) for r in gate_rows)}); scoring {len(scored)} ({'all items, --no-gate' if a.no_gate else 'gated'})")
        fres = {"bank": bank, "n_items": len(items), "n_gated": len(gated), "n_immediate": len(immediate_ids), "n_scored": len(scored), "no_gate": bool(a.no_gate), "gate_rows": gate_rows, "contract": contract.__dict__ if hasattr(contract, "__dict__") else str(contract), "arms": {}}
        if not scored:
            results["families"][fam] = fres; json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False); continue
        # read positions: the final token (pos), plus the earlier pieces of the typo span with --x3-positions span
        for it in scored:
            T_ = it["_ids"].shape[1]
            if a.x3_positions == "span": extra = [q for q in typo_span(tok, it["_ids"], it.get("typo_word")) if q < it["_pos"]]
            elif a.x3_positions == "all": extra = [q for q in range(max(0, T_ - a.x3_pos_max) if a.x3_pos_max else 0, it["_pos"])]  # A22: every earlier prompt position
            else: extra = []
            it["_offs"] = {1: it["_pos"], **{T_ - q: q for q in extra}}  # offset from the end -> absolute position
        H = {l: {} for l in need}
        for it in scored:
            for l in need:
                hl = backend.resid(it["_ids"], l)[0]; H[l][it["id"]] = {q: hl[q].float().clone() for q in it["_offs"].values()}
        subs = [str(it.get("subfamily")) for it in scored]; tgts = [str(it.get("target")) for it in scored]
        perm = e12.shuffled_perm(subs, a.seed, tgts) if "shuffled" in ctrl_names else None
        xperm = e12.cross_perm(subs, a.seed, tgts) if "crosscat" in ctrl_names else None  # A2 / A20: partner from another subfamily
        ctrls = ["none"] + [c for c in ("shuffled", "crosscat", "random") if c in ctrl_names]
        fres_ctrl = {"n_shuffled_dropped": int((perm < 0).sum()) if perm is not None else None, "n_crosscat_dropped": int((xperm < 0).sum()) if xperm is not None else None}
        gr = torch.Generator().manual_seed(a.seed + 7919 * (1 + R.MT_FAMILIES.index(fam) if fam in R.MT_FAMILIES else 0))
        rows = {}  # (arm, ctrl) -> list of contract rows
        prows = {}  # (arm, ctrl) -> list of stored-prefix rows (p3, chains)
        for n, it in enumerate(scored):
            tokstr = tok.decode([int(it["_ids"][0, it["_pos"]])]); pos = int(it["_pos"])
            for ctrl in ctrls:
                # the source vectors of this condition: srcH[l][position] (the item's own, a partner's, or norm-matched Gaussian), with the source's own offsets / last position
                if ctrl == "none": src = it; srcH = {l: H[l][it["id"]] for l in need}
                elif ctrl == "random":  # A1 / A20: a Gaussian direction with the norm of the item's own h at that (layer, position); seeded per family
                    src = it; srcH = {}
                    for l in need:
                        srcH[l] = {}
                        for q_, hv in H[l][it["id"]].items():
                            r0 = torch.randn(hv.shape, generator=gr, dtype=torch.float32).to(hv.device); srcH[l][q_] = r0 / r0.norm().clamp_min(1e-12) * hv.norm()
                else:
                    pm = perm if ctrl == "shuffled" else xperm; j = int(pm[n])
                    if j < 0: continue
                    src = scored[j]; srcH = {l: H[l][src["id"]] for l in need}
                hs = {l: srcH[l][src["_pos"]] for l in need}
                for l in layers:
                    h = hs[l]
                    for rd in ("jlens", "logitlens"):
                        if rd not in readers or (rd == "jlens" and l not in jl_layers): continue
                        if rd in bench_lens:
                            ro = bench_lens[rd].read(h.detach().float().cpu(), l); row = {"id": it["id"], "layer": l, "pos": pos, "token": tokstr, "tokens": list(ro.tokens), "scores": list(ro.scores)}
                        else:
                            z = R.jlens_cosine_scores(Dl(l), h) if rd == "jlens" else backend.logits(h); top = torch.topk(z.float(), R.TOP_K); row = token_row(tok, it["id"], l, pos, tokstr, top.indices.tolist(), top.values.tolist())
                        rows.setdefault((rd, ctrl), []).append(row)
                    for rd in ("jlens_chain", "logitlens_chain"):
                        if rd not in readers or (rd == "jlens_chain" and l not in jl_layers): continue
                        z = R.jlens_cosine_scores(Dl(l), h).double() if rd == "jlens_chain" else backend.logits(h).double()
                        cands, pre = x7.ordered_bag_chain(tok, z, z, ok, a.k1, a.max_pieces)
                        rows.setdefault((rd, ctrl), []).append(prose_row(it["id"], l, pos, tokstr, samples_of(rd, pre, a.rank, tau)))
                        prows.setdefault((rd, ctrl), []).append(prefix_row(it["id"], l, pos, tokstr, it.get("target"), cands, pre))
                for l in read_layers:
                    h = hs[l]; hf = hs[lf] if (lf is not None and lf in hs) else h
                    if "p3" in readers and l in p3_layers:
                        zf = x7.first_scores(a.first, backend, hf, Dl(lf if lf is not None else l) if a.first != "logitlens" else None)
                        res = x7.chain_read(backend, tok, ids_ctx, l, h, zf, ok, xlast[l], cache, a.k1, a.beam, a.max_pieces, a.chunk)
                        rows.setdefault(("p3", ctrl), []).append(prose_row(it["id"], l, pos, tokstr, samples_of("p3", res["prefixes"], a.rank, tau)))
                        rows.setdefault(("p3_top1", ctrl), []).append(prose_row(it["id"], l, pos, tokstr, samples_of("p3_top1", res["prefixes"], a.rank, tau)))
                        prows.setdefault(("p3", ctrl), []).append(prefix_row(it["id"], l, pos, tokstr, it.get("target"), res["cands"], res["prefixes"]))
                    if l not in x3_layers: continue
                    if ctrl == "none" and nopatch:  # A20: the bare-carrier samples, written for every item at every x3 read layer (final token only)
                        for prefix, (cut_s, raw_s) in nopatch.items():
                            rows.setdefault((prefix + "_nopatch", "none"), []).append(prose_row(it["id"], l, pos, tokstr, cut_s))
                            if raw_s is not None and not a.no_uncut_arms: rows.setdefault((prefix + "_nopatch_uncut", "none"), []).append(prose_row(it["id"], l, pos, tokstr, raw_s))
                    for off, q in sorted(it["_offs"].items()):  # the final token (off 1) and, with span / all positions, the earlier pieces (arms _off<k>)
                        # control: the partner's own token at the same offset from ITS end (clipped to its last token; A14 rule); random: the item's own position
                        hq = hs[l] if off == 1 else srcH[l].get(src["_offs"].get(off), srcH[l][src["_pos"]])
                        tq = tok.decode([int(it["_ids"][0, q])]); sfx = "" if off == 1 else f"_off{off}"
                        for prefix, kind, car, cpos, n_gen, cutk in specs:
                            seeds = None
                            if prefix.startswith("x3s"):
                                sl = prefix.split("_")[1]
                                if sl == "jl" and l not in jl_layers: continue
                                z = R.jlens_cosine_scores(Dl(l), hq) if sl == "jl" else backend.logits(hq); seeds = lens_seeds(z, SEED_K, special_ids)
                            for t in x3_targets:
                                tl = l if t == "same" else int(t); arm = arm_of(prefix, t, l) + sfx; gens, raw = [], []
                                for ci, (cid, cp) in enumerate(zip(car, cpos)):
                                    if seeds is None:
                                        g_ = x3_generate(backend, cid, cp, hq, tl, n_gen, xclean[(kind, ci, tl)]); txt = tok.decode(g_); raw.append(txt); gens.append(x3.cut(txt, cutk) if cutk else txt)
                                    else:
                                        for sd in seeds: gens.append(tok.decode(seeded_generate(backend, cid, cp, hq, tl, [sd], n_gen, xclean[(kind, ci, tl)])))
                                rows.setdefault((arm, ctrl), []).append(prose_row(it["id"], l, q, tq, gens))
                                if cutk and not a.no_uncut_arms: rows.setdefault((arm + "_uncut", ctrl), []).append(prose_row(it["id"], l, q, tq, raw))
            if n % 10 == 0 or n == len(scored) - 1:
                x3r = [v for (arm, c), v in rows.items() if arm.startswith("x3") and c == "none" and not arm.endswith("_uncut")]; p3r = rows.get(("p3", "none"), [])
                log(f"{fam} {n + 1}/{len(scored)} {it['id']}: target {it.get('target')!r}; x3 {x3r[-1][-1]['samples'][:3] if x3r else None}; p3 {p3r[-1]['samples'][:3] if p3r else None}; jlens {rows.get(('jlens', 'none'), [{}])[-1].get('tokens', [])[:5]}")
        # write the prefixes (offline re-ranking), the readouts files, and score every arm
        for (arm, ctrl), rs in prows.items():
            sub = arm + ("" if ctrl == "none" else "__" + ctrl); d = os.path.join(out_dir, "prefixes", sub); os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, fam + ".jsonl"), "w", encoding="utf-8") as fh:
                for r in rs: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        for (arm, ctrl), rs in rows.items():
            sub, f = write_readouts(out_dir, arm, ctrl, fam, rs)
            arm_layers = x3_layers if arm.startswith("x3") else p3_layers if arm in ("p3", "p3_top1") else (jl_layers if arm.startswith("jlens") else layers)
            if arm.startswith("x3s_jl"): arm_layers = [l for l in x3_layers if l in jl_layers]
            e = score_arm(f, sub, arm, scored, contract, arm_layers, read_layers, a.seed, gated_ids, immediate_ids, a.no_gate); e["file"] = os.path.relpath(f, os.path.dirname(HERE)); fres["arms"][sub] = e
            if ctrl == "none":
                ids_arg = ",".join(it["id"] for it in scored)
                cmds.append(f"uv run wsbench judge family={fam} readouts={os.path.abspath(f)} out=outputs/judged/{a.tag}/{sub}/{fam} items={ids_arg}")
            dn = e["denominators"]
            log(f"{fam} {sub:24s}: pass (any layer, raw regex) {e['pass_rate']:.3f} (n {e['n_items_decided']}), any_hit {e['any_hit_rate']:.3f}, by layer " + ", ".join(f"L{l} {v:.2f}" for l, v in e["pass_rate_by_layer"].items() if v is not None)
                + (f"; on read layers {e['on_read_layers']['pass_rate']:.3f}" if e.get("on_read_layers") else "") + f"; all {_fmt(dn['all']['pass'])} (n {dn['all']['n']}) gated {_fmt(dn['gated']['pass'])} (n {dn['gated']['n']}) immediate {_fmt(dn['gated_immediate']['pass'])} (n {dn['gated_immediate']['n']}); primary-only {_fmt(e['primary_only']['pass_rate'])}")
        fres.update(fres_ctrl); fres["position_profile"] = position_profile(fres["arms"]); fres["nopatch_withdrawal"] = nopatch_withdrawal(fres["arms"])
        for sub, w in fres["nopatch_withdrawal"].items():
            if w["withdrawn_ids"]: log(f"{fam} {sub}: A20 no-patch baseline passes {w['n_nopatch_pass']} items; withdrawn from this arm: {w['withdrawn_ids']} -> pass after withdrawal {_fmt(w['pass_rate_after_withdrawal'])}")
        tarms = [s for s in fres["arms"] if s in ("jlens", "logitlens")]
        if tarms: gcmds.append(f"GEMINI_KEY_FILE=$HOME/sjlens/.gemini_key python scripts/wsbench_judge_gemini.py --run runs/{a.tag} --family {fam} --arms {','.join(tarms)}")
        fres["seconds"] = time.time() - t0; results["families"][fam] = fres
        json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
    open(os.path.join(out_dir, "judge_cmds.sh"), "w").write("\n".join(cmds + ["# direct Gemini key (login node, proxy): the summarized token-arm column"] + gcmds) + "\n")
    finish(results, out_f, log)


if __name__ == "__main__":
    main()
