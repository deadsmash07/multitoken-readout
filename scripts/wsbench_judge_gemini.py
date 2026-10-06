"""The bench's summarized token-arm number with a DIRECT Gemini API key (login node, API calls only, no model compute).

WHY THIS EXISTS. WorkspaceBench scores a token lens (J-lens / logit lens top-10 bag) in two steps: (1) ONE shared
"blind interpretation" call turns the bag into one or two sentences (src/wsbench/summarizer.py, prompt `interp-v1`),
(2) the regex contract is applied to that interpretation as ONE more sample beside the raw tokens
(src/wsbench/multitoken/family.py::run_regex, `mt-regex-summarized-2026-09-23`). The bench's client
(src/wsbench/llm.py) routes every non-`claude-*` model to OpenRouter and `api_key()` REJECTS any key that does not
start with `sk-or-` ("OPENROUTER_API_KEY is missing or not an OpenRouter key"), so `wsbench judge` cannot use a
direct Gemini key. This script reproduces the protocol against the Gemini REST API with the same prompt, schema,
model, bundle text, cache key / fingerprint and scoring; every element is quoted from commit 92d763e (2026-09-23):
  model      judge_config.py: DEFAULT_JUDGE = "google/gemini-3.8-flash" (OpenRouter id) -> Gemini API id
             "gemini-3.8-flash" (--model); DEFAULT_REASONING = {"effort": "minimal"} ("Gemini cannot turn reasoning
             off") -> generationConfig.thinkingConfig.thinkingLevel (--thinking-level minimal; dropped with a note if
             the API rejects it, and recorded in results.json); no temperature is sent (summarize() passes none);
             max_tokens 8000 (llm._DEFAULT_MAX_TOKENS["openrouter"]).
  prompts    summarizer.py INTERP_SYSTEM / INTERP_USER, verbatim below (docs/summarizer.md "Prompt").
  schema     summarizer.py INTERP_SCHEMA = schema_block("interp", {"interpretation": {"type": "string"}},
             ["interpretation"]) (strict, additionalProperties false) -> responseMimeType application/json +
             responseJsonSchema.
  bundle     summarizer.render_bag(tokens, scores): "tok (score)" sorted best first, 2 dp, joined by " | ".
  cache      <out>/cells.jsonl rows {"key": "summ:<id>__L<layer:03d>__p<pos>", "fp": sha256(json.dumps(
             ["interp-v1", model, bundle], sort_keys)[:16], "result": {...} | null} (cache.py fingerprint,
             readouts.Cell.key), so a bench checkout could reuse them.
  scoring    samples = the raw token strings + the interpretation (family._cell_samples); a failed / empty summary
             leaves the cell unsummarized and the item UNDECIDED (None) unless another layer passes (tri_state);
             `passing_layers_by_source` (bag_only / summary_only / both / mixed) as in run_regex; the raw-regex number
             (tokens only) is recomputed beside it. Plus the three denominators of x8 (all / gated / gated-immediate).
KEY. Read at run time from the file named by $GEMINI_KEY_FILE (chmod 600; e.g. ~/sjlens/.gemini_key) or from
$GEMINI_API_KEY; never printed, logged or written. Network through the login node's proxy: urllib honours
HTTPS_PROXY / https_proxy (export https_proxy=<your proxy> if needed).
Usage (from sjlens/): GEMINI_KEY_FILE=~/sjlens/.gemini_key python scripts/wsbench_judge_gemini.py --run runs/<tag>
  --family basic_readout_mt --arms jlens,logitlens [--limit 5] [--dry-run] [--rpm 30]
Output: runs/<tag>/judged_gemini/<arm>/<family>/{cells.jsonl,results.json}; the summarized pass rate is
results.json["pass_rate"] (bench rule), the raw one results.json["raw"]["pass_rate"]."""
import argparse, hashlib, importlib.util, json, os, random, sys, time, urllib.error, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens import wsbench_regex as R

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
SUMMARIZER_PROMPT_VERSION = "interp-v1"  # summarizer.py
INTERP_SYSTEM = ("You are shown the top-10 token readouts from an interpretability lens at one position inside a language model that was "
                 "reading a passage you cannot see. Tokens may include noise, fragments, other languages (translate them), or byte artifacts. "
                 "In one or two sentences, state what these outputs are collectively trying to say — the situation or mental content they "
                 "point to. Commit to the most specific reading the tokens support; do not just say they are noisy.")  # summarizer.py INTERP_SYSTEM (verbatim)
INTERP_USER = "TOKEN READOUTS:\n{txt}"  # summarizer.py INTERP_USER
INTERP_SCHEMA = {"name": "interp", "strict": True, "schema": {"type": "object", "additionalProperties": False, "required": ["interpretation"], "properties": {"interpretation": {"type": "string"}}}}  # llm.schema_block
DEFAULT_MODEL = "gemini-3.8-flash"  # judge_config.DEFAULT_JUDGE "google/gemini-3.8-flash" without the OpenRouter vendor prefix
DEFAULT_THINKING = "minimal"  # judge_config.DEFAULT_REASONING {"effort": "minimal"}
MAX_TOKENS = 8000  # llm._DEFAULT_MAX_TOKENS["openrouter"]
TRANSIENT = (408, 409, 429, 500, 502, 503, 504, 529); ATTEMPTS = 12  # llm._TRANSIENT / _ATTEMPTS
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
OR_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"  # the bench's own route (llm.py -> OpenRouter)
OR_MODEL = {"gemini-3.8-flash": "google/gemini-3.8-flash"}  # judge_config.DEFAULT_JUDGE
OR_MAX_TOKENS = int(os.environ.get("OR_MAX_TOKENS", "1024"))  # DEVIATION from the bench's 8000: OpenRouter reserves credit for
# max_tokens per in-flight request, so 8000 x many parallel requests exceeds a small balance (402). Real summaries are
# 40-160 tokens (reasoning "minimal"), so a 1024 cap does not change any output; recorded in results.json config_used.


def read_or_key():
    """OpenRouter key from $OPENROUTER_KEY_FILE (chmod 600) or $OPENROUTER_API_KEY, or None; never echoed. Used only
    after every Gemini key is over quota."""
    f = os.environ.get("OPENROUTER_KEY_FILE", "")
    if f and os.path.exists(os.path.expanduser(f)):
        k = open(os.path.expanduser(f), encoding="utf-8").read().strip()
        if k: return k
    return os.environ.get("OPENROUTER_API_KEY", "").strip() or None


def render_user(txt):
    return INTERP_USER.replace("{txt}", txt)


def render_bag(tokens, scores):
    """summarizer.render_bag: best first, 2 dp, " | "-joined; tokens alone without scores."""
    if scores is None: return " | ".join(tokens)
    pairs = sorted(zip(tokens, scores), key=lambda p: -p[1])
    return " | ".join(f"{t} ({s:.2f})" for t, s in pairs)


def fingerprint(*parts):
    """cache.fingerprint: sha256 of the JSON list, first 16 hex digits."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def cell_key(item_id, layer, pos):
    """readouts.Cell.key."""
    return f"{item_id}__L{layer:03d}__p{pos}"


def read_keys():
    """the API key(s) from $GEMINI_KEY_FILE (a chmod-600 file, one key per line; used in order until each hits its
    quota) or $GEMINI_API_KEY; never echoed."""
    f = os.environ.get("GEMINI_KEY_FILE", "")
    if f:
        ks = [l.strip() for l in open(os.path.expanduser(f), encoding="utf-8") if l.strip()]
        if ks: return ks
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if not k: raise SystemExit("no key: set GEMINI_KEY_FILE=<path to a chmod-600 file holding the key(s)> (or GEMINI_API_KEY)")
    return [k]


class QuotaExhausted(Exception):
    """every key returned a quota 429 (not a transient rate limit): stop cleanly, keep the cache, rerun later."""


def read_key():
    return read_keys()[0]


def request_body(txt, thinking_level, schema_mode="json_schema"):
    """the generateContent body: system instruction + user turn, JSON output under the interp schema, no temperature."""
    gc = {"responseMimeType": "application/json", "maxOutputTokens": MAX_TOKENS}
    if schema_mode == "json_schema": gc["responseJsonSchema"] = INTERP_SCHEMA["schema"]
    elif schema_mode == "openapi": gc["responseSchema"] = {"type": "OBJECT", "properties": {"interpretation": {"type": "STRING"}}, "required": ["interpretation"]}
    if thinking_level and thinking_level != "none": gc["thinkingConfig"] = {"thinkingLevel": thinking_level}
    return {"systemInstruction": {"parts": [{"text": INTERP_SYSTEM}]}, "contents": [{"role": "user", "parts": [{"text": render_user(txt)}]}], "generationConfig": gc}


def parse_interpretation(resp):
    """{"interpretation": str} from a generateContent response, or None (empty / garbled = a failed summary)."""
    try:
        parts = resp["candidates"][0]["content"]["parts"]; text = "".join(p.get("text", "") for p in parts).strip()
        if text.startswith("```"): text = text.strip("`"); text = text[4:] if text.startswith("json") else text
        d = json.loads(text)
        return d if isinstance(d, dict) and isinstance(d.get("interpretation"), str) and d["interpretation"].strip() else None
    except (KeyError, IndexError, TypeError, ValueError):
        return None


class Gemini:
    """one sequential REST client with the bench's pacing / backoff shape; the config actually accepted by the API is
    recorded in `used` (thinking level and schema mode may be downgraded on a 400)."""
    def __init__(s, key, model, thinking_level, rpm, timeout=180.0, post=None):
        s.keys = list(key) if isinstance(key, (list, tuple)) else [key]; s.ki = 0; s.key = s.keys[0]
        s.or_key = read_or_key(); s.mode = "openrouter" if (s.or_key and os.environ.get("JUDGE_ROUTE", "") == "openrouter") else "gemini"
        s.model, s.rpm, s.timeout = model, rpm, timeout; s.used = {"thinking_level": thinking_level, "schema_mode": "json_schema"}
        if s.mode == "openrouter": s.used["route"] = "openrouter"; s.used["or_max_tokens"] = OR_MAX_TOKENS
        s.spend = {"calls": 0, "retries": 0, "errors": 0, "prompt_tokens": 0, "output_tokens": 0}; s._next = 0.0; s.post = post or s._post

    def _post(s, body):
        req = urllib.request.Request(ENDPOINT.format(model=s.model), data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json", "x-goog-api-key": s.key}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=s.timeout) as r: return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try: msg = e.read().decode("utf-8", "replace")
            except Exception: msg = ""
            return e.code, {"error": msg[:600]}

    def _post_or(s, txt):
        """the same summary through OpenRouter (bench route): system + user turn, strict JSON schema, reasoning minimal."""
        body = {"model": OR_MODEL.get(s.model, s.model), "max_tokens": OR_MAX_TOKENS,
                "messages": [{"role": "system", "content": INTERP_SYSTEM}, {"role": "user", "content": render_user(txt)}],
                "response_format": {"type": "json_schema", "json_schema": INTERP_SCHEMA}, "reasoning": {"effort": "minimal"}}
        req = urllib.request.Request(OR_ENDPOINT, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json", "Authorization": "Bearer " + s.or_key, "X-Title": "sjlens-wsbench-judge"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=s.timeout) as r: status, resp = r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try: msg = e.read().decode("utf-8", "replace")
            except Exception: msg = ""
            return e.code, {"error": msg[:600]}
        if status == 200 and "choices" in resp:  # reshape to the Gemini response so parse_interpretation / spend work unchanged
            u = resp.get("usage") or {}
            resp = {"candidates": [{"content": {"parts": [{"text": resp["choices"][0]["message"].get("content") or ""}]}}],
                    "usageMetadata": {"promptTokenCount": u.get("prompt_tokens", 0), "candidatesTokenCount": u.get("completion_tokens", 0)}}
        return status, resp

    def _pace(s):
        now = time.monotonic(); wait = s._next - now
        if wait > 0: time.sleep(wait)
        s._next = max(now, s._next) + 60.0 / s.rpm

    def summarize(s, txt):
        for attempt in range(ATTEMPTS):
            s._pace(); body = request_body(txt, s.used["thinking_level"], s.used["schema_mode"])
            try: status, resp = s._post_or(txt) if s.mode == "openrouter" else s.post(body)
            except Exception as e:  # timeouts / connection errors: transient
                status, resp = 0, {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            if status == 200:
                s.spend["calls"] += 1; um = resp.get("usageMetadata") or {}; s.spend["prompt_tokens"] += int(um.get("promptTokenCount", 0) or 0); s.spend["output_tokens"] += int(um.get("candidatesTokenCount", 0) or 0)
                return parse_interpretation(resp)
            err = str(resp.get("error", ""))
            if status == 400 and s.used["thinking_level"] != "none" and "thinking" in err.lower():
                print(f"  gemini: thinkingConfig {s.used['thinking_level']!r} rejected; dropping it for this run (recorded)"); s.used["thinking_level"] = "none"; continue
            if status == 400 and s.used["schema_mode"] == "json_schema" and ("schema" in err.lower() or "unknown name" in err.lower()):
                print("  gemini: responseJsonSchema rejected; falling back to responseSchema (OpenAPI subset; recorded)"); s.used["schema_mode"] = "openapi"; continue
            if status in (401, 403, 404) or (status == 400 and "api key" in err.lower()): raise SystemExit(f"fatal gemini error {status}: {err[:300]}")
            if status == 402 and s.mode == "openrouter" and "in-flight" in err.lower() and attempt < ATTEMPTS - 1:  # credit reservation: wait, retry
                s.spend["retries"] += 1; time.sleep(5.0 + 10.0 * random.random()); continue
            if status == 429 and s.mode == "gemini" and ("quota" in err.lower() or "exceeded" in err.lower()):  # daily / project quota, not a rate blip
                if s.ki + 1 < len(s.keys):
                    s.ki += 1; s.key = s.keys[s.ki]; print(f"  gemini: key {s.ki} of {len(s.keys)} over quota; switching to key {s.ki + 1}"); continue
                if s.mode == "gemini" and s.or_key:
                    s.mode = "openrouter"; s.used["route"] = "openrouter"; s.used["or_max_tokens"] = OR_MAX_TOKENS; print("  gemini: all keys over quota; switching to OpenRouter (google/gemini-3.8-flash)"); continue
                raise QuotaExhausted(f"all {len(s.keys)} keys over quota" + (" and OpenRouter limit reached" if s.mode == "openrouter" else ""))
            if (status in TRANSIENT or status == 0) and attempt < ATTEMPTS - 1:
                s.spend["retries"] += 1; time.sleep(min(90.0 if status == 429 else 30.0, 2.0 * 2 ** attempt) * (0.5 + random.random())); continue
            if s.mode == "gemini" and s.or_key and (status in TRANSIENT or status == 0):  # Gemini free tier stuck on 503s: use the bench route
                s.mode = "openrouter"; s.used["route"] = "openrouter"; print(f"  gemini: still {status} after {ATTEMPTS} attempts; switching to OpenRouter (google/gemini-3.8-flash)"); return s.summarize(txt)
            s.spend["errors"] += 1; print(f"  gemini error {status}: {err[:200]}"); return None
        return None


def load_cache(path):
    out = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try: r = json.loads(line); out[(r["key"], r["fp"])] = r
            except Exception: continue
    return out


def score(cells, items, contract, layers, summaries, seed=0, gated_ids=None, immediate_ids=None, no_gate=False):
    """family.run_regex on our cells: samples = raw tokens + the interpretation; a cell whose summary failed leaves the
    item undecided unless another layer passes. Returns the summarized block, with the raw (tokens-only) number and
    the passing-layers source split beside it."""
    units_of = {it["id"]: R.scored_units(it, contract) for it in items}; by = {(c["id"], c["layer"]): c for c in cells}
    rows = []; src_by = {}
    for it in items:
        hits, missing, unsum, empty = {}, [], [], []
        for l in layers:
            c = by.get((it["id"], l))
            if c is None: missing.append(l); continue
            key = cell_key(c["id"], c["layer"], c["pos"]); summ = summaries.get(key)
            if summ is None: unsum.append(l)
            raw = R.cell_samples(c); samples = raw + ([R.extract_phrase(summ)] if summ is not None else []); samples = [x for x in samples if x]
            if not samples: empty.append(l)
            u = units_of[it["id"]]; hits[l] = R.layer_unit_hits(samples, u)
            rh = R.layer_unit_hits(raw, u); sh = R.layer_unit_hits([R.extract_phrase(summ)], u) if summ is not None else {}
            src_by[(it["id"], l)] = {x.role: [s_ for s_, h in (("bag", rh), ("summary", sh)) if h.get(x.role)] for x in u}
        res = R.item_result(hits, units_of[it["id"]]); incomplete = bool(missing) or bool(unsum) or not hits
        rows.append({"id": it["id"], "roles": [u.role for u in units_of[it["id"]] if u.required], "optional_roles": [u.role for u in units_of[it["id"]] if not u.required],
                     "pass": True if res["pass"] else (None if incomplete else False), "earliest_layer": res["earliest_layer"], "passing_layers": res["passing_layers"], "unit_hit": res["unit_hit"], "unit_langs": res["unit_langs"], "any_hit": res["any_hit"],
                     "missing_layers": missing, "unsummarized_layers": unsum, "empty_layers": empty, "layers": {str(l): {"hits": h, "source": src_by.get((it["id"], l), {})} for l, h in sorted(hits.items())}})
    decided = [r for r in rows if r["pass"] is not None]; rate = lambda xs: (sum(xs) / len(xs)) if xs else None
    roles = sorted({x for r in rows for x in r["roles"] + r["optional_roles"]}); src = {"both": 0, "bag_only": 0, "summary_only": 0, "mixed": 0}
    for r in decided:
        for l in r["passing_layers"]:
            req = [src_by[(r["id"], l)].get(x, []) for x in r["roles"]]
            src["both" if all("bag" in x for x in req) and all("summary" in x for x in req) else "bag_only" if all("bag" in x for x in req) else "summary_only" if all("summary" in x for x in req) else "mixed"] += 1
    raw_sc = R.score_cells(cells, items, contract, layers)
    ids = {it["id"] for it in items}; g = set(gated_ids if gated_ids is not None else ids) & ids; im = set(immediate_ids or ()) & g
    sub = lambda sel: {"n": len([r for r in rows if r["id"] in sel]), "n_decided": len([r for r in decided if r["id"] in sel]), "pass": rate([float(bool(r["pass"])) for r in decided if r["id"] in sel]), "any_hit": rate([float(r["any_hit"]) for r in decided if r["id"] in sel])}
    return {"scorer_version": R.SUMMARIZED_SCORER_VERSION, "summarizer": SUMMARIZER_PROMPT_VERSION, "n_items": len(rows), "n_items_decided": len(decided), "pass_rate": rate([float(bool(r["pass"])) for r in decided]), "any_hit_rate": rate([float(r["any_hit"]) for r in decided]),
            "pass_rate_by_layer": {str(l): rate([float(l in r["passing_layers"]) for r in decided]) for l in layers}, "unit_any_layer": {x: rate([float(r["unit_hit"][x]) for r in decided if x in r["unit_hit"]]) for x in roles},
            "passing_layers_by_source": src, "n_unsummarized_items": sum(1 for r in rows if r["unsummarized_layers"]),
            "denominators": {"all": sub(ids) if no_gate else {"n": None, "pass": None, "note": "gated run"}, "gated": sub(g), "gated_immediate": sub(im)},
            "raw": {"scorer_version": R.SCORER_VERSION, "pass_rate": raw_sc["pass_rate"], "any_hit_rate": raw_sc["any_hit_rate"], "n_items_decided": raw_sc["n_items_decided"]}, "rows": rows}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True, help="runs/<tag> of an x8_wsbench_items.py run (results.json + readouts/)"); p.add_argument("--family", required=True); p.add_argument("--arms", default="jlens,logitlens", help="token arms (comma list; controls as <arm>__shuffled)")
    p.add_argument("--model", default=DEFAULT_MODEL); p.add_argument("--thinking-level", default=DEFAULT_THINKING, help="generationConfig.thinkingConfig.thinkingLevel (the bench's reasoning effort 'minimal'); none = omit")
    p.add_argument("--layers", default="", help="comma list; default: the arm's layers in results.json"); p.add_argument("--items", default="", help="comma list of item ids; default: every scored item of the run")
    p.add_argument("--limit", type=int, default=0, help="only the first N cells (smoke)"); p.add_argument("--rpm", type=float, default=30.0); p.add_argument("--workers", type=int, default=1, help="concurrent summary calls (one cell per call)"); p.add_argument("--dry-run", action="store_true", help="print the first cell's summarizer prompt and request body (no key, no network) and the cell counts")
    p.add_argument("--out", default="", help="default runs/<tag>/judged_gemini"); p.add_argument("--seed", type=int, default=0); a = p.parse_args()
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True); log(" ".join(sys.argv))
    run_dir = a.run if os.path.isabs(a.run) else os.path.join(ROOT, a.run); res = json.load(open(os.path.join(run_dir, "results.json"), encoding="utf-8")); fres = res["families"][a.family]
    bank = fres["bank"] if os.path.exists(fres["bank"]) else os.path.join(ROOT, "data", "wsbench", a.family + ".json"); header, items = R.load_bank(bank); contract = R.contract_for(header, header.get("family", a.family))
    x8 = (lambda s_: (lambda mod: (s_.loader.exec_module(mod), mod)[1])(importlib.util.module_from_spec(s_)))(importlib.util.spec_from_file_location("x8", os.path.join(HERE, "x8_wsbench_items.py")))
    scored, gated_ids, im_ids = x8.gate_sets(fres, items)
    if a.items: sel = set(a.items.split(",")); scored = [it for it in scored if it["id"] in sel]
    out_root = a.out or os.path.join(run_dir, "judged_gemini"); client = None
    for arm in [x for x in a.arms.split(",") if x]:
        e = fres["arms"].get(arm)
        if e is None or e.get("kind") != "tokens": log(f"{arm}: not a token arm of this run; skipped"); continue
        f = os.path.join(ROOT, e["file"]) if not os.path.isabs(e["file"]) else e["file"]
        cells = [R.parse_readout_row(json.loads(l)) for l in open(f, encoding="utf-8")]; cells = [c for c in cells if c is not None and c["tokens"] is not None]
        layers = [int(x) for x in a.layers.split(",")] if a.layers else list(e["layers"]); ids = {it["id"] for it in scored}
        cells = [c for c in cells if c["id"] in ids and c["layer"] in layers]
        if a.limit: cells = cells[: a.limit]
        bundles = {cell_key(c["id"], c["layer"], c["pos"]): render_bag(c["tokens"], c["scores"]) for c in cells if c["tokens"]}
        out_dir = os.path.join(out_root, arm, a.family); os.makedirs(out_dir, exist_ok=True); cache_f = os.path.join(out_dir, "cells.jsonl"); cache = load_cache(cache_f)
        if a.dry_run:
            for key, txt in bundles.items():
                print(f"--- summarizer prompt for {key} (model={a.model}, thinking={a.thinking_level}) ---\n[system]\n{INTERP_SYSTEM}\n[user]\n{render_user(txt)}\n[request body]\n{json.dumps(request_body(txt, a.thinking_level), ensure_ascii=False, indent=1)}"); break
            log(f"{arm}: dry run; {len(bundles)} cells ({len(cells)} rows, {len(scored)} items x {len(layers)} layers); cached {sum(1 for k, t in bundles.items() if ('summ:' + k, fingerprint(SUMMARIZER_PROMPT_VERSION, a.model, t)) in cache)}"); continue
        if client is None: client = Gemini(read_keys(), a.model, a.thinking_level, a.rpm)
        summaries = {}; pending = []
        for key, txt in bundles.items():
            fp = fingerprint(SUMMARIZER_PROMPT_VERSION, a.model, txt); row = cache.get(("summ:" + key, fp))
            ok_ = row is not None and isinstance(row.get("result"), dict) and isinstance(row["result"].get("interpretation"), str) and row["result"]["interpretation"].strip()
            if ok_: summaries[key] = row["result"]["interpretation"]
            else: pending.append((key, fp, txt))  # never tried, or a failed (null) summary: retry
        log(f"{arm} {a.family}: summarizer {len(pending)} calls ({len(bundles) - len(pending)} cached)")
        with open(cache_f, "a", encoding="utf-8") as fh:
            # one summary per call (the bench's protocol, no batching of cells into one prompt); --workers calls run
            # concurrently. Each worker owns its own client (pacing / mode state), writes go through one lock.
            import threading, concurrent.futures as cf
            lock = threading.Lock(); done_n = [0]; stop = [None]
            workers = max(1, a.workers)
            clients = [client] + [Gemini(read_keys(), a.model, a.thinking_level, a.rpm) for _ in range(workers - 1)]
            tl = threading.local(); free = list(clients)
            def mine():
                if not hasattr(tl, "c"):
                    with lock: tl.c = free.pop()
                return tl.c
            def job(idx, item):
                key, fp, txt = item
                if stop[0]: return
                c_ = mine()
                try: r_ = c_.summarize(txt)
                except QuotaExhausted as qe:
                    stop[0] = str(qe); return
                with lock:
                    summaries[key] = r_["interpretation"] if r_ else None
                    fh.write(json.dumps({"key": "summ:" + key, "fp": fp, "result": r_, "model": a.model, "used": dict(c_.used)}, ensure_ascii=False) + "\n"); fh.flush()
                    done_n[0] += 1
                    if done_n[0] % 20 == 0 or done_n[0] == len(pending): log(f"  {done_n[0]}/{len(pending)} {key}: {(r_ or {}).get('interpretation', None)!r}"[:220])
            with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(lambda t: job(*t), enumerate(pending)))
            for c_ in clients[1:]:
                for k_, v_ in c_.spend.items(): client.spend[k_] = client.spend.get(k_, 0) + v_
            if stop[0]:
                log(f"  STOP: {stop[0]}; {done_n[0]} of {len(pending)} done this run; cache kept at {cache_f}; rerun later to resume"); raise SystemExit(3)
        sc = score(cells, [it for it in scored if any(c["id"] == it["id"] for c in cells)], contract, layers, summaries, a.seed, gated_ids, im_ids, bool(fres.get("no_gate")))
        outj = {"run": os.path.relpath(run_dir, ROOT), "family": a.family, "arm": arm, "readouts": e["file"], "model": a.model, "route": "gemini-direct (generativelanguage.googleapis.com v1beta generateContent)", "config_used": dict(client.used), "max_output_tokens": MAX_TOKENS, "temperature": None,
                "bench_commit": R.BENCH_COMMIT, "layers": layers, "n_cells": len(cells), "n_summaries": len(bundles), "n_summaries_failed": sum(1 for k in bundles if summaries.get(k) is None), "spend": dict(client.spend), **sc}
        json.dump(outj, open(os.path.join(out_dir, "results.json"), "w"), indent=1, ensure_ascii=False)
        d = sc["denominators"]; fmt = lambda v: "-" if v is None else f"{v:.3f}"
        log(f"{arm} {a.family}: SUMMARIZED pass {fmt(sc['pass_rate'])} (n {sc['n_items_decided']}; raw {fmt(sc['raw']['pass_rate'])}); by source {sc['passing_layers_by_source']}; all {fmt(d['all']['pass'])} gated {fmt(d['gated']['pass'])} (n {d['gated']['n']}) immediate {fmt(d['gated_immediate']['pass'])} (n {d['gated_immediate']['n']}); failed summaries {outj['n_summaries_failed']}; {client.spend} -> {out_dir}/results.json")


if __name__ == "__main__":
    main()
