"""Vendored port of WorkspaceBench's multi-token regex contract and the small bench helpers our P6 adapter needs.

ATTRIBUTION. The matcher, contract, unit builder and item verdict below are a line-for-line port of
``src/wsbench/multitoken/regex.py`` of https://github.com/camilablank/workspace-bench (MIT licence, commit
92d763e722377f5bfae045e829a308ab5919a820, 2026-09-23; the bench itself ports the source repo's
``olens_suite/bank/{matching,contract,conjunctive}.py`` and ``olens_sglang/{score_targets,common}.py``). The only
edits are syntactic: Python 3.9-compatible annotations (the Mac test interpreter is 3.9; the bench declares 3.12),
so ``X | None`` became ``Optional[X]`` and ``list[str]`` in runtime-evaluated positions became ``List[str]``. The
semantics are pinned by the bench's own goldens, vendored under tests/wsbench_golden/ and checked by
tests/test_p2p3_tiny.py (44 matcher pairs, hit_forms, the grid verdict, and scored_units for every item of the six
banks). Also ported here (same commit): ``label_of`` / ``load_bank`` (banks.py), ``display_tokens`` (produce/render.py),
the read-site rule of ``readplan._final_token_families``, ``render_bag`` (summarizer.py) and the cosine J-lens
readout of ``produce/methods.py::JLens.read``. When the real package is importable (the HPC checkout on PYTHONPATH),
scripts/x8_wsbench_items.py uses the bench's modules and this file is only the fallback / the Mac test path.

The scorer of record for the six multi-token families (SCORER_VERSION) applies the contract to a cell's samples. For a
TOKEN lens the bench first turns the bag into prose with an LLM summarizer (Gemini via OpenRouter; docs/summarizer.md)
and scores the interpretation as one more sample (SUMMARIZED_SCORER_VERSION). That step needs an API key, so this file
scores token bags RAW only (each token one sample): the number is the bench's ``regex-raw`` column, not its headline
for token arms. The readouts files x8 writes are in the bench's contract, so ``wsbench judge`` can add the summarized
number later.
"""
from __future__ import annotations
import json, re, unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

SCORER_VERSION = "mt-regex-2026-09-23"
SUMMARIZED_SCORER_VERSION = "mt-regex-summarized-2026-09-23"
BENCH_COMMIT = "92d763e722377f5bfae045e829a308ab5919a820"
MT_FAMILIES = ["basic_readout_mt", "multihop_mt", "multilingual_mt", "typo_mt", "multilingual_typo", "multilingual_multihop"]
TOP_K = 10  # the bench's token-lens top-k (produce/methods.py TOP_K)

# ---------------------------------------------------------------------------------------------
# olens_sglang/common.py - scaffolding stripped from every sample before scoring
# ---------------------------------------------------------------------------------------------
_STRIP_PATTERNS = re.compile(r"</?explanation>|<\|im_start\|>(system|user|assistant)?|<\|im_end\|>|<think>|</think>")


def extract_phrase(text: str) -> str:
    """Strip chat scaffolding + ``<explanation>`` tags from one lens sample."""
    return _STRIP_PATTERNS.sub("", text).strip()


# ---------------------------------------------------------------------------------------------
# global_workspace/glossary.py - the two numeric helpers the matcher needs
# ---------------------------------------------------------------------------------------------
FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９．", "0123456789.")
CJK_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6}
CJK_DIGIT.update({"七": 7, "八": 8, "九": 9})
CJK_UNIT = {"十": 10, "百": 100, "千": 1000}


def parse_cjk_numeral(s: str) -> Optional[int]:
    """Standard CJK numerals up to 9999 (十五=15, 四十九=49, 三百=300, 两=2), else ``None``."""
    if not s or any(c not in CJK_DIGIT and c not in CJK_UNIT for c in s):
        return None
    total, current = 0, 0
    for c in s:
        if c in CJK_DIGIT:
            current = CJK_DIGIT[c]
        else:
            total += (current or 1) * CJK_UNIT[c]
            current = 0
    return total + current


# ---------------------------------------------------------------------------------------------
# olens_suite/bank/matching.py - hit semantics for conjunctive units
# ---------------------------------------------------------------------------------------------
_WRAP = r"[\s$*{(\\`：（＊]*"  # markdown/mathjax/fullwidth wrapping before a number
_ANSWER_CONTEXT = (
    r"(?:=|->|→|\bequals\b|\b(?:answer|result|value|total|sum|product|quotient)"
    r"\b(?:\s+is)?\s*:?"
    r"|(?:答案|结果|总和|总数)(?:是|为|：|:)?"
    r"|(?:值|积|商|差|和)(?:是|为|：|:)"
    r"|等于|得出|即|共计|共)"
)
_HAN = r"[一-鿿]"


def _number_pattern(t: str) -> str:
    """The digits of a purely numeric target with full-number identity: 84 not in 84.5 / 184."""
    return re.escape(t) + r"(?![\d.,]?\d)(?!\.\d)"


def _numeric_matcher(target: str) -> Callable[[str], bool]:
    """The source ``word_matcher`` numeric branch: the number in ANSWER position - after an answer marker (English or
    Chinese), or alone (possibly wrapped) at the start of the sample and not opening an expression - or a CJK numeral
    there that parses to the value. Fullwidth digits are folded first. The sample is NOT lowercased on this path (as
    in the source's conjunctive scorer), so the English markers match in lowercase only."""
    t = target.lower()
    num = _number_pattern(t)
    after_eq = re.compile(_ANSWER_CONTEXT + _WRAP + num)
    at_start = re.compile(r"\A[\s$*#>\-]*" + num + r"(?!\s*[+\-*/×÷^=]\s*\d)")
    cjk_numeral = re.compile(r"(?:" + _ANSWER_CONTEXT + _WRAP + r"|\A[\s$*#>\-]*)" r"([零一二两三四五六七八九十百千]{1,6})(?!" + _HAN + ")")
    value = int(t)

    def numeric(text: str) -> bool:
        folded = text.translate(FULLWIDTH_DIGITS)
        if after_eq.search(folded) or at_start.search(folded):
            return True
        return any(parse_cjk_numeral(m) == value for m in cjk_numeral.findall(folded))

    return numeric


def fold(s: str) -> str:
    """NFKD, strip combining marks, straighten the right single quote, casefold - applied to BOTH the form and the
    sample (Mexico with/without the accent, pointed and unpointed Hebrew, curly and straight apostrophes coincide)."""
    s = unicodedata.normalize("NFKD", s).replace("’", "'")
    return "".join(c for c in s if not unicodedata.combining(c)).casefold()


_CJK_HANGUL = re.compile(r"[぀-ヿ㐀-鿿가-힯]")
_WORDY = re.compile(r"[^\W_]+(?:[\s\-'][^\W_]+)*")


def unicode_word_matcher(form: str) -> Callable[[str], bool]:
    """Boundary match on folded text for a unit form (any script); CJK/Hangul and non-wordy forms are substrings of
    the folded sample; purely numeric forms use the answer-context rule. The script decision is made on the RAW form
    (NFKD decomposes Hangul syllables)."""
    f = fold(form)
    if re.fullmatch(r"\d+", f):
        return _numeric_matcher(f)
    if _CJK_HANGUL.search(form) or not _WORDY.fullmatch(f):
        return lambda text: f in fold(text)
    parts = [re.escape(w) for w in re.split(r"[\s\-]+", f) if w]
    pat = re.compile(r"(?<![^\W_])" + r"[\s\-]+".join(parts) + r"(?![^\W_])")
    return lambda text: bool(pat.search(fold(text)))


def _standalone_number_matcher(form: str) -> Callable[[str], bool]:
    """``numeric_match: "standalone"``: the integer anywhere as a standalone number. None of the six banks sets it;
    ported so a bank that does scores as the source would."""
    pat = re.compile(r"(?<![\w.,-])" + re.escape(form.strip()) + r"(?![\w.,]?\d)(?!\w)")
    return lambda text: bool(pat.search(fold(text).replace(",", "")))


def hit_forms(samples: List[str], forms: Mapping[str, List[str]], numeric_match: str = "context") -> List[str]:
    """The language keys of ``forms`` whose strings hit any SINGLE sample, in dict order."""

    def matcher(f: str) -> Callable[[str], bool]:
        if numeric_match == "standalone" and re.fullmatch(r"-?\d+", f.strip()):
            return _standalone_number_matcher(f)
        return unicode_word_matcher(f)

    out: List[str] = []
    for lang, fs in forms.items():
        matchers = [matcher(f) for f in fs]
        if matchers and any(m(s) for m in matchers for s in samples):
            out.append(lang)
    return out


# ---------------------------------------------------------------------------------------------
# olens_suite/bank/contract.py - the bank contract and the per-unit form builder
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BankContract:
    multi_token: bool  # only multi-token forms may award credit
    include_target: bool  # the item's ``target`` is itself a scored (optional) unit
    conjunctive_units: bool  # a layer passes only when EVERY required unit hits there


def contract_for(header: Mapping[str, Any], family: str) -> BankContract:
    """The header's ``contract`` block, else the source's legacy name-suffix rules."""
    block = header.get("contract")
    if isinstance(block, Mapping):
        return BankContract(multi_token=bool(block.get("multi_token", family.endswith("-mt"))), include_target=bool(block.get("include_target", True)),
                            conjunctive_units=bool(block.get("conjunctive_units", False)))
    return BankContract(multi_token=family.endswith("-mt"), include_target=family != "multilingual-mt", conjunctive_units=False)


@dataclass(frozen=True)
class ScoredUnit:
    role: str
    required: bool
    forms: Dict[str, List[str]]  # language -> the strings that may award credit
    numeric_match: str = "context"

    def all_forms(self) -> List[str]:
        out: List[str] = []
        for fs in self.forms.values():
            for f in fs:
                if f not in out:
                    out.append(f)
        return out

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"role": self.role, "required": self.required, "forms": dict(self.forms)}
        if self.numeric_match != "context":
            out["numeric_match"] = self.numeric_match
        return out


def _numeric_match_of(raw: Mapping[str, Any]) -> str:
    mode = str(raw.get("numeric_match", "context"))
    if mode not in ("context", "standalone"):
        raise ValueError(f"unit {raw.get('role')!r}: unknown numeric_match {mode!r}")
    return "standalone" if mode == "standalone" else "context"


def _unit_forms(raw: Mapping[str, Any]) -> Dict[str, List[str]]:
    forms = raw.get("forms")
    if isinstance(forms, Mapping) and forms:
        return {str(k): [str(x) for x in v] for k, v in forms.items()}
    return {"": [str(m) for m in raw.get("match", [])]}


def _counts(item: Mapping[str, Any], role: str, lang: str, n: int) -> List[int]:
    """``probe_token_lens.units[role][lang]`` - the bank's Qwen3.6-27B token count per form, positionally aligned with
    ``forms[lang]``. A missing stamp is a bank error, never a guess."""
    try:
        counts = [int(x) for x in item["probe_token_lens"]["units"][role][lang]]
    except (KeyError, TypeError) as e:
        raise ValueError(f"item {item.get('name')!r}: no probe_token_lens count for unit {role!r} {lang!r}") from e
    if len(counts) != n:
        raise ValueError(f"item {item.get('name')!r}: probe_token_lens.units[{role!r}][{lang!r}] has {len(counts)} counts for {n} forms")
    return counts


def scored_units(item: Mapping[str, Any], contract: BankContract) -> List[ScoredUnit]:
    """Units of one item under ``contract`` - the source's ``scored_units`` with the token counts read from the bank's
    ``probe_token_lens`` stamp instead of a live tokenizer. No alias pruning: forms are an authored OR-list."""
    units: List[ScoredUnit] = []
    for raw in item.get("units") or []:
        role = str(raw.get("role", "unit"))
        forms = _unit_forms(raw)
        if contract.multi_token and bool(raw.get("multi_token", True)):
            kept: Dict[str, List[str]] = {}
            for lang, fs in forms.items():
                counts = _counts(item, role, lang, len(fs))
                fs2 = [f for f, n in zip(fs, counts) if n > 1]
                if fs2:
                    kept[lang] = fs2
            forms = kept
        required = bool(raw.get("required", True))
        if required and not forms:
            raise ValueError(f"item {item.get('name')!r}: required unit {role!r} has no creditable (multi-token) form")
        if forms:
            units.append(ScoredUnit(role, required, forms, _numeric_match_of(raw)))
    roles = {u.role for u in units}
    target = item.get("target")
    if contract.include_target and target and "target" not in roles:
        alts = [str(a) for a in item.get("target_alts", [])]
        tforms = [str(target)] + alts
        if contract.multi_token:
            ptl = item.get("probe_token_lens") or {}
            try:
                counts = [int(ptl["target"])] + [int(x) for x in ptl["target_alts"]]
            except (KeyError, TypeError) as e:
                raise ValueError(f"item {item.get('name')!r}: include_target needs probe_token_lens.target and .target_alts") from e
            if len(counts) != len(tforms):
                raise ValueError(f"item {item.get('name')!r}: probe_token_lens.target_alts has {len(counts) - 1} counts for {len(alts)} target_alts")
            tforms = [f for f, n in zip(tforms, counts) if n > 1]
        covered = {f.lower() for u in units for f in u.all_forms()}
        tforms = [f for f in tforms if f.lower() not in covered]
        if tforms:
            units.append(ScoredUnit("target", False, {"en": tforms}))
    return units


# ---------------------------------------------------------------------------------------------
# olens_suite/bank/conjunctive.py - per-layer unit hits and the item verdict
# ---------------------------------------------------------------------------------------------
def layer_unit_hits(samples: List[str], units: List[ScoredUnit]) -> Dict[str, List[str]]:
    """role -> the languages in which the unit hit any single sample at this layer."""
    return {u.role: hit_forms(samples, u.forms, numeric_match=u.numeric_match) for u in units}


def layer_passes(hits: Mapping[str, List[str]], units: List[ScoredUnit]) -> bool:
    return all(bool(hits.get(u.role)) for u in units if u.required)


def item_result(by_layer: Mapping[int, Mapping[str, List[str]]], units: List[ScoredUnit]) -> Dict[str, Any]:
    """Fold per-layer unit hits into the item row: pass = some layer where every required unit hits; ``unit_langs``
    lists the languages in the unit's authored order (L2 -> en -> zh)."""
    passing = sorted(layer for layer, hits in by_layer.items() if layer_passes(hits, units))
    unit_hit = {u.role: any(bool(h.get(u.role)) for h in by_layer.values()) for u in units}
    unit_langs = {u.role: [lang for lang in u.forms if any(lang in h.get(u.role, []) for h in by_layer.values())] for u in units}
    return {"pass": bool(passing), "earliest_layer": passing[0] if passing else None, "n_passing_layers": len(passing), "passing_layers": passing,
            "unit_hit": unit_hit, "unit_langs": unit_langs, "first_lang": {role: (langs[0] if langs else None) for role, langs in unit_langs.items()},
            "any_hit": any(unit_hit.values())}


# ---------------------------------------------------------------------------------------------
# banks.py / readplan.py / produce/render.py / summarizer.py / produce/methods.py helpers (same commit)
# ---------------------------------------------------------------------------------------------
_LABEL_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def label_of(name: str) -> str:
    """The readout id of a bank item: its name with every character outside ``[A-Za-z0-9._-]`` replaced by ``_``."""
    return _LABEL_UNSAFE.sub("_", name)


def load_bank(path) -> "tuple[Dict[str, Any], List[Dict[str, Any]]]":
    """``(header, items)`` of a bank file; each item gets ``id = label_of(name)``; two names sharing a label is an error."""
    d = json.load(open(path, encoding="utf-8"))
    items = d["items"]
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        raise ValueError(f"{path}: 'items' must be a list of objects")
    seen: Dict[str, str] = {}
    for it in items:
        label = label_of(it["name"])
        if label in seen:
            raise ValueError(f"{path}: items {seen[label]!r} and {it['name']!r} share label {label!r}")
        seen[label] = it["name"]
        it["id"] = label
    header = {k: v for k, v in d.items() if k != "items"}
    return header, items


def read_offset(item: Mapping[str, Any]) -> int:
    """readplan._final_token_families: the read position counted from the end (1 = the final token). The six mt banks
    all carry ``readout.offsets == [-1]``; a bank with ``[-k]`` reads the token k from the end."""
    spec = item.get("readout") or {}
    if spec.get("kind") in ("final_prompt_token", "last_word_token") and spec.get("offsets"):
        return -int(spec["offsets"][0])
    return 1


def display_tokens(tokenizer, ids: List[int]) -> List[str]:
    """Token strings as the bench's readouts spell them: byte-level BPE with ``Ġ``/``▁`` shown as the space they
    encode (``" the"``), what the regex scorer matches for a token lens."""
    return [t.replace("Ġ", " ").replace("▁", " ") if isinstance(t, str) else "" for t in tokenizer.convert_ids_to_tokens(ids)]


def render_bag(tokens, scores=None) -> str:
    """summarizer.render_bag: ``tok (score)`` best first, 2 dp, joined by ' | ' (the text the bench's summarizer sees)."""
    if scores is None:
        return " | ".join(tokens)
    return " | ".join(f"{t} ({s:.2f})" for t, s in zip(tokens, scores))


def jlens_cosine_scores(D, h):
    """produce/methods.py::JLens.read, the bench's J-lens readout of record: ``(W_U J h) / ||J^T W_U[t]||`` per token,
    i.e. the rows of D = W_U J normalised (the dot product D h divided by the row norms). D [V, d] float, h [d]."""
    Dn = D.norm(dim=1).clamp_min(1e-9)
    return (D @ h.to(D.dtype)) / Dn


def parse_readout_row(row) -> Optional[Dict[str, Any]]:
    """readouts._parse_row: a validated cell dict, or None if the row is malformed (used to check the files x8 writes)."""
    if not isinstance(row, dict): return None
    id_, layer, pos = row.get("id"), row.get("layer"), row.get("pos")
    if not isinstance(id_, str) or isinstance(layer, bool) or not isinstance(layer, int) or isinstance(pos, bool) or not isinstance(pos, int): return None
    has_samples, has_tokens = "samples" in row, "tokens" in row
    if has_samples == has_tokens: return None
    token = row.get("token")
    if token is not None and not isinstance(token, str): return None
    if has_samples:
        if "scores" in row or not (isinstance(row["samples"], list) and all(isinstance(s, str) for s in row["samples"])): return None
        return {"id": id_, "layer": layer, "pos": pos, "samples": list(row["samples"]), "tokens": None, "scores": None, "token": token}
    tokens = row["tokens"]
    if not (isinstance(tokens, list) and all(isinstance(s, str) for s in tokens)): return None
    scores = None
    if "scores" in row:
        sc = row["scores"]
        if not isinstance(sc, list) or len(sc) != len(tokens) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in sc): return None
        scores = [float(v) for v in sc]
    return {"id": id_, "layer": layer, "pos": pos, "samples": None, "tokens": list(tokens), "scores": scores, "token": token}


def cell_samples(cell: Mapping[str, Any]) -> List[str]:
    """family._cell_samples WITHOUT the summarizer sample: one sample per prose sample, or one per top-k token string;
    scaffolding stripped; blanks dropped. Samples are never joined."""
    raw = cell["tokens"] if cell.get("tokens") is not None else (cell.get("samples") or [])
    out = [extract_phrase(str(s)) for s in raw]
    return [s for s in out if s]


def primary_units(units: List[ScoredUnit]) -> List[ScoredUnit]:
    """A20's primary-string-only column: every unit keeps only the FIRST creditable form of each language (the
    bank's authored order puts the item's `target` / `intermediates[k]` first, the aliases after it). Not the bench's
    rule (the bench counts every alias); a stricter report-only column printed beside it."""
    return [ScoredUnit(u.role, u.required, {lang: fs[:1] for lang, fs in u.forms.items() if fs}, u.numeric_match) for u in units]


def score_cells(cells: List[Mapping[str, Any]], items: List[Mapping[str, Any]], contract: BankContract, layers: List[int], units_of: Optional[Mapping[str, List[ScoredUnit]]] = None) -> Dict[str, Any]:
    """The raw-regex verdict of family.run_regex over a list of cell dicts (parse_readout_row output) for the given
    items and layers: per item pass / earliest layer / unit hits / any_hit, and the family rates. An item with a
    missing (item, layer) cell and no pass is undecided (None), as in the bench; an empty cell is a negative.
    units_of (optional): item id -> the units to score instead of scored_units(item, contract) (A20 primary-only)."""
    units_of = {it["id"]: scored_units(it, contract) for it in items} if units_of is None else {it["id"]: units_of[it["id"]] for it in items}
    by = {}
    for c in cells:
        by.setdefault((c["id"], c["layer"]), []).append(c)
    rows = []
    for it in items:
        hits_by, missing, empty = {}, [], []
        for l in layers:
            cs = by.get((it["id"], l))
            if not cs:
                missing.append(l); continue
            if len({c["pos"] for c in cs}) > 1:
                raise ValueError(f"{it['id']} L{l}: more than one read position (the mt families read the final prompt token only)")
            samples = cell_samples(cs[0])
            if not samples: empty.append(l)
            hits_by[l] = layer_unit_hits(samples, units_of[it["id"]])
        res = item_result(hits_by, units_of[it["id"]])
        passed = res["pass"]; incomplete = bool(missing) or not hits_by
        rows.append({"id": it["id"], "roles": [u.role for u in units_of[it["id"]] if u.required], "optional_roles": [u.role for u in units_of[it["id"]] if not u.required],
                     "pass": True if passed else (None if incomplete else False), "earliest_layer": res["earliest_layer"], "passing_layers": res["passing_layers"],
                     "unit_hit": res["unit_hit"], "unit_langs": res["unit_langs"], "first_lang": res["first_lang"], "any_hit": res["any_hit"],
                     "missing_layers": missing, "empty_layers": empty, "layers": {str(l): h for l, h in sorted(hits_by.items())}})
    decided = [r for r in rows if r["pass"] is not None]
    roles = sorted({r for row in rows for r in (row["roles"] + row["optional_roles"])})
    rate = lambda xs: (sum(xs) / len(xs)) if xs else None
    per_layer = {str(l): rate([float(l in r["passing_layers"]) for r in decided]) for l in layers}
    return {"scorer_version": SCORER_VERSION, "token_bags": "raw only (no summarizer); the bench's token-arm headline is " + SUMMARIZED_SCORER_VERSION,
            "n_items": len(rows), "n_items_decided": len(decided), "pass_rate": rate([float(bool(r["pass"])) for r in decided]),
            "any_hit_rate": rate([float(r["any_hit"]) for r in decided]), "pass_rate_by_layer": per_layer,
            "unit_any_layer": {r: rate([float(row["unit_hit"][r]) for row in decided if r in row["unit_hit"]]) for r in roles}, "rows": rows}
