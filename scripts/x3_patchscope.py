"""X3 (R3 verbalisation, R5 latent ceiling): does the model's own decoder spell the string from h alone?

Patchscopes-style (Ghandeharioun et al. 2024, arXiv 2401.06102), prompt-blind: h is the residual at a source position
of the item's prompt at layer l; it is patched into a FIXED carrier prompt at the carrier's patch position, at a
target layer (default: the same layer), and the model greedy-decodes --gen tokens. No item text ever enters the
carrier, and no gradient / lens is involved - this is the non-linear, finite-scale counterpart of X1.

Carriers (--carriers identity,describe,identity2,describe2; --n-carriers variants per kind, FIXED lists in this
file, never tuned on items). Day-1 kinds, unchanged:
  identity  "cat -> cat; 1135 -> 1135; hello -> hello; ?"     patch at the final "?" token
  describe  "Syria: Country in the Middle East. Leonardo DiCaprio: American actor. Samsung: South Korean
             multinational corporation. x:"                    patch at the "x" token
Wave-2 kinds (ANALYSIS_DAY1 N2, C4 / C5; prereg amendment A5):
  identity2 "Kuala Lumpur -> Kuala Lumpur; hippopotamus -> hippopotamus; 1945 -> 1945; ?"  patch at the final "?":
             MULTI-token exemplars (a two-word name, a 3-4-piece word, a 4-digit year), so the format no longer
             pushes " ->" right after the first emitted piece (C5).
  describe2 "Syria -> Syria is a country in the Middle East. Leonardo DiCaprio -> Leonardo DiCaprio is an American
             actor. Samsung -> Samsung is a South Korean electronics company. ? ->"   patch at the "?" ENTITY SLOT,
             which is NOT the last token: the " ->" after it (and every generated token) reads the patched residual
             through attention, so the name must be copied out of h rather than transplanted as the next token
             (C4 / C8). The natural output is "<name> is <description>"; the exact-string match is on the cut before
             " is ". Checked on CPU with the real 1.7B: with a literal entity in the slot the un-patched carrier
             produces "<entity> is <sensible description>" (LOG, agent G).
Amendment A12 (agent N0; FULL_REPORT 4.11 (b), bug 18):
  identity3 "Grand Rapids -> Grand Rapids; kaleidoscope -> kaleidoscope; 3817 -> 3817; ?"  the identity2 format with
             exemplars CHECKED against every item string, target and pieces-join of data/{h2a,phrase,single}_items.json
             and every target / alias / unit form / match / bridge answer of the six data/wsbench/*.json banks: no
             exemplar (case-folded, whitespace-collapsed) equals or is contained in any of them (carrier_overlap();
             asserted at import time and in main()). identity2 is kept UNCHANGED for the reproducibility of the wave-2
             runs: 20 of its 24 exemplars are item strings (see the comment above IDENTITY2), so its results are
             reported with the overlapping items excluded as a sensitivity analysis (A12).
Wave 4 (prereg A15 / A17; ANALYSIS_WAVE3 section 6 E1 / E3): three more FIXED kinds, all checked item-free at import time
  identity3_long  identity3 format with 3-4-word exemplars ("the Willow Creek Library -> the Willow Creek Library; ...")
  continuation    8 neutral prefixes (generic sentences ending mid-clause, answer frames) patched at their LAST token and
                  simply continued (no stop cut); continuation_seed = the 2 answer frames x8's lens-seeded x3s arm uses
  spellfix        "recieve -> receive; definately -> definitely; occurence -> occurrence; ?" (the typo families)
Scoring: exact string match after cutting the generation at the carrier's stop marker (STOPS) and normalising (strip,
collapse internal whitespace, casefold - see cut() and norm()), with a majority vote across the carriers of a kind
(an item counts iff > half of them emit the string). Prefix match (the generation starts with the string) and the
first-piece flag are reported as secondary, softer statistics. C2 fix: the first-piece flag is now TOKEN-level (the
first generated token, skipping a leading pure-whitespace token, equals s1's token id) and, since a one-character
first piece (a digit) is matched by whatever the carrier emits that starts with that digit, the summary also reports
vote_first restricted to p1_chars >= 2 (vote_first_p1ge2), which is the honest first-piece number.

Modes: replace (overwrite the carrier residual at the patch position), replace_nm (overwrite with h rescaled to the
mean carrier residual norm at the target layer's patch position; the residual norm grows ~10x from block 4 to block
22, so a deep h replaced raw at an early target saturates the carrier - the CPU smoke gave " The -> The -> The") and,
with --modes add, a norm-matched addition (delta = alpha * mean carrier residual norm / ||h|| * h). Controls: shuffled h within category (fixed derangement,
never the same string; items without a valid partner are dropped and counted), crosscat (h of an item of another
category), norm-matched random direction. --target-layer same,2,4,8 patches h (read at the source layer) after block
l' of the carrier; arms of other targets append ":t<l'>" to the Day-1 arm key.
Positions (--positions last | all | firsthop | last,firsthop): `last` = the last prompt token; `all` scans EVERY
prompt position (bridges; reports the per-position profile plus the best position per item, a per-item selection
labelled as such); `firsthop` = the last token of the first-hop entity of a curated bridge prompt (e12.firsthop_pos:
"... whose capital is Gaborone is" -> the last token of "Gaborone"; items without a template position are skipped
and counted). With several rules the arm key appends ":<rule>". Held-out half for the answer headline (--split).
Wave 5 (prereg A21 / A22 / A24):
  --target-layer all<k>   h is WRITTEN (overwrite) at the patch position at the embedding and after every block 0..k
                          (qwen3_min.forward inject_set), so no representation of the carrier's placeholder survives
                          there at any depth <= k; the blocks above k run clean. Arm suffix ":all<k>". Mode replace only.
  --placeholder <s>       the final "?" of every identity-format carrier is replaced by <s> (must be one final token);
                          arm suffix ":ph=<s>". Together these separate "less placeholder contamination" from "more
                          computation after the patch" (Patchscopes App. C; A21).
  --positions firsthop-1 / firsthop+1   the token before / after the first-hop token (controls at the PARTNER's own
                          first-hop +- k token, the A14 rule); the continuation carrier is a --carriers kind at any position.
  --track-filter A|B      data/fresh_items.json (A24) holds both tracks in one file.
"""
import argparse, importlib.util, json, os, re, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sjlens.model.qwen3_min import forward, logits as true_logits
from sjlens.lens.jlens import valid_mask
from sjlens.eval.common import load_q
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("e12", os.path.join(HERE, "e12_second_token.py")); e12 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e12)
PASS_RECALL, KILL_RECALL, CTRL_MAX = 0.30, 0.05, 0.05  # docs/prereg_R.yaml R3

# Fixed carrier variants. Each is (prompt, marker): the patch goes at the LAST token of the prompt, which is the
# marker token ("?" or "x"). Eight per kind; the list is part of the pre-registration and is never chosen on items.
IDENTITY = [
    "cat -> cat; 1135 -> 1135; hello -> hello; ?",
    "apple -> apple; 42 -> 42; river -> river; ?",
    "book -> book; 7 -> 7; music -> music; ?",
    "table -> table; 2024 -> 2024; window -> window; ?",
    "green -> green; 91 -> 91; father -> father; ?",
    "stone -> stone; 365 -> 365; letter -> letter; ?",
    "paper -> paper; 18 -> 18; garden -> garden; ?",
    "silver -> silver; 500 -> 500; morning -> morning; ?",
]
DESCRIBE = [
    "Syria: Country in the Middle East. Leonardo DiCaprio: American actor. Samsung: South Korean multinational corporation. x:",
    "Norway: Country in northern Europe. Ada Lovelace: English mathematician. Toyota: Japanese car manufacturer. x:",
    "Peru: Country in South America. Miles Davis: American jazz trumpeter. Nokia: Finnish telecommunications company. x:",
    "Egypt: Country in north Africa. Marie Curie: Polish-French physicist. Ikea: Swedish furniture retailer. x:",
    "Vietnam: Country in southeast Asia. Frida Kahlo: Mexican painter. Nestle: Swiss food company. x:",
    "Iceland: Island country in the north Atlantic. Alan Turing: British computer scientist. Airbus: European aircraft manufacturer. x:",
    "Kenya: Country in east Africa. Pablo Picasso: Spanish painter. Bosch: German engineering company. x:",
    "Chile: Country on the west coast of South America. Jane Austen: English novelist. Philips: Dutch electronics company. x:",
]
# Wave-2 (amendment A5). identity2: every exemplar is multi-token (a two-word place name, a 3-4-piece word, a
# 4-digit year). FROZEN for reproducibility (w2A_x3v2_*, w2t_A_x3_14b_*): the original comment here claimed no exemplar
# is an item string; that was FALSE (FULL_REPORT 4.11 (b), bug 18): carrier_overlap("identity2") finds 20 of the 24
# exemplars in the item files (1066, 1492, 1789, 1815, 1912, 1945, 1969, Kuala Lumpur, New Delhi, Schmetterling,
# Sri Lanka, Tchaikovsky, chimpanzee, hippopotamus, molybdenum, rhinoceros in h2a_items; Buenos Aires, Los Angeles,
# Rio de Janeiro, San Francisco in phrase_items; Addis Ababa is a single_items target); only 2024, bibliothèque and
# praseodymium are clean. New X3 / P6 runs use IDENTITY3 (A12); identity2 results are reported with the overlapping
# items excluded as a sensitivity analysis.
IDENTITY2 = [
    "Kuala Lumpur -> Kuala Lumpur; hippopotamus -> hippopotamus; 1945 -> 1945; ?",
    "New Delhi -> New Delhi; chimpanzee -> chimpanzee; 1789 -> 1789; ?",
    "Buenos Aires -> Buenos Aires; Schmetterling -> Schmetterling; 1066 -> 1066; ?",
    "Addis Ababa -> Addis Ababa; molybdenum -> molybdenum; 2024 -> 2024; ?",
    "San Francisco -> San Francisco; rhinoceros -> rhinoceros; 1492 -> 1492; ?",
    "Rio de Janeiro -> Rio de Janeiro; Tchaikovsky -> Tchaikovsky; 1815 -> 1815; ?",
    "Los Angeles -> Los Angeles; bibliothèque -> bibliothèque; 1969 -> 1969; ?",
    "Sri Lanka -> Sri Lanka; praseodymium -> praseodymium; 1912 -> 1912; ?",
]
# describe2: the "?" is the entity slot; " ->" follows it, so the patched residual is read by later tokens.
DESCRIBE2 = [
    "Syria -> Syria is a country in the Middle East. Leonardo DiCaprio -> Leonardo DiCaprio is an American actor. Samsung -> Samsung is a South Korean electronics company. ? ->",
    "Norway -> Norway is a country in northern Europe. Ada Lovelace -> Ada Lovelace is an English mathematician. Toyota -> Toyota is a Japanese car manufacturer. ? ->",
    "Peru -> Peru is a country in South America. Miles Davis -> Miles Davis is an American jazz trumpeter. Nokia -> Nokia is a Finnish telecommunications company. ? ->",
    "Egypt -> Egypt is a country in north Africa. Marie Curie -> Marie Curie is a Polish-French physicist. Ikea -> Ikea is a Swedish furniture retailer. ? ->",
    "Vietnam -> Vietnam is a country in southeast Asia. Frida Kahlo -> Frida Kahlo is a Mexican painter. Nestle -> Nestle is a Swiss food company. ? ->",
    "Iceland -> Iceland is an island country in the north Atlantic. Alan Turing -> Alan Turing is a British computer scientist. Airbus -> Airbus is a European aircraft manufacturer. ? ->",
    "Kenya -> Kenya is a country in east Africa. Pablo Picasso -> Pablo Picasso is a Spanish painter. Bosch -> Bosch is a German engineering company. ? ->",
    "Chile -> Chile is a country on the west coast of South America. Jane Austen -> Jane Austen is an English novelist. Philips -> Philips is a Dutch electronics company. ? ->",
]
# Amendment A12 (agent N0). identity3: the identity2 format (a two-word place name, a 3-4-piece word, a 4-digit number)
# with exemplars chosen so that NONE equals or is contained in any item string / target / pieces-join of the three item
# files or any target / alias / unit form / match / bridge answer of the six WorkspaceBench banks (carrier_overlap
# below; asserted at import time). Piece counts with the Qwen3 tokenizer (leading space): places 2-4, words 3-4,
# numbers 4 (one token per digit) + the space. The numbers are deliberately NOT years (the YEARS items) and not round.
IDENTITY3 = [
    "Grand Rapids -> Grand Rapids; kaleidoscope -> kaleidoscope; 3817 -> 3817; ?",
    "Chiang Mai -> Chiang Mai; serendipity -> serendipity; 5926 -> 5926; ?",
    "Alice Springs -> Alice Springs; ventriloquist -> ventriloquist; 7403 -> 7403; ?",
    "Milton Keynes -> Milton Keynes; gobbledygook -> gobbledygook; 6158 -> 6158; ?",
    "Punta Arenas -> Punta Arenas; flabbergasted -> flabbergasted; 4271 -> 4271; ?",
    "Ann Arbor -> Ann Arbor; pomegranate -> pomegranate; 8539 -> 8539; ?",
    "Boca Raton -> Boca Raton; Schadenfreude -> Schadenfreude; 2764 -> 2764; ?",
    "Coffs Harbour -> Coffs Harbour; paraphernalia -> paraphernalia; 9385 -> 9385; ?",
]
# Wave 4 (prereg A15 / A17; ANALYSIS_WAVE3 section 6 E1 / E3). Three new FIXED carrier kinds, never tuned on items;
# every one is checked item-free at import time like identity3 (exemplars vs item / bench strings in BOTH directions,
# and every item / bench string of >= 3 characters as a whole word inside the carrier text; carrier_overlap below).
# identity3_long: the identity3 format with 3-4-WORD exemplars, so that the format no longer closes a multi-word
#   entity after its first word (ANALYSIS_WAVE3 4.8: ' sodium ->' on 'sodium chloride'). Patch at "?", identity stops.
IDENTITY3_LONG = [
    "the Willow Creek Library -> the Willow Creek Library; enamel baking dish -> enamel baking dish; Flight 8462 to Perth -> Flight 8462 to Perth; ?",
    "the Royal Botanic Gardens -> the Royal Botanic Gardens; hand-painted ceramic bowl -> hand-painted ceramic bowl; Route 3817 northbound -> Route 3817 northbound; ?",
    "the Blue Heron Inn -> the Blue Heron Inn; second-hand bicycle shop -> second-hand bicycle shop; Room 5926 top floor -> Room 5926 top floor; ?",
    "the Sandy Cove Lodge -> the Sandy Cove Lodge; freshly ground coffee beans -> freshly ground coffee beans; Platform 7403 at noon -> Platform 7403 at noon; ?",
    "the Riverside Market Hall -> the Riverside Market Hall; wooden garden bench -> wooden garden bench; Gate 6158 for boarding -> Gate 6158 for boarding; ?",
    "the Hilltop Observatory dome -> the Hilltop Observatory dome; hand-knitted woollen scarf -> hand-knitted woollen scarf; Locker 4271 downstairs -> Locker 4271 downstairs; ?",
    "the Forbidden City walls -> the Forbidden City walls; vintage leather armchair -> vintage leather armchair; Bus 8539 harbour route -> Bus 8539 harbour route; ?",
    "the Harbour Lights Hotel -> the Harbour Lights Hotel; rusty garden rake -> rusty garden rake; Suite 2764 upstairs -> Suite 2764 upstairs; ?",
]
# continuation (x3c): 8 neutral prefixes (4 generic sentences ending mid-clause + 4 answer frames); h REPLACES the
#   residual at the carrier's LAST token (" the" / " is" / ":" - the same kind of prediction site as the bench's read
#   token) after block t, and the model simply continues (greedy, no stop cut: the raw generation is one prose sample).
CONTINUATION = [
    "The committee met on Tuesday to review the proposal, and after a short discussion the chair announced that the",
    "According to the report published last spring, the most important factor turned out to be the",
    "She opened the letter slowly and read the first line, which mentioned the",
    "In the second chapter the author turns to the question of the",
    "Question: what is it called?\nAnswer: the",
    "The answer is",
    "Q: What is its name?\nA:",
    "The correct term for it is",
]
CONTINUATION_SEED = ["The answer is", "Q: What is its name?\nA:"]  # x3s: the 2 answer frames a lens-seeded first piece is appended to
# spellfix (E3): "misspelling -> correct spelling" exemplars, patch at "?", identity stops. A carrier whose format asks
#   for the CORRECTION of the string under construction (the bench's typo families read the misspelt word's last piece).
SPELLFIX = [
    "recieve -> receive; definately -> definitely; occurence -> occurrence; ?",
    "seperate -> separate; accomodate -> accommodate; wierd -> weird; ?",
    "calender -> calendar; goverment -> government; tommorow -> tomorrow; ?",
    "neccessary -> necessary; embarass -> embarrass; untill -> until; ?",
    "arguement -> argument; achive -> achieve; publically -> publicly; ?",
    "occassion -> occasion; maintainance -> maintenance; priviledge -> privilege; ?",
    "recomend -> recommend; truely -> truly; existance -> existence; ?",
    "harrass -> harass; liason -> liaison; milennium -> millennium; ?",
]
CARRIERS = {"identity": IDENTITY, "describe": DESCRIBE, "identity2": IDENTITY2, "describe2": DESCRIBE2, "identity3": IDENTITY3,
            "identity3_long": IDENTITY3_LONG, "continuation": CONTINUATION, "continuation_seed": CONTINUATION_SEED, "spellfix": SPELLFIX}
IDENTITY_KINDS = ("identity", "identity2", "identity3", "identity3_long", "spellfix")  # patch at the final "?"; stops ";", "->", newline
CONTINUATION_KINDS = ("continuation", "continuation_seed")  # patch at the final token; no exemplars, no stop cut
WAVE4_KINDS = ("identity3_long", "continuation", "continuation_seed", "spellfix")  # asserted item-free at import time (A15 / A17)
DATA_DIR = os.path.join(os.path.dirname(HERE), "data")
ITEM_FILES = ("h2a_items.json", "phrase_items.json", "single_items.json")


def exemplars_of(prompt):
    """the exemplar strings of an identity-format carrier: "A -> A; B -> B; C -> C; ?" -> ["A", "B", "C"]."""
    return [seg.split("->")[0].strip() for seg in prompt.split(";")[:-1] if "->" in seg]


def _flat(x):
    if x is None: return []
    if isinstance(x, (list, tuple)): return [y for z in x for y in _flat(z)]
    return [str(x)]


def item_strings(root=DATA_DIR):
    """every answer-side string a carrier exemplar must not touch, normalised with norm(): (string, source) pairs from
    the item files (string, target, pieces-join; targets may be lists) and from the WorkspaceBench banks (target,
    target_alts, intermediates, every unit form and match string, every bridge answer). Prompts are not included:
    an exemplar occurring in a QUESTION is not an answer leak."""
    out = set()
    for f in ITEM_FILES:
        p = os.path.join(root, f)
        if not os.path.exists(p): continue
        for it in json.load(open(p, encoding="utf-8")):
            for s in _flat(it.get("string")) + _flat(it.get("target")): out.add((norm(s), f))
            out.add((norm("".join(it.get("pieces_text", []))), f))
    bench = os.path.join(root, "wsbench")
    for f in sorted(os.listdir(bench)) if os.path.isdir(bench) else []:
        if not f.endswith(".json"): continue
        for it in json.load(open(os.path.join(bench, f), encoding="utf-8")).get("items", []):
            for s in _flat(it.get("target")) + _flat(it.get("target_alts")) + _flat(it.get("intermediates")) + _flat(it.get("typo_word")): out.add((norm(s), "wsbench/" + f))  # typo_word: A17 (spellfix)
            for u in it.get("units") or []:
                for fs in (u.get("forms") or {}).values(): out.update((norm(s), "wsbench/" + f) for s in _flat(fs))
                out.update((norm(s), "wsbench/" + f) for s in _flat(u.get("match")))
            for b in it.get("bridges") or []: out.update((norm(s), "wsbench/" + f) for s in _flat(b.get("answers")))
    return {x for x in out if x[0]}


def exemplar_sides_of(prompt):
    """both sides of every "A -> B;" exemplar (identity: A == B; spellfix: misspelling and correction)."""
    out = []
    for seg in prompt.split(";")[:-1]:
        if "->" in seg: out.extend(x.strip() for x in seg.split("->"))
    return [x for x in out if x]


def _word_in(needle, hay):
    """needle (normalised) as a whole word / phrase inside hay (normalised): no letter or digit on either side."""
    return re.search(r"(?<![^\W_])" + re.escape(needle) + r"(?![^\W_])", hay) is not None


def carrier_overlap(kind, root=DATA_DIR, strings=None):
    """[(exemplar, matched string, source file)] for every exemplar of CARRIERS[kind] that EQUALS or is CONTAINED in
    an item / bench string (both normalised: strip, collapse whitespace, casefold). Empty = clean.
    Wave-4 kinds (WAVE4_KINDS) are checked in BOTH directions and over the whole carrier text: an exemplar (either side
    of "A -> B") equal to / inside an item string, an item string of >= 3 characters as a whole word inside an exemplar,
    and an item string of >= 3 characters as a whole word anywhere in the carrier prompt (continuation kinds have no
    exemplars, so this is their only check). The 3-character floor skips 1-2-letter strings ('x', 'Al')."""
    strings = item_strings(root) if strings is None else strings
    seen, out = set(), []
    strict = kind in WAVE4_KINDS
    for prompt in CARRIERS[kind]:
        exs = exemplar_sides_of(prompt) if strict else exemplars_of(prompt)
        for e in exs:
            en = norm(e)
            if not en or en in seen: continue
            seen.add(en)
            for s, src in sorted(strings):
                if en == s or en in s or (strict and len(s) >= 3 and _word_in(s, en)): out.append((e, s, src))
        if strict:
            pn = norm(prompt)
            for s, src in sorted(strings):
                if len(s) >= 3 and _word_in(s, pn) and not any(s == m for _, m, _ in out): out.append((prompt, s, src))
    return out


def assert_no_overlap(kind="identity3", root=DATA_DIR):
    """raise if any exemplar of `kind` overlaps an item / bench string (A12). Returns the (empty) overlap list."""
    ov = carrier_overlap(kind, root)
    assert not ov, f"carrier kind {kind!r}: {len(ov)} exemplar overlaps with item / bench strings (A12): " + "; ".join(f"{e!r} in {s!r} ({src})" for e, s, src in ov[:12])
    return ov


def patch_pos(tok, cid, kind):
    """the carrier position the patch replaces: the final "?" (identity / identity2 / identity3), the "x" of "x:"
    (describe, usually the second-to-last token) or the "?" of "? ->" (describe2, second-to-last). Falls back to the
    last position if the marker is not its own token."""
    if kind in IDENTITY_KINDS or kind in CONTINUATION_KINDS: return cid.shape[1] - 1
    marker = "x" if kind == "describe" else "?"
    for i in range(cid.shape[1] - 1, max(cid.shape[1] - 6, -1), -1):
        if tok.decode([int(cid[0, i])]).strip().lower() == marker: return i
    return cid.shape[1] - 1


# Where a generation is cut before the exact-string match, per carrier kind. The identity carrier's natural output is
# "<answer> -> <answer>; ..." and the description carrier's is "<description>. ...", so without a stop rule the
# 6-token generation never equals the string and R3 would fail vacuously. Fixed a priori, never tuned on items.
# describe2's natural output is "<name> is <description>." so the cut is at " is " (with both spaces: " island" is
# not a stop), " ->", ";", "." or a newline.
STOPS = {"identity": (";", "->", "\n"), "describe": (".", ";", "\n"), "identity2": (";", "->", "\n"), "describe2": (" is ", " was ", " ->", ";", ".", "\n"), "identity3": (";", "->", "\n"),
         "identity3_long": (";", "->", "\n"), "spellfix": (";", "->", "\n"), "continuation": (), "continuation_seed": ()}  # continuation kinds: no cut (the raw generation is the sample)


def cut(text, kind):
    """the generation up to the first stop marker of this carrier kind (the whole text if none occurs)."""
    ends = [text.find(m) for m in STOPS.get(kind, ()) if text.find(m) >= 0]
    return text[: min(ends)] if ends else text


def norm(s):
    """normalisation for the exact-string match: strip, collapse internal whitespace, casefold."""
    return re.sub(r"\s+", " ", s).strip().casefold()


if os.path.isdir(DATA_DIR):  # import-time guard: identity3 (A12) and the wave-4 kinds (A15 / A17) must stay item-free
    assert_no_overlap("identity3")
    for _k in WAVE4_KINDS: assert_no_overlap(_k)


def is_multi_target(t):
    """an A21 multi-layer target ("all", k): u is WRITTEN at the patch position at the embedding and after every block
    0..k, so no representation of the carrier's placeholder token exists there at any depth <= k."""
    return isinstance(t, tuple) and len(t) == 2 and t[0] == "all"


def patched_generate(q, cid, layer, pos, u, n_gen, mode="replace", alpha=1.0, hn=None, target_layer=None):
    """greedy-decode n_gen tokens from carrier ids [1, T] with u patched at position `pos` of layer `target_layer`
    (replace: overwrite that residual; replace_nm: overwrite with u rescaled to hn, the mean carrier residual norm at
    the target layer's patch position - needed when the source layer is much deeper than the target, since the
    residual norm grows with depth; add: norm-matched addition). Returns the token list.
    target_layer ("all", k) (prereg A21): mode replace only; u OVERWRITES the residual at `pos` at the embedding and
    after every block 0..k (qwen3_min.forward inject_set), the blocks above k run clean."""
    tl = layer if target_layer is None else target_layer
    T = cid.shape[1]; m = torch.zeros(1, T, dtype=torch.bool, device=cid.device); m[0, pos] = True
    ad = q.w.emb.dtype if q.act_dtype is None else q.act_dtype
    with torch.no_grad():
        if is_multi_target(tl):
            assert mode == "replace", f"multi-layer target {tl} supports mode replace only"
            hL, cache = forward(q.w, cid, None, act_dtype=q.act_dtype, inject_set=(list(range(-1, int(tl[1]) + 1)), m, u.to(ad)))
        else:
            if mode == "add":
                d = ((alpha * hn / u.norm().clamp_min(1e-12)) * u).view(1, 1, -1).expand(1, T, -1)
            else:
                uu = u * (hn / u.norm().clamp_min(1e-12)) if mode == "replace_nm" else u
                x = q.resid(cid, tl)[0, pos]; d = (uu - x).view(1, 1, -1).expand(1, T, -1)
            hL, cache = forward(q.w, cid, inject=(tl, m, d.to(ad)), act_dtype=q.act_dtype)
        out = []
        for _ in range(n_gen):  # the patch is already baked into the cache; new tokens run clean
            t = int(true_logits(q.w, hL[0, -1]).argmax()); out.append(t)
            hL, cache = forward(q.w, torch.tensor([[t]], device=cid.device), None, cache, act_dtype=q.act_dtype)
    return out


CHECK_ENTITIES = ["Uzbekistan", "Gaborone", "Tchaikovsky", "cheetah", "1945", "Kathmandu", "Van Gogh", "144", "Sri Lanka", "sodium chloride"]


def literal_carrier(kind, prompt, entity):
    """the carrier with a LITERAL entity in its slot and no patch: identity / identity2 / identity3 "...; <entity>",
    describe "... <entity>:", describe2 "... <entity> ->" (the sanity check that the carrier format itself works)."""
    if kind in IDENTITY_KINDS: return prompt[: -1] + entity
    if kind == "describe": return prompt[: -2] + entity + ":"
    return prompt[: -4] + entity + " ->"


def carrier_check(q, tok, kinds, n_carriers, n_gen, log):
    """--carrier-check: greedy-decode every carrier variant with each CHECK_ENTITIES entity in the slot, un-patched.
    Reported per (kind, entity): the generations and the fraction whose cut() equals the entity (identity2 / describe2
    must echo the name; describe2 must continue "<entity> is <description>"). Written to results.json["carrier_check"]."""
    out = {}
    for k in kinds:
        for e in CHECK_ENTITIES:
            gens = []
            for c in CARRIERS[k][: n_carriers]:
                ids = tok(literal_carrier(k, c, e), return_tensors="pt").input_ids.to(q.w.emb.device)
                with torch.no_grad():
                    hL, cache = q.prefill(ids); toks = []
                    for _ in range(n_gen):
                        t = int(true_logits(q.w, hL[0, -1]).argmax()); toks.append(t)
                        hL, cache = forward(q.w, torch.tensor([[t]], device=ids.device), None, cache, act_dtype=q.act_dtype)
                gens.append(tok.decode(toks))
            it = {"string": " " + e, "pieces_text": [" " + e.split()[0]]}
            ok = [int(norm(cut(g, k)) == norm(it["string"])) for g in gens] if k in ("identity2", "identity3", "identity3_long", "describe2", "identity") else [int(norm(g).startswith(norm(e))) for g in gens]
            out[f"{k}:{e}"] = {"echo_rate": sum(ok) / len(ok), "gens": gens}
            log(f"carrier check {k} {e!r}: echo {sum(ok)}/{len(ok)}; {gens[:3]}")
    return out


def first_piece_hit(gen_ids, it, tok=None):
    """C2 fix: the first generated token (skipping one leading pure-whitespace token, e.g. the ' ' before a digit
    string) equals the item's first piece token id. Token-level, so the exemplar-inflated text prefix match is gone;
    for 1-character first pieces the flag stays uninformative and the summary restricts to p1_chars >= 2."""
    if not gen_ids: return 0
    g = list(gen_ids)
    if tok is not None and len(g) > 1 and tok.decode([g[0]]).strip() == "": g = g[1:]
    return int(g[0] == it["pieces"][0])


def score(gen_text, it, kind=None, gen_ids=None, tok=None):
    """(exact, prefix, first_piece) match flags of one generation against the item's string. `exact` is tested on the
    generation cut at the carrier's stop marker; `prefix` on the raw generation; `first_piece` at the token level when
    gen_ids is given (C2), else (old callers) as a text prefix of the first piece."""
    s = norm(it["string"]); g = norm(gen_text); p1 = norm(it["pieces_text"][0])
    fp = first_piece_hit(gen_ids, it, tok) if gen_ids is not None else int(bool(p1) and g.startswith(p1))
    return int(norm(cut(gen_text, kind)) == s), int(g.startswith(s)), fp


def boot(xs, n=1000, seed=0):
    if not xs: return None, None, None
    t = torch.tensor(xs, dtype=torch.float64); g = torch.Generator().manual_seed(seed)
    b = torch.stack([t[torch.randint(0, len(t), (len(t),), generator=g)].mean() for _ in range(n)])
    return float(t.mean()), float(b.quantile(0.025)), float(b.quantile(0.975))


def summarise(rows, seed=0):
    """per (arm, control): majority-vote exact recall (and prefix / first-piece), overall, per kind and per category;
    vote_first_p1ge2 restricts the first-piece flag to items with p1_chars >= 2 (C2); paired real-minus-control
    exact-vote differences with a paired bootstrap CI and McNemar counts."""
    out = {}
    for r in rows: out.setdefault(r["arm"], {}).setdefault(r["control"], []).append(r)
    summ = {}
    for arm, byc in out.items():
        summ[arm] = {}
        for ctrl, rs in byc.items():
            e = {"n": len(rs)}
            for k in ("vote_exact", "vote_prefix", "vote_first", "any_exact"):
                e[k], e[k + "_lo"], e[k + "_hi"] = boot([r[k] for r in rs], seed=seed)
            ge2 = [r["vote_first"] for r in rs if (r.get("p1_chars") or 0) >= 2]
            e["n_p1ge2"] = len(ge2); e["vote_first_p1ge2"] = sum(ge2) / len(ge2) if ge2 else None
            for key, sel in (("kind", lambda r: r["kind"]), ("category", lambda r: r["category"])):
                per = {}
                for r in rs: per.setdefault(sel(r), []).append(r)
                e["by_" + key] = {c: {"n": len(v), "vote_exact": sum(x["vote_exact"] for x in v) / len(v), "any_exact": sum(x["any_exact"] for x in v) / len(v),
                                      "vote_first": sum(x["vote_first"] for x in v) / len(v)} for c, v in sorted(per.items())}
            summ[arm][ctrl] = e
        if "none" in byc:
            real = {r["id"]: r for r in byc["none"] if r.get("id") is not None}
            for ctrl, rs in byc.items():
                if ctrl == "none": continue
                pairs = [(real[r["id"]], r) for r in rs if r.get("id") in real]; p = {"n_pairs": len(pairs)}
                for k in ("vote_exact", "any_exact"):
                    a = [float(x[k]) for x, y in pairs]; b = [float(y[k]) for x, y in pairs]
                    if a:
                        d = torch.tensor(a) - torch.tensor(b); g = torch.Generator().manual_seed(seed)
                        bs = torch.stack([d[torch.randint(0, len(d), (len(d),), generator=g)].mean() for _ in range(1000)])
                        p[k + "_diff"], p[k + "_diff_lo"], p[k + "_diff_hi"] = float(d.mean()), float(bs.quantile(0.025)), float(bs.quantile(0.975))
                        p[k + "_mcnemar"] = {"b": int(sum(x > y for x, y in zip(a, b))), "c": int(sum(y > x for x, y in zip(a, b)))}
                summ[arm][ctrl]["paired"] = p
    return summ


def verdict(summ):
    out = {}
    for arm, byc in summ.items():
        if "none" not in byc: continue
        v = byc["none"]["vote_exact"]; ctrl = max([byc[c]["vote_exact"] for c in byc if c != "none"] or [0.0])
        out[arm] = {"vote_exact": v, "control_max": ctrl, "pass": bool(v >= PASS_RECALL and ctrl <= CTRL_MAX), "kill": bool(v < KILL_RECALL)}
    return {"arms": out, "any_pass": any(x["pass"] for x in out.values()), "all_kill": bool(out) and all(x["kill"] for x in out.values())}


def parse_targets(s):
    """same | <block> | all<k> (A21: u written at the patch position at the embedding and after every block 0..k)."""
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if not x: continue
        if x in ("same", "-1"): out.append("same")
        elif x.startswith("all"): out.append(("all", int(x[3:])))
        else: out.append(int(x))
    return out or ["same"]


def target_key(t):
    """the arm-key suffix of a target: '' (same), ':t<l>' or ':all<k>'."""
    if t == "same": return ""
    return f":all{t[1]}" if is_multi_target(t) else f":t{t}"


def with_placeholder(prompt, kind, placeholder):
    """the carrier with its final "?" marker replaced by `placeholder` (identity kinds only; A21 placeholder swap)."""
    if not placeholder or kind not in IDENTITY_KINDS: return prompt
    assert prompt.endswith("?"), prompt
    return prompt[:-1] + placeholder


FIRSTHOP_RULE = re.compile(r"^firsthop([+-]\d+)?$")


def rule_pos(rule, it_or_pos, npos):
    """the read position of a firsthop[+-k] rule for an item (or a given firsthop position): None when the item has
    no first-hop position or the offset leaves the prompt."""
    fh = it_or_pos if isinstance(it_or_pos, int) or it_or_pos is None else it_or_pos.get("firsthop_pos")
    if fh is None: return None
    m = FIRSTHOP_RULE.match(rule); assert m, rule
    p = fh + int(m.group(1) or 0)
    return p if 0 <= p < npos else None


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--items", default=os.path.join(os.path.dirname(HERE), "data", "h2a_items.json"))
    p.add_argument("--split", default="heldout", help="heldout | tuning | all (answer ids of data/h2a_split.json)")
    p.add_argument("--split-file", default=os.path.join(os.path.dirname(HERE), "data", "h2a_split.json"))
    p.add_argument("--layers", default="22"); p.add_argument("--target-layer", default="same", help="comma list: same (= -1, the source layer) and/or block indices")
    p.add_argument("--carriers", default="identity,describe", help="comma list of identity, describe, identity2, describe2, identity3 (A12: item-free exemplars)"); p.add_argument("--n-carriers", type=int, default=8)
    p.add_argument("--modes", default="replace", help="comma list of replace, replace_nm (u rescaled to the target layer's carrier norm), add"); p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--gen", type=int, default=6); p.add_argument("--positions", default="last", help="last | all | firsthop | last,firsthop")
    p.add_argument("--controls", default="", help="comma list of shuffled,crosscat,random"); p.add_argument("--kinds", default="answer")
    p.add_argument("--min-p1-chars", type=int, default=0); p.add_argument("--per-category", type=int, default=0)
    p.add_argument("--max-items", type=int, default=0, help="cap the items AFTER gating"); p.add_argument("--max-scan", type=int, default=0, help="cap the items BEFORE gating (smokes: gating greedy-decodes every item)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--track", default="", help="A (h2a sub-word) | B (phrase) | S (single); default from the items file name")
    p.add_argument("--track-filter", default="", help="keep only items whose `track` field equals this (data/fresh_items.json holds both tracks; A24)")
    p.add_argument("--dtype", default="fp32", help="fp32 | bf16 | bf16-mixed (bf16 weights, fp32 activations; = fp32 for Qwen3)")
    p.add_argument("--device", default="cuda"); p.add_argument("--tag", default="x3_patchscope"); p.add_argument("--resume", action="store_true")
    p.add_argument("--no-item-gate", action="store_true", help="A26: keep every item without the greedy-answer gate (the neutral-context items were gated through their two-hop originals; their own prompts do not ask for the bridge by design)")
    p.add_argument("--no-real", action="store_true", help="run only the listed controls (no real 'none' rows); used to re-run controls after the A14 partner-position fix")
    p.add_argument("--placeholder", default="", help="A21: replace the final '?' of the identity-format carriers by this string (e.g. 'x', '_', '...'); arms get the suffix :ph=<string>")
    p.add_argument("--carrier-check", action="store_true", help="sanity check only: decode each carrier with a LITERAL entity in the slot, no patch, no items; write results.json and exit"); a = p.parse_args()
    out_dir = os.path.join(os.path.dirname(HERE), "runs", a.tag); os.makedirs(out_dir, exist_ok=True); out_f = os.path.join(out_dir, "results.json")
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True); log(" ".join(sys.argv))

    tok, m, q = load_q(a.model, a.dtype, a.device)
    if a.carrier_check:
        cc = carrier_check(q, tok, [k for k in a.carriers.split(",") if k], a.n_carriers, a.gen, log)
        json.dump({"model": a.model, "carrier_check": cc, "n_carriers": a.n_carriers, "gen": a.gen}, open(out_f, "w"), indent=1, ensure_ascii=False); log(f"carrier check -> {out_f}"); return
    items = e12.split_items(e12.subset(e12.load_items(tok, a.items), a), a.split_file, a.split)
    if a.max_scan: items = items[: a.max_scan]
    kept = []
    for it in items:
        pid = tok(it["prompt"], return_tensors="pt").input_ids.to(a.device)
        it["correct"], it["gated"], it["model_greedy"] = e12.gate(q, tok, it, pid); it["prompt_ids"] = pid
        it["firsthop_pos"] = e12.firsthop_pos(tok, it, pid[0].tolist())
        if a.no_item_gate or (it["correct"] and it["gated"]): kept.append(it)
    if a.max_items: kept = kept[: a.max_items]
    log(f"{len(items)} items ({a.split} split, kinds {a.kinds}); {len(kept)} gated")
    kinds = [k for k in a.carriers.split(",") if k]; modes = [x for x in a.modes.split(",") if x]; ctrls = ([] if a.no_real else ["none"]) + [c for c in a.controls.split(",") if c]
    targets = parse_targets(a.target_layer); pos_rules = [x for x in a.positions.split(",") if x]
    # A12: exemplar / item overlap, recorded per kind; identity3 must be clean (run-time assertion, beside the import-time one)
    overlap = {k: [list(x) for x in carrier_overlap(k)] for k in kinds if k in IDENTITY_KINDS}
    if "identity3" in kinds: assert_no_overlap("identity3")
    for k, ov in overlap.items():
        if ov: log(f"WARNING carrier kind {k}: {len(ov)} exemplar / item overlaps (A12 sensitivity analysis: exclude these items): {sorted({e for e, _, _ in ov})}")
    prompts = {k: [with_placeholder(s, k, a.placeholder) for s in CARRIERS[k][: a.n_carriers]] for k in kinds}
    car = {k: [tok(s, return_tensors="pt").input_ids.to(a.device) for s in prompts[k]] for k in kinds}
    cpos = {k: [patch_pos(tok, c, k) for c in car[k]] for k in kinds}
    if a.placeholder:  # A21: the swapped marker must be its own final token, else the patch position would not be the marker
        for k in kinds:
            if k in IDENTITY_KINDS: assert all(tok.decode([int(c[0, -1])]).strip() == a.placeholder.strip() for c in car[k]), f"--placeholder {a.placeholder!r} is not a single final token of the {k} carriers"
    log("carrier patch tokens: " + "; ".join(f"{k} " + repr([tok.decode([int(c[0, p])]) for c, p in zip(car[k], cpos[k])][:3]) for k in kinds))
    strings = [it["string"] for it in kept]; cats = [it["category"] for it in kept]
    perm = e12.shuffled_perm(cats, a.seed, strings); xperm = e12.cross_perm(cats, a.seed, strings)
    n_fh = sum(it["firsthop_pos"] is not None for it in kept)
    results = {"model": a.model, "dtype": a.dtype, "split": a.split, "kinds": kinds, "n_carriers": a.n_carriers, "modes": modes, "controls": ctrls,
               "positions": a.positions, "track": e12.track_of(a.items, a.track), "gen": a.gen, "target_layer": a.target_layer, "target_layers": [list(t) if is_multi_target(t) else t for t in targets], "seed": a.seed, "n_items": len(items), "n_kept": len(kept),
               "n_shuffled_dropped": int((perm < 0).sum()), "n_crosscat_dropped": int((xperm < 0).sum()), "n_firsthop": n_fh, "n_no_firsthop": len(kept) - n_fh,
               "carrier_prompts": prompts, "placeholder": a.placeholder or None, "stops": {k: STOPS[k] for k in kinds}, "carrier_overlap": overlap,
               "prereg": "docs/prereg_R.yaml (R3, R5; amendments A5, A12, A21 multi-layer targets / placeholder, A22 firsthop+-k)", "layers": {}}
    log(f"controls: shuffled partners missing for {results['n_shuffled_dropped']} items, crosscat for {results['n_crosscat_dropped']}; first-hop position known for {n_fh}/{len(kept)}")
    done = set()
    if a.resume and os.path.exists(out_f):
        old = json.load(open(out_f)); results["layers"] = old.get("layers", {})
        done = {(int(l), r["id"], r["arm"], r["control"]) for l, s in results["layers"].items() for r in s["rows"]}; log(f"resuming: {len(done)} rows")

    ph_suf = f":ph={a.placeholder}" if a.placeholder else ""
    for l in [int(x) for x in a.layers.split(",")]:
        t0 = time.time(); tls = {t: (l if t == "same" else t) for t in targets}
        norm_layer = lambda tl: int(tl[1]) if is_multi_target(tl) else tl  # replace_nm / add norms at the top of a multi-layer set
        hn = {(k, tl): float(torch.stack([q.resid(c, norm_layer(tl))[0, p].float().norm() for c, p in zip(car[k], cpos[k])]).mean()) for k in kinds for tl in set(tls.values())}
        summ = results["layers"].setdefault(str(l), {"rows": []}); rows = summ["rows"]
        g = torch.Generator().manual_seed(a.seed + l)
        H = {}
        for it in kept:
            with torch.no_grad(): H[it["id"]] = q.resid(it["prompt_ids"], l)[0]  # [T_prompt, d]: every prompt position
        for n, it in enumerate(kept):
            hall = H[it["id"]]; npos = hall.shape[0]
            hs_all = {"none": hall}
            partner = {}  # control -> partner item; its source position is taken from the PARTNER's own prompt (A14 fix)
            if int(perm[n]) >= 0: partner["shuffled"] = kept[int(perm[n])]; hs_all["shuffled"] = H[partner["shuffled"]["id"]]
            if int(xperm[n]) >= 0: partner["crosscat"] = kept[int(xperm[n])]; hs_all["crosscat"] = H[partner["crosscat"]["id"]]
            r0 = torch.randn(hall.shape, generator=g, dtype=torch.float32).to(hall.device, hall.dtype)
            hs_all["random"] = r0 / r0.norm(dim=-1, keepdim=True) * hall.norm(dim=-1, keepdim=True)
            base = {"id": it["id"], "category": it["category"], "kind": it["kind"], "string": it["string"], "p1_chars": it.get("p1_chars"), "n_positions": npos,
                    "firsthop_pos": it["firsthop_pos"], "shuffled_from": kept[int(perm[n])]["id"] if int(perm[n]) >= 0 else None}
            for rule in pos_rules:
                if rule == "last": poss = [npos - 1]
                elif rule == "all": poss = list(range(npos))
                elif FIRSTHOP_RULE.match(rule):  # firsthop | firsthop-1 | firsthop+1 ... (A22: the first-hop token and its neighbours)
                    fp = rule_pos(rule, it, npos)
                    if fp is None: continue
                    poss = [fp]
                else: raise ValueError(rule)
                psuf = "" if (rule == "last" and len(pos_rules) == 1) or rule == "all" else f":{rule}"
                for ctrl in ctrls:
                    if ctrl not in hs_all: continue
                    if ctrl in partner and FIRSTHOP_RULE.match(rule) and rule_pos(rule, partner[ctrl], hs_all[ctrl].shape[0]) is None: continue
                    for kind in kinds:
                        for mode in modes:
                            for target in targets:
                                tl = tls[target]; arm = f"{kind}:{mode}" + target_key(target) + psuf + ph_suf
                                if (l, it["id"], arm, ctrl) in done: continue
                                prof = []  # per source position: (exact votes, prefix votes, first votes, sample generation)
                                for pos in poss:
                                    if ctrl in partner and rule == "last": sp = hs_all[ctrl].shape[0] - 1  # partner's own last token
                                    elif ctrl in partner and FIRSTHOP_RULE.match(rule): sp = rule_pos(rule, partner[ctrl], hs_all[ctrl].shape[0])  # partner's own first-hop (+-k) token
                                    else: sp = min(pos, hs_all[ctrl].shape[0] - 1)  # real / random: same index; 'all' scan: clipped index
                                    u = hs_all[ctrl][sp]
                                    ex = pr = fp = 0; gens = []
                                    for cid, cp in zip(car[kind], cpos[kind]):
                                        out = patched_generate(q, cid, l, cp, u, a.gen, mode, a.alpha, hn[(kind, tl)], tl)
                                        txt = tok.decode(out); gens.append(txt); e, p_, f_ = score(txt, it, kind, out, tok); ex += e; pr += p_; fp += f_
                                    nc = len(car[kind])
                                    prof.append({"pos": pos, "exact": ex / nc, "prefix": pr / nc, "first": fp / nc, "gen": gens[0], "gens": gens if rule != "all" else None})
                                best = max(prof, key=lambda x: (x["exact"], x["prefix"], x["first"]))
                                row = {**base, "arm": arm, "control": ctrl, "position_rule": rule, "target_layer": tl,
                                       "vote_exact": int(best["exact"] > 0.5), "vote_prefix": int(best["prefix"] > 0.5), "vote_first": int(best["first"] > 0.5),
                                       "any_exact": int(best["exact"] > 0), "best_pos": best["pos"], "best_gen": best["gen"], "last_gen": prof[-1]["gen"]}
                                if rule == "all": row["profile"] = [{k: v for k, v in e_.items() if k != "gens"} for e_ in prof]  # per-position profile (best_pos is a per-item selection)
                                else: row["gens"] = prof[0]["gens"]
                                rows.append(row)
            shown = [r for r in rows if r["id"] == it["id"] and r["control"] == "none"]
            log(f"L{l} {n+1}/{len(kept)} {it['string']!r} ({it['category']}): " + ", ".join(f"{r['arm']} vote {r['vote_exact']} @pos {r['best_pos']} gen {r['best_gen']!r}" for r in shown[:3]))
            summ["summary"] = summarise(rows, a.seed); summ["seconds"] = time.time() - t0
            json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
        summ["summary"] = summarise(rows, a.seed); summ["verdict"] = verdict(summ["summary"]); summ["seconds"] = time.time() - t0
        if "all" in pos_rules and rows:  # mean over items of the per-position exact rate, by relative depth in the prompt
            pp = {}
            for r in rows:
                if r["control"] != "none" or "profile" not in r: continue
                for e in r["profile"]: pp.setdefault(round(e["pos"] / max(r["n_positions"] - 1, 1), 2), []).append(e["exact"])
            summ["position_profile"] = {str(k): {"n": len(v), "exact": sum(v) / len(v)} for k, v in sorted(pp.items())}
        json.dump(results, open(out_f, "w"), indent=1, ensure_ascii=False)
        for arm, byc in sorted(summ["summary"].items()):
            for ctrl, e in sorted(byc.items()):
                fp2 = f"{e['vote_first_p1ge2']:.3f} (n {e['n_p1ge2']})" if e["vote_first_p1ge2"] is not None else "-"
                log(f"L{l} {arm} [{ctrl}]: n {e['n']}, exact-string (majority vote) {e['vote_exact']:.3f} [{e['vote_exact_lo']:.3f}, {e['vote_exact_hi']:.3f}], "
                    f"any-carrier {e['any_exact']:.3f}, prefix {e['vote_prefix']:.3f}, first-piece {e['vote_first']:.3f} (p1_chars >= 2: {fp2})")
        v = summ["verdict"]; log(f"L{l} verdict vs prereg R3 (pass >= {PASS_RECALL} with controls <= {CTRL_MAX}; kill < {KILL_RECALL}): "
                                 f"any_pass {v['any_pass']}, all_kill {v['all_kill']}; {summ['seconds']:.0f}s")


if __name__ == "__main__":
    main()
