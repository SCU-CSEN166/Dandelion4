"""Interface Spec v2 NLP extraction and PII redaction system powered by GLiNER.

Loads ``urchade/gliner_medium-v2.1`` (preferring ``models/gliner_medium-v2.1`` when present
locally) once per ``GLiNERSystem`` instance, identifies person names and phone numbers for strict
v2 ``NAME`` / ``PHONE`` redactions, determines activity relevance and v2 ``activity_type`` in
reading order, copies trimmed attribute fields (``location``, ``date_text``, ``event_name``,
``event_desc``), and extracts ``reported_count`` from reported work counts.

GLiNER configuration:
    - Model: ``urchade/gliner_medium-v2.1`` (local fallback ``models/gliner_medium-v2.1``)
    - Labels: ``("person", "location", "facility", "event", "date", "phone number")``
    - Threshold: ``0.38`` (balanced recall before deterministic boundary/type filtering)
    - Relevance / activity / count: rule + semantic hybrid over reported-work clauses; plans,
      schedules, questions, personal logistics, and greetings mark ``relevant=False``.
    - Confidence calculation: mean GLiNER score over retained entity evidence (clamped ``0..1``),
      or a conservative ``0.75`` (relevant) / ``0.80`` (irrelevant) heuristic confidence when no
      entity spans contribute.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.asr import clean_transcripts  # noqa: E402
from src.core import contracts  # noqa: E402

MODEL_ID = "urchade/gliner_medium-v2.1"
LOCAL_MODEL_DIR = REPO_ROOT / "models" / "gliner_medium-v2.1"
LOCAL_DEBERTA_DIR = REPO_ROOT / "models" / "deberta-v3-base"
LABELS: tuple[str, ...] = ("person", "location", "facility", "event", "date", "phone number")
THRESHOLD = 0.38

_PERSON_TITLES = (
    "Coordinator",
    "Dr.",
    "Dr",
    "Mrs.",
    "Mrs",
    "Mr.",
    "Mr",
    "Ms.",
    "Ms",
    "Miss",
    "Sister",
    "Brother",
    "Mama",
    "Father",
    "Pastor",
    "CEO",
    "VP",
    "Corporal",
    "Sergeant",
    "Chief",
    "Nurse",
)
_TITLE_RE = re.compile(
    r"(?:^|\b)(" + "|".join(re.escape(t).rstrip(r"\.") + r"\.?" for t in _PERSON_TITLES) + r")\s+$",
    re.IGNORECASE,
)

_PHONE_PATTERNS = (
    re.compile(r"(?<!\w)(?:\+?254[\s-]?0?|0)[17](?:[\s-]?\d){8}(?!\w)"),
    re.compile(r"(?<!\w)\+\d{1,3}(?:[\s-]?\(?\d{2,4}\)?){2,4}(?!\w)"),
)

# Syntactic person intro/contact cues for accented/lowercase ASR transcripts.
_INTRO_NAME_RE = re.compile(
    r"\b(?:this is|it's|its|my name is|called|named|helping)\s+"
    r"(?:(?:"
    + "|".join(re.escape(t).rstrip(r"\.") + r"\.?" for t in _PERSON_TITLES)
    + r")\s+)?"
    r"([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+){0,2})(?=\s+(?:here|reporting|from|at|in|on|with|and)|[,.]|$)",
    re.IGNORECASE,
)
_GREETING_NAME_RE = re.compile(
    r"\b(?:Hello|Hi|Hey)\s+team[,\s]+([A-Z][a-z]{5,})(?=\s+(?:I|we|here|reporting)\b)"
)
_ROLE_TITLE_NAME_RE = re.compile(
    r"\b(?:Dr\.|Dr|Mrs\.|Mrs|Mr\.|Mr|Ms\.|Ms|Sister|Brother|Mama|Pastor|Corporal|Sergeant)"
    r"\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"
)
_TRANSFER_NAME_RE = re.compile(
    r"\b(?:to|call|contact)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,1})\s+"
    r"(?:on\s+(?:Mpesa|WhatsApp|07|\+?254)|at\s+(?:07|\+?254))",
    re.IGNORECASE,
)
_POSSESSIVE_PHONE_NAME_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:'s|\s+s)\s+(?:phone\s+)?number\b"
)

_PLACE_SUFFIX_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]*(?:\s+(?:[A-Z][A-Za-z0-9]*|ya|Level|level|\d+)){0,4}\s+"
    r"(?:Dispensary|Health\s+Cent(?:er|re)|Hospital|Clinic|Primary\s+School|Secondary\s+School|"
    r"School|Community\s+Center|Community\s+Hall))\b"
)
_CHIEFS_CAMP_RE = re.compile(r"\bchief(?:'s|s)?\s+camp\b", re.IGNORECASE)
_LEVEL_HOSPITAL_RE = re.compile(
    r"\b([A-Z][a-z]+)\s+level\s+(\d+)(?:\s+hospital)?\b",
    re.IGNORECASE,
)
_KNOWN_LOCATIONS = (
    "Baringo",
    "Nakuru",
    "Narok",
    "Kajiado",
    "Naivasha",
    "Marigat",
    "Nairobi",
    "Kisumu",
    "Mombasa",
    "Eldoret",
    "Kiambu",
    "Mogotio",
    "Rongai",
    "Solai",
    "Kabarak",
)
_PREP_PLACE_RE = re.compile(
    r"\b(?:in|at|from|to|near)\s+(?:the\s+)?"
    r"([A-Z][A-Za-z]+(?:\s+(?:ya|Level|\d+|[A-Z][A-Za-z]+)){0,3})\b"
)

_NON_PERSON_WORDS = frozenset(
    {
        "everyone",
        "team",
        "tim",
        "ladies",
        "gentlemen",
        "women",
        "mothers",
        "girls",
        "boys",
        "men",
        "children",
        "members",
        "families",
        "households",
        "participants",
        "patients",
        "residents",
        "volunteers",
        "elders",
        "farmers",
        "parents",
        "people",
        "groups",
        "managers",
        "chv",
        "nurse",
        "officer",
        "police",
        "gender",
        "desk",
        "village",
        "agriculture",
        "head",
        "teacher",
        "treasurer",
        "counselor",
        "coordinator",
        "doctor",
        "today",
        "yesterday",
        "tomorrow",
        "morning",
        "afternoon",
        "evening",
        "night",
        "happy",
        "fine",
        "sorry",
        "good",
        "bad",
        "asante",
        "asanti",
        "here",
        "there",
        "home",
        "school",
        "hospital",
        "clinic",
        "dispensary",
        "camp",
        "center",
        "centre",
        "office",
        "whatsapp",
        "mpesa",
        "shillings",
        "english",
        "me",
        "you",
        "us",
        "her",
        "him",
        "them",
        "someone",
        "anyone",
        "nobody",
        "nothing",
        "something",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "sukuma",
        "wiki",
        "spinach",
        "technophone",
        "mandotsi",
        "mandazi",
        "implants",
        "outreach",
        "screening",
        "meeting",
        "training",
        "session",
        "dialogue",
        "banking",
        "follow-up",
        "followup",
    }
)

_NON_LOCATION_WORDS = frozenset(
    {
        "today",
        "tomorrow",
        "yesterday",
        "morning",
        "afternoon",
        "evening",
        "tonight",
        "weekend",
        "month",
        "week",
        "year",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "whatsapp",
        "mpesa",
        "english",
        "home",
        "work",
        "details",
        "stock",
        "total",
        "queue",
        "group",
        "class",
        "grace",
        "nancy",
        "mary",
        "ruth",
        "lucy",
        "esther",
        "peter",
        "rose",
        "faith",
        "janet",
        "caroline",
        "joyce",
        "beatrice",
        "agnes",
    }
)

_LEADING_TRIM_RE = re.compile(r"^(?:the|a|an|in|at|on|by)\s+", re.IGNORECASE)

_STRONG_CHITCHAT_RE = re.compile(
    r"\b(?:"
    r"congratulations|happy birthday|new baby|baby shower|wedding|"
    r"contribute\s+\d+\s+shillings\s+for\s+a\s+(?:small\s+)?gift|"
    r"broken?\s+down|broke\s+down|left\s+mine\s+at\s+home|charger|"
    r"who\s+is\s+bringing\s+tea|bring\s+\d+\s+pens|need\s+moving|where\s+to\s+buy|"
    r"wishing\s+you|good\s+(?:morning|afternoon|evening)\s+everyone\b(?!.*\b(?:attended|screened|visited|referred|held|outreach|training|session|dialogue|meeting)\b)"
    r")\b",
    re.IGNORECASE,
)

_FUTURE_ONLY_CUES_RE = re.compile(
    r"\b(?:scheduled\s+for|let'?s\s+plan\s+to|planning\s+to|will\s+be\s+(?:visiting|holding|having)|"
    r"want\s+to\s+invite|what\s+time\s+is|can\s+we\s+schedule)\b",
    re.IGNORECASE,
)

_REPORTED_WORK_CUES_RE = re.compile(
    r"\b(?:"
    r"visited|screened|referred|attended|came|served|held|organized|conducted|"
    r"completed|checked\s+on|reached|followed\s+up|delivered|gave\s+services|"
    r"received\s+(?:seedlings|packs|sanitary|supplements)|saved\s+[\d,]+\s+shillings|"
    r"table\s+banking|outreach|screening(?:\s+camp)?|community\s+meeting|"
    r"community\s+dialogue|training|hygiene\s+session|household\s+outreach|"
    r"health\s+referral(?:s)?|follow-up\s+visit|workshop|was\s+helping|"
    r"reporting\s+from|want\s+to\s+report|want\s+to\s+share\s+about|"
    r"yesterday\s+we\s+went|today\s+was\s+our|today\s+we\s+(?:did|held|visited|screened|referred)"
    r")\b",
    re.IGNORECASE,
)

_ACTIVITY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "health_screening",
        re.compile(
            r"\b(?:screen(?:ed|ing)?|health\s+camp|cervical\s+cancer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "household_outreach",
        re.compile(
            r"\b(?:household\s+outreach|visited\s+.*households?|home\s+visits?|"
            r"reached\s+\d*\s*(?:families|households|homes)|follow-up\s+visit)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "health_referral",
        re.compile(
            r"\b(?:referred\s+\d*\s*(?:women|patients|mothers|people)|health\s+referrals?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "community_meeting",
        re.compile(
            r"\b(?:meeting|table\s+banking|community\s+dialogue|dialogue|workshop|"
            r"training|hygiene\s+session|outreach|talk)\b",
            re.IGNORECASE,
        ),
    ),
)

_PARTICIPANT_NOUNS = (
    r"women|mothers|girls|boys|men|members|participants|attendees|patients|"
    r"families|households|homes|people|students|farmers|volunteers|residents|elders"
)
_COUNT_BEFORE_NOUN_RE = re.compile(
    rf"\b(?:(?:about|around|over|nearly)\s+)?(\d+)(?:\s*(?:-|to)\s*\d+)?\s+(?:[A-Za-z]+\s+){{0,2}}"
    rf"(?:{_PARTICIPANT_NOUNS})\b",
    re.IGNORECASE,
)
_ADDITIVE_COUNT_RE = re.compile(
    r"\b(\d+)\s+(?:women|men|mothers|girls|boys|participants|members)\s+and\s+"
    r"(\d+)\s+(?:women|men|mothers|girls|boys|participants|members)\b",
    re.IGNORECASE,
)
_IGNORE_NUMBER_CONTEXT_RE = re.compile(
    r"(?:shillings|level\s+\d+|class\s+\d+(?:\s+to\s+\d+)?|\d+\s+am\b|\d+\s+pm\b|"
    r"\d+\s+in\s+the\s+morning|\bon\s+the\s+\d+(?:st|nd|rd|th)?\b)",
    re.IGNORECASE,
)
_SUBSET_SENTENCE_RE = re.compile(
    r"\b(?:referred|follow-up|private\s+follow-up|positive\s+results|small\s+loans|left\s+home)\b",
    re.IGNORECASE,
)


def resolve_gliner_model(model_id: str | None = None) -> tuple[str, bool]:
    """Return ``(model_source, local_files_only)``, preferring ``models/gliner_medium-v2.1``."""
    if model_id is not None:
        p = Path(model_id)
        return (str(p), True) if p.is_dir() else (model_id, False)
    if (LOCAL_MODEL_DIR / "gliner_config.json").is_file() and (
        LOCAL_MODEL_DIR / "pytorch_model.bin"
    ).is_file():
        return str(LOCAL_MODEL_DIR), True
    return MODEL_ID, False


def trim_copied_value(value: str | None, text_redacted: str | None = None) -> str | None:
    """Normalize a copied text attribute per Interface Spec v2 §4 rules."""
    if value is None:
        return None
    cleaned = value.strip()
    while True:
        updated = _LEADING_TRIM_RE.sub("", cleaned).strip()
        if updated == cleaned:
            break
        cleaned = updated
    while cleaned and not cleaned.endswith(("[NAME]", "[PHONE]")):
        if unicodedata.category(cleaned[-1]).startswith("P"):
            cleaned = cleaned[:-1].rstrip()
        else:
            break
    if not cleaned:
        return None
    if text_redacted is not None and cleaned.casefold() not in text_redacted.casefold():
        return None
    return cleaned


def _merge_spans(candidates: list[tuple[int, int, str]]) -> list[dict[str, Any]]:
    """Resolve overlapping candidate spans cleanly by priority then leftmost/longest."""
    priority = {"PHONE": 2, "NAME": 1, "LOCATION": 0}
    ordered = sorted(
        candidates,
        key=lambda item: (item[0], -priority.get(item[2], 0), -(item[1] - item[0])),
    )
    accepted: list[dict[str, Any]] = []
    for start, end, typ in ordered:
        if start >= end:
            continue
        if not accepted:
            accepted.append({"type": typ, "start": start, "end": end})
            continue
        prev = accepted[-1]
        if start < int(prev["end"]):
            # Replace only when strictly higher priority or identical type with larger span.
            prev_prio = priority.get(str(prev["type"]), 0)
            cur_prio = priority.get(typ, 0)
            if cur_prio > prev_prio or (
                cur_prio == prev_prio and end - start > int(prev["end"]) - int(prev["start"])
            ):
                accepted[-1] = {"type": typ, "start": start, "end": end}
            continue
        accepted.append({"type": typ, "start": start, "end": end})
    return accepted


def _looks_like_person_phrase(phrase: str, location_hints: set[str]) -> bool:
    tokens = [re.sub(r"[^A-Za-z']", "", t).lower().rstrip("'") for t in phrase.split()]
    tokens = [t for t in tokens if t]
    if not tokens or len(tokens) > 4:
        return False
    if phrase.casefold() in location_hints:
        return False
    if any(
        w in phrase.casefold()
        for w in (
            "dispensary",
            "hospital",
            "clinic",
            "school",
            "center",
            "centre",
            "program",
            "meeting",
            "outreach",
            "screening",
            "training",
            "session",
            "dialogue",
            "whatsapp",
            "mpesa",
        )
    ):
        return False
    title_set = {x.lower().rstrip(".") for x in _PERSON_TITLES}
    title_only = all(t in _NON_PERSON_WORDS or t in title_set for t in tokens)
    if title_only:
        return False
    return all(
        t not in {"in", "at", "on", "to", "from", "the", "a", "an", "of", "for"} for t in tokens
    )


def _expand_person_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Attach preceding title tokens and strip articles / role nouns per §5."""
    chunk = text[start:end]
    lead = re.match(
        r"^(?:the\s+|a\s+|an\s+|nurse\s+|officer\s+|treasurer\s+|head\s+teacher\s+)+",
        chunk,
        re.IGNORECASE,
    )
    if lead:
        start += lead.end()
    prefix = text[:start]
    title_match = _TITLE_RE.search(prefix)
    if title_match:
        new_start = title_match.start(1)
        art_before = re.match(r"^(?:the|a|an)\s+$", text[:new_start], re.IGNORECASE)
        if not art_before:
            start = new_start
    span_text = text[start:end]
    if span_text.endswith("'s"):
        end -= 2
    while end > start and text[end - 1].isspace():
        end -= 1
    while start < end and text[start].isspace():
        start += 1
    return start, end


def extract_phone_spans(
    text: str, entities: list[dict[str, Any]] | None = None
) -> list[tuple[int, int, str]]:
    """Extract PHONE spans from regex rules plus high-digit GLiNER phone spans."""
    spans: list[tuple[int, int, str]] = []
    for pattern in _PHONE_PATTERNS:
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), "PHONE"))
    for ent in entities or ():
        if str(ent.get("label", "")).lower() == "phone number":
            s, e = int(ent["start"]), int(ent["end"])
            raw = text[s:e]
            digits = sum(c.isdigit() for c in raw)
            if digits >= 7 and not re.search(r"shillings", raw, re.IGNORECASE):
                spans.append((s, e, "PHONE"))
    return spans


def extract_name_spans(
    text: str,
    entities: list[dict[str, Any]],
    location_hints: set[str],
) -> list[tuple[int, int, str]]:
    """Extract NAME spans combining GLiNER ``person`` predictions with syntactic person cues."""
    candidates: list[tuple[int, int, str]] = []
    for ent in entities:
        if str(ent.get("label", "")).lower() != "person":
            continue
        s, e = _expand_person_span(text, int(ent["start"]), int(ent["end"]))
        if s < e and _looks_like_person_phrase(text[s:e], location_hints):
            candidates.append((s, e, "NAME"))

    for pattern in (
        _ROLE_TITLE_NAME_RE,
        _INTRO_NAME_RE,
        _GREETING_NAME_RE,
        _TRANSFER_NAME_RE,
        _POSSESSIVE_PHONE_NAME_RE,
    ):
        for match in pattern.finditer(text):
            group_idx = 1 if match.lastindex else 0
            s, e = _expand_person_span(text, match.start(group_idx), match.end(group_idx))
            phrase = text[s:e]
            if s < e and _looks_like_person_phrase(phrase, location_hints):
                # For intro lowercase patterns, discard common filler clauses.
                first_tok = phrase.split()[0].lower()
                if first_tok not in _NON_PERSON_WORDS:
                    candidates.append((s, e, "NAME"))
    return candidates


def extract_location_spans(
    text: str,
    entities: list[dict[str, Any]],
    blocked_spans: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Extract non-overlapping LOCATION spans (for v2 ``location`` field and CSV anonymization)."""
    blocked = blocked_spans or []

    def overlaps_blocked(s: int, e: int) -> bool:
        return any(s < int(b["end"]) and e > int(b["start"]) for b in blocked)

    candidates: list[tuple[int, int, str]] = []
    for pattern in (_PLACE_SUFFIX_RE, _LEVEL_HOSPITAL_RE):
        for match in pattern.finditer(text):
            s, e = match.start(), match.end()
            if not overlaps_blocked(s, e):
                candidates.append((s, e, "LOCATION"))

    for match in _CHIEFS_CAMP_RE.finditer(text):
        s, e = match.start(), match.end()
        if not overlaps_blocked(s, e):
            candidates.append((s, e, "LOCATION"))

    for place in _KNOWN_LOCATIONS:
        for match in re.finditer(rf"\b{re.escape(place)}\b", text, re.IGNORECASE):
            s, e = match.start(), match.end()
            if not overlaps_blocked(s, e):
                candidates.append((s, e, "LOCATION"))

    title_words = {t.lower().rstrip(".") for t in _PERSON_TITLES}

    def _valid_location(phrase: str) -> bool:
        words = [re.sub(r"[^a-z']", "", w.lower()) for w in phrase.split()]
        words = [w for w in words if w]
        if not words or all(w in _NON_LOCATION_WORDS for w in words):
            return False
        if words[0] in _NON_LOCATION_WORDS:
            return False
        if any(w in title_words for w in words):
            return False
        venue_words = {"center", "centre", "hall", "school", "clinic", "hospital"}
        return not (
            any(w in {"women", "women's", "womens", "men", "too", "many"} for w in words)
            and not any(w in venue_words for w in words)
        )

    for match in _PREP_PLACE_RE.finditer(text):
        s, e = match.start(1), match.end(1)
        phrase = text[s:e].strip()
        if s < e and not overlaps_blocked(s, e) and _valid_location(phrase):
            candidates.append((s, e, "LOCATION"))

    for ent in entities:
        label = str(ent.get("label", "")).lower()
        if label not in {"location", "facility"}:
            continue
        s, e = int(ent["start"]), int(ent["end"])
        trimmed = trim_copied_value(text[s:e])
        if not trimmed:
            continue
        adj_s = text.find(trimmed, s, e)
        if adj_s == -1:
            continue
        adj_e = adj_s + len(trimmed)
        if not overlaps_blocked(adj_s, adj_e) and _valid_location(trimmed):
            candidates.append((adj_s, adj_e, "LOCATION"))

    return _merge_spans(candidates)


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in re.finditer(r"[.!?]+(?:\s+|$)", text):
        end = match.end()
        piece = text[start:end].strip()
        if piece:
            spans.append((start, end, piece))
        start = end
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text), text[start:].strip()))
    return spans or [(0, len(text), text)]


def classify_message_semantics(text: str) -> tuple[bool, str | None, str | None]:
    """Return ``(relevant, v2_activity_type, specific_event_name)``."""
    sentences = _split_sentences(text)
    has_strong_report = any(_REPORTED_WORK_CUES_RE.search(sent) for _, _, sent in sentences)
    has_chitchat_marker = bool(_STRONG_CHITCHAT_RE.search(text))

    # Reject pure personal/chitchat messages and pure future/scheduled plans with no work report.
    if has_chitchat_marker and not has_strong_report:
        return False, None, None
    if _FUTURE_ONLY_CUES_RE.search(text) and not has_strong_report:
        return False, None, None

    # Also check messages ending in '?' without any past activity report.
    if text.strip().endswith("?") and not has_strong_report:
        return False, None, None

    # Find first sentence reporting an activity, or fall back to full text.
    target_text = text
    for _, _, sent in sentences:
        if _REPORTED_WORK_CUES_RE.search(sent):
            target_text = sent
            break

    if not has_strong_report:
        return False, None, None

    matched_v2: str | None = None
    for act_type, pattern in _ACTIVITY_RULES:
        if pattern.search(target_text):
            matched_v2 = act_type
            break
    if matched_v2 is None:
        for act_type, pattern in _ACTIVITY_RULES:
            if pattern.search(text):
                matched_v2 = act_type
                break
    if matched_v2 is None:
        matched_v2 = "other"

    event_name = _extract_event_name(text)
    return True, matched_v2, event_name


_EVENT_PHRASE_PATTERNS = (
    re.compile(
        r"\b((?:cervical\s+cancer\s+)?screening(?:\s+camp)?|"
        r"(?:family\s+planning|household)\s+outreach|"
        r"(?:family\s+planning)|"
        r"(?:monthly\s+)?table\s+banking(?:\s+day)?|"
        r"menstrual\s+hygiene\s+session|"
        r"community\s+dialogue(?:\s+on\s+gender-based\s+violence)?|"
        r"kitchen\s+garden\s+training|"
        r"health\s+training|community\s+meeting|community\s+training|"
        r"follow-up\s+visit|health\s+referrals?|nutrition\s+talk|workshop)\b",
        re.IGNORECASE,
    ),
)


def _extract_event_name(
    text: str,
    entities: list[dict[str, Any]] | None = None,
    location_hints: set[str] | None = None,
) -> str | None:
    for pattern in _EVENT_PHRASE_PATTERNS:
        m = pattern.search(text)
        if m:
            val = trim_copied_value(m.group(1))
            if val:
                if val.lower() == "screening camp" and re.search(
                    r"\bcervical\s+cancer\b", text, re.IGNORECASE
                ):
                    return "cervical cancer screening camp"
                return val
    hints = location_hints or set()
    for ent in entities or ():
        if str(ent.get("label", "")).lower() == "event":
            val = trim_copied_value(str(ent.get("text", "")))
            if val and not any(tok.casefold() in hints for tok in val.split()):
                return val
    return None


def extract_reported_count(text: str) -> int | None:
    """Apply §4 counting rules using word-to-number normalization and noise filters."""
    normalized = clean_transcripts.convert_number_words(text)
    # Strip phone numbers before extracting counts so phone digits never leak.
    scrubbed = normalized
    for pat in _PHONE_PATTERNS:
        scrubbed = pat.sub("[PHONE]", scrubbed)

    # Search sentences prioritizing primary attendance over follow-up/referral subsets.
    sentences = _split_sentences(scrubbed)
    ordered_sents = sorted(
        sentences,
        key=lambda item: (1 if _SUBSET_SENTENCE_RE.search(item[2]) else 0, item[0]),
    )

    for _, _, sent in ordered_sents:
        cleaned_sent = _IGNORE_NUMBER_CONTEXT_RE.sub(" ", sent)
        add_match = _ADDITIVE_COUNT_RE.search(cleaned_sent)
        if add_match:
            return int(add_match.group(1)) + int(add_match.group(2))
        count_match = _COUNT_BEFORE_NOUN_RE.search(cleaned_sent)
        if count_match:
            return int(count_match.group(1))
    return None


class GLiNERSystem:
    """Interface Spec v2 NLP system wrapping ``urchade/gliner_medium-v2.1``."""

    name = "gliner"

    def __init__(
        self,
        model_id: str | None = None,
        labels: tuple[str, ...] = LABELS,
        threshold: float = THRESHOLD,
    ) -> None:
        self.model_id = model_id
        self.labels = labels
        self.threshold = threshold
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is None:  # pragma: no cover - requires model weights
            import importlib

            resolved, local_only = resolve_gliner_model(self.model_id)
            gliner = importlib.import_module("gliner")
            model_path = Path(resolved)
            if (
                local_only
                and (model_path / "gliner_config.json").is_file()
                and LOCAL_DEBERTA_DIR.is_dir()
            ):
                import json as _json

                cfg_path = model_path / "gliner_config.json"
                cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
                if cfg.get("model_name") == "microsoft/deberta-v3-base":
                    cfg["model_name"] = str(LOCAL_DEBERTA_DIR)
                    cfg_path.write_text(_json.dumps(cfg, indent=2), encoding="utf-8")
            self._model = gliner.GLiNER.from_pretrained(resolved, local_files_only=local_only)
        return self._model

    def _predict_entities(self, text: str) -> list[dict[str, Any]]:
        model = self._load_model()
        raw = model.predict_entities(text, list(self.labels), threshold=self.threshold)
        return [dict(item) for item in raw]

    def process_message_detailed(
        self, message: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Process one v2 ``Message`` returning ``(v2_record, location_spans)``."""
        text = str(message["message"])
        entities = self._predict_entities(text)

        raw_locs = extract_location_spans(text, entities)
        location_hints = {text[int(s["start"]) : int(s["end"])].casefold() for s in raw_locs}

        phone_spans = extract_phone_spans(text, entities)
        name_spans = extract_name_spans(text, entities, location_hints)
        redactions = _merge_spans([*phone_spans, *name_spans])
        location_spans = extract_location_spans(text, entities, blocked_spans=redactions)
        final_loc_hints = {
            tok.casefold()
            for s in location_spans
            for tok in text[int(s["start"]) : int(s["end"])].split()
        }

        text_redacted = contracts.apply_redactions(text, redactions)
        relevant, v2_activity, regex_event = classify_message_semantics(text)

        location_val: str | None = None
        if location_spans:
            first_loc = location_spans[0]
            raw_loc = text[int(first_loc["start"]) : int(first_loc["end"])]
            location_val = trim_copied_value(raw_loc, text_redacted)

        date_val: str | None = None
        for ent in entities:
            if str(ent.get("label", "")).lower() == "date":
                date_val = trim_copied_value(str(ent.get("text", "")), text_redacted)
                if date_val:
                    break

        event_name_val: str | None = None
        if relevant:
            cand_event = _extract_event_name(text, entities, final_loc_hints) or regex_event
            event_name_val = trim_copied_value(cand_event, text_redacted)

        reported_count = extract_reported_count(text) if relevant else None

        scores = [
            float(ent["score"])
            for ent in entities
            if isinstance(ent.get("score"), (int, float))
        ]
        confidence = round(sum(scores) / len(scores), 4) if scores else (0.75 if relevant else 0.80)
        confidence = max(0.0, min(1.0, confidence))

        record: dict[str, Any] = contracts.build_record(
            message,
            relevant=relevant,
            activity_type=v2_activity,
            location=location_val,
            reported_count=reported_count,
            date_text=date_val,
            event_name=event_name_val,
            event_desc=None,
            redactions=redactions,
            confidence=confidence,
        )
        return record, location_spans

    def process_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process a list of v2 ``Message`` dicts into validated v2 ``Record`` dicts."""
        records: list[dict[str, Any]] = []
        for msg in copy.deepcopy(messages):
            record, _ = self.process_message_detailed(msg)
            records.append(record)
        problems = contracts.validate_file("records", records, messages)
        if problems:
            raise ValueError("GLiNERSystem produced invalid records:\n" + "\n".join(problems))
        return records


def process_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Module-level Interface Spec v2 ``process_messages`` entry point."""
    return GLiNERSystem().process_messages(messages)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GLiNER v2 extraction/redaction system.")
    parser.add_argument("input", nargs="?", default="examples/synthetic_messages.json")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    out_path = (
        args.out
        if args.out is not None
        else REPO_ROOT
        / "examples"
        / (
            "gliner_output.json"
            if input_path.name == "synthetic_messages.json"
            else f"gliner_output_{input_path.stem}.json"
        )
    )
    messages = [m.to_dict() for m in contracts.load_messages(input_path)]
    records = process_messages(messages)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    contracts.write_json(out_path, records)
    print(f"Wrote {len(records)} GLiNER v2 records -> {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
