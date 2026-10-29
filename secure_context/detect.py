"""PII detection for advisor notes.

Detectors share one interface: detect(text, client=None) -> list[Span].

RegexDetector       structured identifiers (SSN, account, routing, email, phone, DOB, address)
CRMDictionaryDetector  exact matches of values the CRM already holds for this client
                       (names, aliases, spouse, children, employer, address)
PresidioDetector    Microsoft Presidio with a spaCy model (optional dependency)
GlinerDetector      GLiNER family, e.g. nvidia/gliner-PII or GLiNER2-PII (optional dependency)
EnsembleDetector    union of several detectors with overlap resolution

resolve_coreference() maps every detected mention to a canonical value from
the client record (for example "Mr. Okafor" and "Daniel" to "Daniel Okafor",
"6175550143" to "(617) 555-0143") so one value always gets one token.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable, Optional

from .data import Span, Client

LABELS = ["PERSON", "SSN", "ACCOUNT", "ROUTING", "EMAIL", "PHONE", "DOB", "ADDRESS", "ORG"]

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


@dataclass
class Detected(Span):
    detector: str = ""


class RegexDetector:
    name = "regex"
    PATTERNS = [
        ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
        ("SSN", re.compile(r"(?i)\bssn\D{0,25}?(\d{9}|\d{4})\b")),
        ("ACCOUNT", re.compile(r"\bACCT \d{4}-\d{4}-\d{2}\b")),
        ("ACCOUNT", re.compile(r"(?i)\baccount (?:number |no\.? |#)?(\d[\d-]{7,})\b")),
        ("ROUTING", re.compile(r"(?i)\brouting\D{0,12}(\d{9})\b")),
        ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[A-Za-z]{2,})+")),
        ("PHONE", re.compile(r"\(\d{3}\)\s?\d{3}-\d{4}|\b\d{3}[-.]\d{3}[-.]\d{4}\b")),
        ("DOB", re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")),
        ("DOB", re.compile(rf"\b(?:{MONTHS}) \d{{1,2}}, \d{{4}}\b")),
        ("ADDRESS", re.compile(r"\b\d{1,6}(?: [A-Z][A-Za-z.]+)+,(?: [A-Z][A-Za-z.]+)+, [A-Z]{2} \d{5}\b")),
        ("ACCOUNT", re.compile(r"\b(?:\d{4}[ -]?){3}\d{1,4}\b")),                  # card-like numbers
        ("ACCOUNT", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")),  # IBAN
    ]

    def detect(self, text: str, client: Optional[Client] = None) -> list[Detected]:
        out = []
        for label, pat in self.PATTERNS:
            for m in pat.finditer(text):
                s, e = (m.start(1), m.end(1)) if m.groups() else (m.start(), m.end())
                out.append(Detected(s, e, label, text[s:e], text[s:e], detector=self.name))
        return out


class CRMDictionaryDetector:
    """Uses what the protected environment already knows about this client.

    This is the strongest detector for names because the CRM record holds the
    exact spellings. It cannot catch values that are not in the record.
    """
    name = "crm_dictionary"

    @staticmethod
    def terms(c: Client):
        t = [(c.name, "PERSON", c.name), (f"{c.title} {c.last}", "PERSON", c.name), (c.first, "PERSON", c.name),
             (c.employer, "ORG", c.employer), (c.address, "ADDRESS", c.address), (c.email, "EMAIL", c.email),
             (c.phone, "PHONE", c.phone)]
        if c.spouse:
            t += [(c.spouse, "PERSON", c.spouse), (c.spouse.split()[0], "PERSON", c.spouse)]
        for k in c.children:
            t += [(k, "PERSON", k), (k.split()[0], "PERSON", k)]
        return sorted(t, key=lambda x: -len(x[0]))

    def detect(self, text: str, client: Optional[Client] = None) -> list[Detected]:
        if client is None:
            return []
        out = []
        for term, label, canon in self.terms(client):
            for m in re.finditer(r"(?<![\w])" + re.escape(term) + r"(?![\w])", text):
                out.append(Detected(m.start(), m.end(), label, term, canon, detector=self.name))
        return out


class PresidioDetector:
    name = "presidio"
    MAP = {"PERSON": "PERSON", "US_SSN": "SSN", "US_BANK_NUMBER": "ACCOUNT", "EMAIL_ADDRESS": "EMAIL",
           "PHONE_NUMBER": "PHONE", "DATE_TIME": "DOB", "LOCATION": "ADDRESS", "US_DRIVER_LICENSE": "ACCOUNT",
           "NRP": None, "URL": None}

    def __init__(self, spacy_model: str = "en_core_web_sm", threshold: float = 0.35):
        import logging
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        # tldextract tries to refresh its suffix list online and falls back to a bundled copy offline
        logging.getLogger("tldextract").setLevel(logging.CRITICAL)
        logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": spacy_model}]})
        self.engine = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
        self.threshold = threshold

    def detect(self, text: str, client: Optional[Client] = None) -> list[Detected]:
        out = []
        for r in self.engine.analyze(text=text, language="en", score_threshold=self.threshold):
            label = self.MAP.get(r.entity_type)
            if label:
                out.append(Detected(r.start, r.end, label, text[r.start:r.end], text[r.start:r.end], detector=self.name))
        return out


class GlinerDetector:
    """GLiNER-family model. Needs `pip install gliner` and access to the Hugging Face Hub."""
    name = "gliner"
    LABELS = {"person": "PERSON", "social security number": "SSN", "bank account number": "ACCOUNT",
              "routing number": "ROUTING", "email": "EMAIL", "phone number": "PHONE", "date of birth": "DOB",
              "street address": "ADDRESS", "organization": "ORG"}

    def __init__(self, model_id: str = "nvidia/gliner-PII", threshold: float = 0.4):
        from gliner import GLiNER
        self.model = GLiNER.from_pretrained(model_id)
        self.threshold = threshold
        self.name = f"gliner:{model_id}"

    def detect(self, text: str, client: Optional[Client] = None) -> list[Detected]:
        ents = self.model.predict_entities(text, list(self.LABELS), threshold=self.threshold)
        return [Detected(e["start"], e["end"], self.LABELS[e["label"]], e["text"], e["text"], detector=self.name)
                for e in ents]


def resolve_overlaps(spans: Iterable[Detected]) -> list[Detected]:
    """Keep the longest span where detections overlap; ties prefer structured-ID detectors."""
    prio = {"regex": 0, "crm_dictionary": 1}
    ordered = sorted(spans, key=lambda s: (-(s.end - s.start), prio.get(s.detector, 2), s.start))
    kept: list[Detected] = []
    for s in ordered:
        if not any(s.start < k.end and s.end > k.start for k in kept):
            kept.append(s)
    return sorted(kept, key=lambda s: s.start)


class EnsembleDetector:
    def __init__(self, detectors):
        self.detectors = list(detectors)
        self.name = "+".join(d.name for d in self.detectors)

    def detect(self, text: str, client: Optional[Client] = None) -> list[Detected]:
        allspans = []
        for d in self.detectors:
            allspans += d.detect(text, client)
        return resolve_overlaps(allspans)


def _digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


def resolve_coreference(spans: list[Detected], c: Optional[Client]) -> list[Detected]:
    """Map each mention to a canonical value from the CRM record when one matches."""
    if c is None:
        return spans
    people = {c.name: c.name, c.first: c.name, f"{c.title} {c.last}": c.name, c.last: c.name}
    if c.spouse:
        people[c.spouse] = c.spouse
        people[c.spouse.split()[0]] = c.spouse
    for k in c.children:
        people[k] = k
        people[k.split()[0]] = k
    out = []
    for s in spans:
        canon = s.canonical
        d = _digits(s.text)
        if s.label == "SSN" and d and _digits(c.ssn).endswith(d):
            canon = c.ssn
        elif s.label == "PHONE" and d == _digits(c.phone):
            canon = c.phone
        elif s.label == "ACCOUNT" and d and d in _digits(c.account):
            canon = c.account
        elif s.label == "ROUTING" and d == c.routing:
            canon = c.routing
        elif s.label == "EMAIL" and s.text.lower() == c.email.lower():
            canon = c.email
        elif s.label == "PERSON" and s.text in people:
            canon = people[s.text]
        elif s.label == "DOB":
            canon = c.dob if _normalize_date(s.text) == c.dob else s.text
        out.append(replace(s, canonical=canon))
    return out


def _normalize_date(t: str) -> str:
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})$", t)
    if m:
        return f"{m[3]}-{int(m[1]):02d}-{int(m[2]):02d}"
    m = re.match(rf"({MONTHS}) (\d{{1,2}}), (\d{{4}})$", t)
    if m:
        mi = MONTHS.split("|").index(m[1]) + 1
        return f"{m[3]}-{mi:02d}-{int(m[2]):02d}"
    return t


def default_detector(use_presidio: bool = False, gliner_model: Optional[str] = None):
    dets = [RegexDetector(), CRMDictionaryDetector()]
    if use_presidio:
        dets.append(PresidioDetector())
    if gliner_model:
        dets.append(GlinerDetector(gliner_model))
    return EnsembleDetector(dets)


# Scoring against ground truth

def score_detection(gold: list[Span], pred: list[Span], text_len: int, cover: float = 0.9):
    """Leak-oriented scoring.

    A gold span counts as caught when at least `cover` of its characters are
    inside some predicted span (any label). A predicted span is a true positive
    when it overlaps any gold span.
    """
    mask = bytearray(text_len)
    for p in pred:
        for i in range(p.start, p.end):
            mask[i] = 1
    caught = []
    for g in gold:
        n = g.end - g.start
        hit = sum(mask[g.start:g.end])
        caught.append(hit / max(n, 1) >= cover)
    tp_pred = sum(1 for p in pred if any(p.start < g.end and p.end > g.start for g in gold))
    return caught, tp_pred, len(pred)
