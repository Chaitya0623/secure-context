"""Session-keyed reversible pseudonymization (dehydrate and rehydrate).

token = "<" + LABEL + "_" + HMAC-SHA256(session_key, LABEL|canonical)[:6] + ">"

* Same value, same session: same token, so the model can still tell people apart.
* New session: new key, every token changes, so logs from two sessions cannot
  be linked by token.
* Without the key a token cannot be reversed or brute-forced, unlike a plain
  hash of an SSN.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import uuid
from dataclasses import dataclass, field

TOKEN_RE = re.compile(r"<([A-Z]+)_([0-9a-f]{6})>")
LOOSE_TOKEN_RE = re.compile(r"<\s*([A-Za-z]+)[ _-]?([0-9a-fA-F]{3,6})\s*>|\b([A-Z]{3,8})_([0-9a-f]{4,6})\b")


@dataclass
class Session:
    key: bytes = field(default_factory=lambda: os.urandom(32))
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    vault: dict = field(default_factory=dict)  # token -> (label, canonical)

    def token(self, label: str, canonical: str) -> str:
        mac = hmac.new(self.key, f"{label}|{canonical}".encode(), hashlib.sha256).hexdigest()[:6]
        t = f"<{label}_{mac}>"
        self.vault[t] = (label, canonical)
        return t

    def fingerprint(self) -> str:
        return hashlib.sha256(self.key).hexdigest()[:8]


def dehydrate(text: str, spans, session: Session) -> str:
    out, pos = [], 0
    for s in sorted(spans, key=lambda s: s.start):
        if s.start < pos:
            continue
        out.append(text[pos:s.start])
        out.append(session.token(s.label, s.canonical))
        pos = s.end
    out.append(text[pos:])
    return "".join(out)


@dataclass
class Rehydrated:
    text: str
    resolved: int
    repaired: list
    unresolved: list


def rehydrate(text: str, session: Session, repair: bool = True) -> Rehydrated:
    """Swap tokens back. Malformed tokens are repaired only when the match is unique."""
    resolved = 0
    repaired, unresolved = [], []

    def one(m):
        nonlocal resolved
        raw = m.group(0)
        if raw in session.vault:
            resolved += 1
            return session.vault[raw][1]
        if repair:
            label = (m.group(1) or m.group(3) or "").upper()
            frag = (m.group(2) or m.group(4) or "").lower()
            cands = [t for t, (lab, _) in session.vault.items()
                     if lab == label and t[len(label) + 2:-1].startswith(frag)]
            if len(cands) == 1:
                repaired.append((raw, cands[0]))
                return session.vault[cands[0]][1]
        unresolved.append(raw)
        return f"[UNRESOLVED {raw}]"

    text = LOOSE_TOKEN_RE.sub(one, text)
    return Rehydrated(text, resolved, repaired, unresolved)
