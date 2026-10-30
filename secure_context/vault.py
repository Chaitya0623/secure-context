"""Split storage: an identity vault and a context store that cannot be joined without the vault.

identity vault (identity.db)            context store (context.db)
  identity(vault_id UUID, name, ...)       client_context(ctx_id UUID, profile, holdings,
  ctx_link(ctx_id UUID -> vault_id)                       notes_storage_tokens, summary)
  storage_tokens(token -> value)
  audit(ts, role, action, ctx_id)

* Both keys are random UUIDs, unrelated to SSN, account number or signup order.
* The only link between the two databases is ctx_link, inside the vault.
* Sensitive vault columns are encrypted with AES-GCM.
* Notes are stored with storage-level tokens. At query time they are re-keyed to
  per-session tokens, so the model never sees the long-lived storage tokens.
* Each object checks the caller's role; the LLM service role has no grant on either store.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .data import Client, Holding
from .detect import resolve_coreference
from .tokens import Session, TOKEN_RE, dehydrate


class AccessDenied(PermissionError):
    pass


VAULT_ROLES = {"rehydrator", "context_service", "ingest"}
CONTEXT_ROLES = {"context_service", "advisor_app", "ingest"}
ENCRYPTED = ("ssn", "account", "routing", "dob")


class IdentityVault:
    def __init__(self, path: str = ":memory:", key: bytes | None = None):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.aead = AESGCM(key or AESGCM.generate_key(bit_length=256))
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS identity(vault_id TEXT PRIMARY KEY, record TEXT, sealed BLOB, nonce BLOB);
        CREATE TABLE IF NOT EXISTS ctx_link(ctx_id TEXT PRIMARY KEY, vault_id TEXT REFERENCES identity);
        CREATE TABLE IF NOT EXISTS storage_tokens(token TEXT PRIMARY KEY, label TEXT, value TEXT);
        CREATE TABLE IF NOT EXISTS audit(ts REAL, role TEXT, action TEXT, ctx_id TEXT);""")

    def _check(self, role, action, ctx_id=None):
        self.db.execute("INSERT INTO audit VALUES (?,?,?,?)", (time.time(), role, action, ctx_id))
        if role not in VAULT_ROLES:
            raise AccessDenied(f"role '{role}' has no grant on the identity vault")

    def put(self, c: Client, role: str = "ingest") -> str:
        self._check(role, "put")
        vault_id = str(uuid.uuid4())
        rec = dict(name=c.name, first=c.first, last=c.last, title=c.title, email=c.email, phone=c.phone,
                   address=c.address, employer=c.employer, spouse=c.spouse, children=c.children)
        secret = json.dumps({k: getattr(c, k) for k in ENCRYPTED}).encode()
        nonce = os.urandom(12)
        self.db.execute("INSERT INTO identity VALUES (?,?,?,?)",
                        (vault_id, json.dumps(rec), self.aead.encrypt(nonce, secret, vault_id.encode()), nonce))
        return vault_id

    def link(self, ctx_id: str, vault_id: str, role: str = "ingest"):
        self._check(role, "link", ctx_id)
        self.db.execute("INSERT INTO ctx_link VALUES (?,?)", (ctx_id, vault_id))

    def identity(self, ctx_id: str, role: str) -> dict:
        self._check(role, "read_identity", ctx_id)
        row = self.db.execute("SELECT i.vault_id, i.record, i.sealed, i.nonce FROM ctx_link l JOIN identity i "
                              "ON l.vault_id = i.vault_id WHERE l.ctx_id = ?", (ctx_id,)).fetchone()
        if not row:
            raise KeyError(ctx_id)
        rec = json.loads(row[1])
        rec.update(json.loads(self.aead.decrypt(row[3], row[2], row[0].encode())))
        return rec

    def save_storage_tokens(self, mapping: dict, role: str = "ingest"):
        self._check(role, "save_tokens")
        self.db.executemany("INSERT OR IGNORE INTO storage_tokens VALUES (?,?,?)",
                            [(t, lab, v) for t, (lab, v) in mapping.items()])

    def storage_token(self, token: str, role: str):
        self._check(role, "read_token")
        return self.db.execute("SELECT label, value FROM storage_tokens WHERE token = ?", (token,)).fetchone()

    def audit_log(self):
        return self.db.execute("SELECT * FROM audit ORDER BY ts").fetchall()


class ContextStore:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS client_context(ctx_id TEXT PRIMARY KEY, profile TEXT, "
                        "holdings TEXT, notes_storage_tokens TEXT, summary TEXT, qi_mentions TEXT)")

    def _check(self, role):
        if role not in CONTEXT_ROLES:
            raise AccessDenied(f"role '{role}' has no grant on the context store")

    def put(self, ctx_id: str, c: Client, notes_tok: str, role: str = "ingest"):
        self._check(role)
        prof = {k: getattr(c, k) for k in ("age", "city", "state", "occupation", "occ_cat", "aum", "household",
                                           "objective", "risk", "horizon", "liquidity", "liquidity_amount")}
        self.db.execute("INSERT INTO client_context VALUES (?,?,?,?,?,?)",
                        (ctx_id, json.dumps(prof), json.dumps([h.__dict__ for h in c.holdings]), notes_tok,
                         c.summary, json.dumps(c.qi_mentions)))

    def get(self, ctx_id: str, role: str) -> dict:
        self._check(role)
        row = self.db.execute("SELECT * FROM client_context WHERE ctx_id = ?", (ctx_id,)).fetchone()
        if not row:
            raise KeyError(ctx_id)
        return dict(ctx_id=row[0], profile=json.loads(row[1]), holdings=[Holding(**h) for h in json.loads(row[2])],
                    notes_storage_tokens=row[3], summary=row[4], qi_mentions=json.loads(row[5]))

    def lookup_any(self, value: str, role: str) -> list:
        """What an attacker holding a dump could try: search every column for a value."""
        self._check(role)
        like = f"%{value}%"
        return self.db.execute("SELECT ctx_id FROM client_context WHERE ctx_id = ? OR profile LIKE ? OR holdings LIKE ? "
                               "OR notes_storage_tokens LIKE ?", (value, like, like, like)).fetchall()

    def columns(self):
        return [r[1] for r in self.db.execute("PRAGMA table_info(client_context)")]


class SplitStorage:
    """Ingests clients and serves session-keyed views of the notes."""

    def __init__(self, detector, vault: IdentityVault | None = None, store: ContextStore | None = None):
        self.detector = detector
        self.vault = vault or IdentityVault()
        self.store = store or ContextStore()
        self._storage_session = Session()

    def ingest(self, c: Client) -> str:
        vault_id = self.vault.put(c)
        ctx_id = str(uuid.uuid4())
        self.vault.link(ctx_id, vault_id)
        spans = resolve_coreference(self.detector.detect(c.notes, c), c)
        before = dict(self._storage_session.vault)
        notes_tok = dehydrate(c.notes, spans, self._storage_session)
        new = {t: v for t, v in self._storage_session.vault.items() if t not in before}
        self.vault.save_storage_tokens(new)
        self.store.put(ctx_id, c, notes_tok)
        return ctx_id

    def session_notes(self, ctx_id: str, session: Session, role: str = "context_service") -> str:
        """Re-key storage tokens to this session's tokens."""
        rec = self.store.get(ctx_id, role)

        def swap(m):
            row = self.vault.storage_token(m.group(0), role)
            return session.token(row[0], row[1]) if row else m.group(0)
        return TOKEN_RE.sub(swap, rec["notes_storage_tokens"])
