"""Ed25519 signing keys for alerts (Stage 13).

The private key lives in one PEM file (ALERT_SIGNING_KEY_PATH, default secrets/alert_key.pem,
never committed) on the machine that runs the alert engine. The public API never opens it: it
serves the public keys from the alert_keys table so anyone can check a signature.

What is signed: the alert as JSON without its "signature" field, keys sorted, no spaces, UTF-8
(`canonical`). A receiver re-serialises the JSON it got the same way and checks the Ed25519
signature (base64) with the public key named by key_id.

python -m alerts.keys --new          make a key (refuses to overwrite) and register its public half
python -m alerts.keys --show         key_id and public key of the current file
"""

import argparse
import base64
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from api.config import ROOT

DEFAULT_PATH = ROOT / "secrets" / "alert_key.pem"


def key_path() -> Path:
    value = os.getenv("ALERT_SIGNING_KEY_PATH")
    if not value:
        return DEFAULT_PATH
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def key_id(public_raw: bytes) -> str:
    """A short, stable name for a public key: hzd- and the first 8 bytes of its SHA-256."""
    return "hzd-" + hashlib.sha256(public_raw).hexdigest()[:16]


def public_raw(private: Ed25519PrivateKey) -> bytes:
    return private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )


def generate(path: Path | None = None) -> Ed25519PrivateKey:
    path = path or key_path()
    if path.exists():
        raise FileExistsError(f"{path} exists; retire it by hand before making a new key")
    private = Ed25519PrivateKey.generate()
    pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(pem)
    return private


def load(path: Path | None = None) -> Ed25519PrivateKey:
    path = path or key_path()
    if not path.exists():
        raise FileNotFoundError(f"no signing key at {path}. Run: python -m alerts.keys --new")
    private = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(private, Ed25519PrivateKey):
        raise ValueError(f"{path} is not an Ed25519 key")
    return private


def canonical(body: dict) -> bytes:
    """The bytes that are signed: the JSON body without its signature, sorted, compact."""
    unsigned = {k: v for k, v in body.items() if k != "signature"}
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sign(private: Ed25519PrivateKey, body: dict) -> str:
    return base64.b64encode(private.sign(canonical(body))).decode()


def verify(public_b64: str, body: dict) -> bool:
    """True when body["signature"] is a valid signature of the rest of body."""
    try:
        public = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_b64))
        public.verify(base64.b64decode(body.get("signature") or ""), canonical(body))
    except (InvalidSignature, ValueError):
        return False
    return True


def register(private: Ed25519PrivateKey) -> str:
    """Store the public half in alert_keys (once) and return its key_id."""
    from api.db import AlertKey, SessionLocal, init_db

    raw = public_raw(private)
    kid = key_id(raw)
    init_db()
    with SessionLocal.begin() as session:
        if session.get(AlertKey, kid) is None:
            session.add(
                AlertKey(
                    key_id=kid,
                    public_key=base64.b64encode(raw).decode(),
                    created_at=datetime.now(UTC),
                )
            )
    return kid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alert signing keys.")
    parser.add_argument("--new", action="store_true", help="make a key and register it")
    parser.add_argument("--show", action="store_true", help="print the current key's id")
    args = parser.parse_args(argv)
    if not (args.new or args.show):
        parser.error("give --new or --show")
    private = generate() if args.new else load()
    raw = public_raw(private)
    if args.new:
        register(private)
        print(f"Wrote {key_path()} (keep it off the public API machine; never commit it)")
    print(f"key_id {key_id(raw)}  public {base64.b64encode(raw).decode()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
