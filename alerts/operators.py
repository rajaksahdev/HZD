"""Operators and their two-factor login (Stage 13).

A red alert, and anything that lowers or ends one, is sent only after an operator approves it
with both factors: a password (stored as an scrypt hash) and a six-digit code from an
authenticator app (TOTP, RFC 6238: SHA-1, 30 s steps, one step of clock drift allowed). A code
works once. Five failures lock the operator out for 15 minutes.

python -m alerts.operators --add NAME      asks for a password; prints the authenticator secret
python -m alerts.operators --list
"""

import argparse
import base64
import getpass
import hashlib
import hmac
import secrets
import struct
import sys
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.db import Operator, SessionLocal, init_db

STEP = 30  # seconds per TOTP code
DIGITS = 6
DRIFT = 1  # codes from one step before or after are accepted
MAX_FAILURES = 5
LOCKOUT = timedelta(minutes=15)
MIN_PASSWORD = 10
SCRYPT = {"n": 2**14, "r": 8, "p": 1}
ISSUER = "HazardLens"


class LoginError(Exception):
    """Wrong name, password or code, or the operator is locked or inactive. The message never
    says which factor failed."""


# --- Passwords ------------------------------------------------------------------------------


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **SCRYPT)
    params = f"{SCRYPT['n']}${SCRYPT['r']}${SCRYPT['p']}"
    return f"scrypt${params}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def check_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        got = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p), dklen=32
        )
    except ValueError:
        return False
    return hmac.compare_digest(got, base64.b64decode(digest))


# --- TOTP -----------------------------------------------------------------------------------


def new_secret() -> str:
    return base64.b32encode(secrets.token_bytes(20)).decode()


def code_at(secret: str, step: int) -> str:
    key = base64.b32decode(secret.upper() + "=" * (-len(secret) % 8))
    mac = hmac.new(key, struct.pack(">Q", step), hashlib.sha1).digest()
    offset = mac[-1] & 0x0F
    number = struct.unpack(">I", mac[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(number % 10**DIGITS).zfill(DIGITS)


def step_of(when: datetime) -> int:
    return int(when.timestamp()) // STEP


def matching_step(secret: str, code: str, when: datetime, after: int = 0) -> int | None:
    """The time step the code belongs to, if it is valid now and newer than `after`."""
    code = code.strip().replace(" ", "")
    now = step_of(when)
    for step in range(now - DRIFT, now + DRIFT + 1):
        if step > after and hmac.compare_digest(code_at(secret, step), code):
            return step
    return None


def provisioning_uri(name: str, secret: str) -> str:
    """Scan this (as a QR code) or type the secret into any authenticator app."""
    return f"otpauth://totp/{ISSUER}:{name}?secret={secret}&issuer={ISSUER}&digits={DIGITS}"


# --- Operators ------------------------------------------------------------------------------


def add(session: Session, name: str, password: str, now: datetime | None = None) -> str:
    """Create an operator and return their TOTP secret (shown once)."""
    if len(password) < MIN_PASSWORD:
        raise ValueError(f"password must have at least {MIN_PASSWORD} characters")
    if session.get(Operator, name) is not None:
        raise ValueError(f"operator {name} exists")
    secret = new_secret()
    session.add(
        Operator(
            name=name,
            password_hash=hash_password(password),
            totp_secret=secret,
            last_step=0,
            failures=0,
            active=True,
            created_at=now or datetime.now(UTC),
        )
    )
    return secret


def login(
    session: Session, name: str, password: str, code: str, now: datetime | None = None
) -> Operator:
    """Check both factors. The caller's transaction must commit even when this raises, so that
    failures are counted."""
    now = now or datetime.now(UTC)
    op = session.get(Operator, name, with_for_update=True)
    if op is None or not op.active:
        raise LoginError("login failed")
    if op.locked_until is not None and op.locked_until > now:
        raise LoginError(f"locked until {op.locked_until:%H:%M} UTC after repeated failures")
    step = matching_step(op.totp_secret, code, now, after=op.last_step)
    if not check_password(password, op.password_hash) or step is None:
        op.failures += 1
        if op.failures >= MAX_FAILURES:
            op.locked_until, op.failures = now + LOCKOUT, 0
        raise LoginError("login failed")
    op.failures, op.locked_until, op.last_step = 0, None, step
    return op


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Operators who approve alerts.")
    parser.add_argument("--add", metavar="NAME")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    init_db()
    if args.add:
        password = getpass.getpass("Password: ")
        if getpass.getpass("Again: ") != password:
            print("The passwords differ.")
            return 1
        with SessionLocal.begin() as session:
            secret = add(session, args.add, password)
        print(f"Operator {args.add} added. Put this in an authenticator app (shown once):")
        print(f"  secret {secret}\n  {provisioning_uri(args.add, secret)}")
    if args.list:
        with SessionLocal() as session:
            for op in session.scalars(select(Operator).order_by(Operator.name)):
                state = "active" if op.active else "inactive"
                print(f"{op.name}  {state}  added {op.created_at:%Y-%m-%d}")
    if not (args.add or args.list):
        parser.error("give --add NAME or --list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
