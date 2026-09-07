import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from cryptography.fernet import Fernet, InvalidToken

from app.config import settings


SETTINGS_ID = "active"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path() -> Path:
    return Path(settings.LLM_SETTINGS_DB_PATH)


def _key_path() -> Path:
    return _db_path().with_name("llm_settings.key")


def _load_or_create_encryption_key() -> bytes:
    configured = (settings.LLM_SETTINGS_ENCRYPTION_KEY or "").strip()
    if configured:
        return configured.encode("utf-8")

    key_file = _key_path()
    if key_file.exists():
        return key_file.read_bytes().strip()

    key_file.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    key_file.write_bytes(key)
    return key


def _fernet() -> Fernet:
    try:
        return Fernet(_load_or_create_encryption_key())
    except ValueError as exc:
        raise RuntimeError(
            "LLM_SETTINGS_ENCRYPTION_KEY must be a valid Fernet key. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        ) from exc


def init_llm_settings_store() -> None:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_settings (
                id TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                encrypted_api_key TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(llm_settings)").fetchall()}
        if "label" not in columns:
            conn.execute("ALTER TABLE llm_settings ADD COLUMN label TEXT")
        if "is_active" not in columns:
            conn.execute("ALTER TABLE llm_settings ADD COLUMN is_active INTEGER NOT NULL DEFAULT 0")
        conn.execute(
            """
            UPDATE llm_settings
            SET is_active = 1
            WHERE id = ?
              AND NOT EXISTS (SELECT 1 FROM llm_settings WHERE is_active = 1)
            """,
            (SETTINGS_ID,),
        )
        conn.commit()


def _connect() -> sqlite3.Connection:
    init_llm_settings_store()
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def encrypt_api_key(api_key: str) -> str:
    return _fernet().encrypt(api_key.encode("utf-8")).decode("utf-8")


def decrypt_api_key(encrypted_api_key: Optional[str]) -> str:
    if not encrypted_api_key:
        return ""
    try:
        return _fernet().decrypt(encrypted_api_key.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise RuntimeError("Stored LLM API key cannot be decrypted with the current encryption key.") from exc


def _row_to_settings(row: sqlite3.Row, include_api_key: bool = False) -> dict:
    result = {
        "id": row["id"],
        "provider": row["provider"],
        "model": row["model"],
        "label": row["label"] or "",
        "has_api_key": bool(row["encrypted_api_key"]),
        "is_active": bool(row["is_active"]),
        "updated_at": row["updated_at"],
    }
    if include_api_key:
        result["api_key"] = decrypt_api_key(row["encrypted_api_key"])
    return result


def list_llm_settings(include_api_key: bool = False) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, provider, model, label, encrypted_api_key, is_active, updated_at
            FROM llm_settings
            ORDER BY is_active DESC, updated_at DESC
            """
        ).fetchall()
    return [_row_to_settings(row, include_api_key=include_api_key) for row in rows]


def get_llm_settings(include_api_key: bool = False) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, provider, model, label, encrypted_api_key, is_active, updated_at
            FROM llm_settings
            ORDER BY is_active DESC, updated_at DESC
            LIMIT 1
            """,
        ).fetchone()

    if not row:
        return None

    return _row_to_settings(row, include_api_key=include_api_key)


def get_llm_setting_by_id(setting_id: str, include_api_key: bool = False) -> Optional[dict]:
    if not setting_id:
        return None
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, provider, model, label, encrypted_api_key, is_active, updated_at
            FROM llm_settings
            WHERE id = ?
            """,
            (setting_id,),
        ).fetchone()
    return _row_to_settings(row, include_api_key=include_api_key) if row else None


def _find_llm_setting(provider: str, model: str, include_api_key: bool = False) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, provider, model, label, encrypted_api_key, is_active, updated_at
            FROM llm_settings
            WHERE provider = ? AND model = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (provider, model),
        ).fetchone()
    return _row_to_settings(row, include_api_key=include_api_key) if row else None


def save_llm_settings(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    setting_id: Optional[str] = None,
    label: Optional[str] = None,
) -> dict:
    provider = (provider or "").strip().lower()
    model = (model or "").strip()
    current = (
        get_llm_setting_by_id((setting_id or "").strip(), include_api_key=True)
        or _find_llm_setting(provider, model, include_api_key=True)
        or {}
    )
    provider = (provider or current.get("provider") or "gemini").strip().lower()
    model = (model or current.get("model") or "").strip()
    api_key = "" if api_key is None else api_key.strip()

    if not model:
        raise ValueError("Model is required.")

    if provider == "ollama":
        encrypted_api_key = ""
    elif api_key:
        encrypted_api_key = encrypt_api_key(api_key)
    elif current.get("api_key"):
        encrypted_api_key = encrypt_api_key(current["api_key"])
    else:
        raise ValueError("API key is required for the selected provider.")

    row_id = current.get("id") or uuid4().hex
    label = (label or current.get("label") or "").strip()
    now = _utc_now()
    with _connect() as conn:
        conn.execute("UPDATE llm_settings SET is_active = 0")
        conn.execute(
            """
            INSERT INTO llm_settings (id, provider, model, encrypted_api_key, label, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                provider = excluded.provider,
                model = excluded.model,
                encrypted_api_key = excluded.encrypted_api_key,
                label = excluded.label,
                is_active = 1,
                updated_at = excluded.updated_at
            """,
            (row_id, provider, model, encrypted_api_key, label, now, now),
        )
        conn.commit()

    return get_llm_settings(include_api_key=False) or {}


def delete_llm_settings() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM llm_settings WHERE is_active = 1")
        conn.commit()


def resolve_llm_settings(overrides: Optional[dict] = None) -> dict:
    overrides = overrides or {}
    saved = get_llm_setting_by_id(str(overrides.get("id") or "").strip(), include_api_key=True)
    if not saved:
        saved = get_llm_settings(include_api_key=True) or {}

    provider = (overrides.get("provider") or saved.get("provider") or "gemini").strip().lower()
    model = (overrides.get("model") or saved.get("model") or "").strip()
    api_key = (overrides.get("api_key") or saved.get("api_key") or "").strip()
    return {"provider": provider, "model": model, "api_key": api_key}
