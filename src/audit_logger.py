import os
import json
import uuid
import base64
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))


class AuditLogger:
    """
    Daily-rotated, append-only AES-GCM encrypted audit logs with a tamper-evident hash chain.

    Files:
      logs/audit-YYYY-MM-DD.jsonl.enc

    Encryption:
      Each line is encrypted with AES-GCM.
      Stored as Base64(nonce || ciphertext_with_tag)

    Hash Chain:
      Chain is computed over canonical JSON of the *event* + prev_hash (per day file).
      Verification decrypts lines and checks hashes.

    Retention:
      Deletes encrypted log files older than retention_days.
    """

    def __init__(self, log_dir: str = "logs", retention_days: int = 30, key_b64_env: str = "AUDIT_AES_KEY_B64"):
        self.log_dir = log_dir
        self.retention_days = retention_days
        os.makedirs(self.log_dir, exist_ok=True)

        key_b64 = os.getenv(key_b64_env, "")
        if not key_b64:
            raise RuntimeError(
                f"Missing AES key. Set env var {key_b64_env} to a Base64-encoded 32-byte key."
            )

        key = _b64d(key_b64)
        if len(key) != 32:
            raise RuntimeError(f"{key_b64_env} must decode to 32 bytes (got {len(key)}).")

        self.aesgcm = AESGCM(key)
        self._cleanup_old_logs()

    def _log_path_for_day(self, day: str) -> str:
        return os.path.join(self.log_dir, f"audit-{day}.jsonl.enc")

    def _log_path_for_today(self) -> str:
        return self._log_path_for_day(_today_utc())

    def _encrypt_line(self, plaintext: str) -> str:
        nonce = os.urandom(12)  # AES-GCM standard nonce size
        ct = self.aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return _b64e(nonce + ct)

    def _decrypt_line(self, line_b64: str) -> str:
        raw = _b64d(line_b64.strip())
        if len(raw) < 13:
            raise ValueError("Encrypted line too short")
        nonce = raw[:12]
        ct = raw[12:]
        pt = self.aesgcm.decrypt(nonce, ct, None)
        return pt.decode("utf-8")

    def _read_last_hash(self, path: str) -> str:
        """
        Read the last encrypted line, decrypt it, and return its event_hash.
        If empty/missing, return genesis hash (per day).
        """
        try:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return "0" * 64

            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                offset = min(8192, f.tell())
                f.seek(-offset, os.SEEK_END)
                lines = f.read().splitlines()
                last_b64 = lines[-1].decode("utf-8")
                last_json = self._decrypt_line(last_b64)
                obj = json.loads(last_json)
                return obj.get("event_hash", "0" * 64)
        except Exception:
            return "0" * 64

    def _cleanup_old_logs(self):
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        for name in os.listdir(self.log_dir):
            if not name.startswith("audit-") or not name.endswith(".jsonl.enc"):
                continue
            try:
                date_part = name[len("audit-"):-len(".jsonl.enc")]
                dt = datetime.strptime(date_part, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if dt < cutoff:
                    os.remove(os.path.join(self.log_dir, name))
            except Exception:
                continue

    def log_event(
        self,
        event_type: str,
        actor: Dict[str, Any],
        request: Dict[str, Any],
        decision: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        day = _today_utc()
        path = self._log_path_for_day(day)
        prev_hash = self._read_last_hash(path)

        event = {
            "v": 1,
            "ts": _utc_iso(),
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "actor": actor,
            "request": request,
            "decision": decision,
            "meta": meta or {},
        }

        canonical = json.dumps(event, sort_keys=True, separators=(",", ":"))
        event_hash = _sha256_hex(prev_hash + canonical)

        record = {
            "prev_hash": prev_hash,
            "event_hash": event_hash,
            "event": event,
        }

        line_plain = json.dumps(record, sort_keys=True, separators=(",", ":"))
        line_enc = self._encrypt_line(line_plain)

        with open(path, "a", encoding="utf-8") as f:
            f.write(line_enc + "\n")

        return record

    def verify_day(self, day: str) -> Dict[str, Any]:
        """
        Verify the hash chain for a specific day (YYYY-MM-DD) by decrypting each line.
        """
        path = self._log_path_for_day(day)
        if not os.path.exists(path):
            return {"ok": False, "error": "file_not_found", "day": day}

        prev = "0" * 64
        line_no = 0

        try:
            with open(path, "r", encoding="utf-8") as f:
                for enc_line in f:
                    enc_line = enc_line.strip()
                    if not enc_line:
                        continue

                    line_no += 1
                    plain = self._decrypt_line(enc_line)
                    obj = json.loads(plain)

                    prev_hash = obj.get("prev_hash")
                    event_hash = obj.get("event_hash")
                    event = obj.get("event")

                    canonical = json.dumps(event, sort_keys=True, separators=(",", ":"))
                    expected = _sha256_hex(prev + canonical)

                    if prev_hash != prev or event_hash != expected:
                        return {
                            "ok": False,
                            "day": day,
                            "lines": line_no,
                            "first_bad_line": line_no,
                        }

                    prev = event_hash

            return {"ok": True, "day": day, "lines": line_no, "first_bad_line": None}
        except Exception as e:
            return {"ok": False, "day": day, "error": "decrypt_or_parse_failed", "detail": str(e), "lines": line_no}
