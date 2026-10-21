"""Outbound delivery for drift alerts: webhooks and email.

The built-in ``print`` and ``file:`` hooks keep alerts on the machine; these
senders push them outward. Style mirrors sitepulse's notifiers: zero
dependencies, passwords only from the environment, and injectable transports
(``opener``, ``smtp_class``) so tests never touch the network.
"""
from __future__ import annotations

import json
import os
import smtplib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from email.message import EmailMessage

SOURCE = "driftwatch"


class NotifyError(Exception):
    """Delivery failed (bad URL, unreachable host, SMTP error, ...)."""


class SmtpConfigError(NotifyError):
    """SMTP environment configuration is missing or invalid."""


# ---------------------------------------------------------------- webhooks


def build_alert_payload(alerts: list) -> dict:
    """One JSON payload for a batch of alerts (each alert duck-types to
    ``to_dict()``)."""
    return {
        "source": SOURCE,
        "alerts": [a.to_dict() for a in alerts],
    }


def send_webhook(url: str, payload: dict, timeout: float = 10.0,
                 opener=None) -> None:
    """POST one JSON payload; raises NotifyError/OSError on failure."""
    if urllib.parse.urlsplit(url).scheme.lower() not in ("http", "https"):
        raise NotifyError(f"webhook URL must be http(s): {url!r}")
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "User-Agent": f"{SOURCE}/alerts"},
    )
    open_ = opener or urllib.request.urlopen
    try:
        with open_(req, timeout=timeout):
            pass
    except (urllib.error.URLError, OSError) as exc:
        raise NotifyError(f"webhook POST to {url!r} failed: {exc}") from exc


def deliver_webhook(url: str, alerts: list, opener=None) -> None:
    """POST all alerts as a single payload; a no-op when there are no alerts
    (an empty batch is not worth an HTTP call)."""
    if not alerts:
        return
    send_webhook(url, build_alert_payload(alerts), opener=opener)


# -------------------------------------------------------------------- email


@dataclass
class SmtpConfig:
    """SMTP connection settings.

    Built only from environment variables — passwords are never written to
    a config file by this tool.
    """
    host: str
    port: int = 587
    user: str = ""
    password: str = ""  # never logged
    from_addr: str = ""
    use_tls: bool = True
    use_ssl: bool = False
    timeout: float = 10.0

    def __repr__(self) -> str:  # pragma: no cover - never log the password
        return (f"SmtpConfig(host={self.host!r}, port={self.port!r}, "
                f"user={self.user!r}, from_addr={self.from_addr!r}, "
                f"use_tls={self.use_tls!r}, use_ssl={self.use_ssl!r})")


def smtp_config_from_env(env: dict | None = None) -> SmtpConfig:
    """Build an :class:`SmtpConfig` from ``DRIFTWATCH_SMTP_*`` env vars.

    Requires ``DRIFTWATCH_SMTP_HOST`` and ``DRIFTWATCH_SMTP_FROM``; port
    defaults to 587 (or 465 when ``DRIFTWATCH_SMTP_SSL=1`` for implicit TLS).
    Raises :class:`SmtpConfigError` naming the missing variables.
    """
    e = env if env is not None else os.environ
    host = (e.get("DRIFTWATCH_SMTP_HOST") or "").strip()
    from_addr = (e.get("DRIFTWATCH_SMTP_FROM") or "").strip()
    missing = [name for name, val in (("DRIFTWATCH_SMTP_HOST", host),
                                      ("DRIFTWATCH_SMTP_FROM", from_addr))
               if not val]
    if missing:
        raise SmtpConfigError(
            "email alerts need SMTP configuration — missing "
            + ", ".join(missing)
            + " (set DRIFTWATCH_SMTP_HOST and DRIFTWATCH_SMTP_FROM; "
            "add DRIFTWATCH_SMTP_USER / DRIFTWATCH_SMTP_PASS if your server "
            "needs auth)")
    use_ssl = (e.get("DRIFTWATCH_SMTP_SSL") or "").strip() == "1"
    port_raw = (e.get("DRIFTWATCH_SMTP_PORT") or "").strip()
    if port_raw:
        try:
            port = int(port_raw)
        except ValueError:
            raise SmtpConfigError(
                f"invalid DRIFTWATCH_SMTP_PORT: {port_raw!r} (must be a number)")
    else:
        port = 465 if use_ssl else 587
    use_tls = (e.get("DRIFTWATCH_SMTP_TLS") or "").strip().lower() \
        not in ("0", "false", "no")
    return SmtpConfig(
        host=host,
        port=port,
        user=(e.get("DRIFTWATCH_SMTP_USER") or "").strip(),
        password=e.get("DRIFTWATCH_SMTP_PASS") or "",
        from_addr=from_addr,
        use_tls=use_tls and not use_ssl,
        use_ssl=use_ssl,
    )


def email_subject(alert) -> str:
    """Subject line for one drift alert email."""
    return (f"[driftwatch] {alert.feature}: "
            f"{alert.previous_verdict} -> {alert.new_verdict}")


def email_body(alert) -> str:
    """Plain-text summary of one drift alert."""
    def num(v):
        return f"{v:.3f}" if v is not None else "n/a"
    return "\n".join([
        "driftwatch alert: a feature newly escalated in the latest snapshot.",
        "",
        f"feature:   {alert.feature}",
        f"change:    {alert.previous_verdict} -> {alert.new_verdict}",
        f"psi:       {num(alert.psi)}",
        f"ks:        {num(alert.ks)}",
        f"direction: {alert.direction}",
        f"snapshot:  {alert.snapshot_label}",
        "",
        "This fired on a verdict escalation (into drift, or rising into "
        "watch) — not on every run. See the trend report for the full PSI "
        "trajectory.",
    ])


def send_email(recipient: str, alerts: list, env: dict | None = None,
               smtp_class=None, smtp_ssl_class=None) -> None:
    """Send one email per alert.

    Raises NotifyError when the SMTP config is missing/invalid or the send
    fails. ``smtp_class``/``smtp_ssl_class`` are injectable fakes for tests.
    """
    if not alerts:
        return
    config = smtp_config_from_env(env)
    smtp_class = smtp_class or smtplib.SMTP
    smtp_ssl_class = smtp_ssl_class or smtplib.SMTP_SSL
    try:
        client = (smtp_ssl_class(config.host, config.port,
                                 timeout=config.timeout)
                  if config.use_ssl
                  else smtp_class(config.host, config.port,
                                  timeout=config.timeout))
    except (OSError, smtplib.SMTPException) as exc:
        raise NotifyError(
            f"SMTP connect to {config.host}:{config.port} failed: {exc}"
        ) from exc
    try:
        client.ehlo_or_helo_if_needed()
        if config.use_tls and not config.use_ssl:
            client.starttls()
            client.ehlo_or_helo_if_needed()
        if config.user:
            client.login(config.user, config.password)
        for alert in alerts:
            msg = EmailMessage()
            msg["Subject"] = email_subject(alert)
            msg["From"] = config.from_addr
            msg["To"] = recipient
            msg.set_content(email_body(alert))
            client.send_message(msg)
    except (OSError, smtplib.SMTPException) as exc:
        raise NotifyError(
            f"sending alert email to {recipient!r} failed: {exc}") from exc
    finally:
        try:
            client.quit()
        except Exception:  # never let cleanup mask the real error
            pass
