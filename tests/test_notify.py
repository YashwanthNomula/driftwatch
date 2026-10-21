"""Tests for webhook/email alert delivery (driftwatch/notify.py + hook kinds)."""
import json
import os
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from driftwatch.alerts import Alert, dispatch_alerts, parse_alert_hook
from driftwatch import notify
from driftwatch.notify import (NotifyError, SmtpConfigError, build_alert_payload,
                               deliver_webhook, send_email, send_webhook,
                               smtp_config_from_env)


def make_alert(feature="age"):
    return Alert(feature=feature, previous_verdict="moderate shift — watch",
                 new_verdict="significant drift", psi=0.67, ks=0.33,
                 direction="rising", snapshot_index=2, snapshot_label="week3")


# ------------------------------------------------------------ hook parsing


def test_parse_webhook_hook():
    h = parse_alert_hook("webhook:https://hooks.example.com/drift")
    assert (h.kind, h.path) == ("webhook", "https://hooks.example.com/drift")


def test_parse_webhook_rejects_non_http():
    with pytest.raises(ValueError, match="must start with http"):
        parse_alert_hook("webhook:ftp://x/y")
    with pytest.raises(ValueError, match="must start with http"):
        parse_alert_hook("webhook:")


def test_parse_email_hook():
    h = parse_alert_hook("email:ops@example.com")
    assert (h.kind, h.path) == ("email", "ops@example.com")


def test_parse_email_rejects_bad_address():
    for bad in ("email:not-an-email", "email:a@b", "email:@x.com", "email:"):
        with pytest.raises(ValueError, match="not a valid email address"):
            parse_alert_hook(bad)


def test_parse_still_rejects_unknown():
    with pytest.raises(ValueError, match="expected 'print', 'file:PATH'"):
        parse_alert_hook("slack:#alerts")


# ------------------------------------------------------- webhook delivery


class FakeOpener:
    """Captures the request; returns a dummy context manager."""
    def __init__(self, fail=None):
        self.requests = []
        self.fail = fail

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        if self.fail:
            raise self.fail
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_send_webhook_posts_json():
    opener = FakeOpener()
    payload = build_alert_payload([make_alert()])
    assert payload["source"] == "driftwatch"
    assert payload["alerts"][0]["feature"] == "age"
    send_webhook("https://hooks.example.com/x", payload, opener=opener)
    (req,) = opener.requests
    assert req.get_method() == "POST"
    assert req.full_url == "https://hooks.example.com/x"
    assert req.headers["Content-type"] == "application/json"
    body = json.loads(req.data.decode())
    assert body["alerts"][0]["new_verdict"] == "significant drift"


def test_send_webhook_wraps_network_failure():
    opener = FakeOpener(fail=urllib.error.URLError("boom"))
    with pytest.raises(NotifyError, match="webhook POST"):
        send_webhook("https://x.example/", {}, opener=opener)


def test_send_webhook_rejects_non_http():
    with pytest.raises(NotifyError, match="must be http"):
        send_webhook("ftp://x/y", {})


def test_deliver_webhook_noop_on_empty():
    opener = FakeOpener()
    deliver_webhook("https://x.example/", [], opener=opener)
    assert opener.requests == []


# ---------------------------------------------------------- email delivery


def test_smtp_config_from_env_ok():
    env = {"DRIFTWATCH_SMTP_HOST": "smtp.example.com",
           "DRIFTWATCH_SMTP_FROM": "drift@example.com",
           "DRIFTWATCH_SMTP_USER": "bot",
           "DRIFTWATCH_SMTP_PASS": "s3cret"}
    cfg = smtp_config_from_env(env)
    assert (cfg.host, cfg.port, cfg.user, cfg.password,
            cfg.from_addr) == ("smtp.example.com", 587, "bot", "s3cret",
                               "drift@example.com")
    assert cfg.use_tls and not cfg.use_ssl


def test_smtp_config_missing_vars_named():
    with pytest.raises(SmtpConfigError, match="DRIFTWATCH_SMTP_HOST"):
        smtp_config_from_env({"DRIFTWATCH_SMTP_FROM": "a@b.com"})
    with pytest.raises(SmtpConfigError, match="DRIFTWATCH_SMTP_FROM"):
        smtp_config_from_env({"DRIFTWATCH_SMTP_HOST": "h"})


def test_smtp_config_ssl_port_default_and_bad_port():
    env = {"DRIFTWATCH_SMTP_HOST": "h", "DRIFTWATCH_SMTP_FROM": "a@b.com",
           "DRIFTWATCH_SMTP_SSL": "1"}
    cfg = smtp_config_from_env(env)
    assert cfg.port == 465 and cfg.use_ssl and not cfg.use_tls
    with pytest.raises(SmtpConfigError, match="must be a number"):
        smtp_config_from_env({**env, "DRIFTWATCH_SMTP_PORT": "abc"})


class FakeSMTP:
    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        self.sent = []
        self.logins = []
        self.tls_started = False

    def ehlo_or_helo_if_needed(self):
        pass

    def starttls(self):
        self.tls_started = True

    def login(self, user, password):
        self.logins.append((user, password))

    def send_message(self, msg):
        self.sent.append(msg)

    def quit(self):
        pass


SMTP_ENV = {"DRIFTWATCH_SMTP_HOST": "smtp.example.com",
            "DRIFTWATCH_SMTP_FROM": "drift@example.com",
            "DRIFTWATCH_SMTP_USER": "bot",
            "DRIFTWATCH_SMTP_PASS": "s3cret"}


def test_send_email_sends_one_per_alert():
    client = FakeSMTP("h", 1)
    send_email("ops@example.com", [make_alert("age"), make_alert("income")],
               env=SMTP_ENV, smtp_class=lambda *a, **k: client)
    assert len(client.sent) == 2
    assert client.logins == [("bot", "s3cret")]
    assert client.tls_started
    msg = client.sent[0]
    assert msg["To"] == "ops@example.com"
    assert msg["From"] == "drift@example.com"
    assert "[driftwatch] age:" in msg["Subject"]
    assert "significant drift" in msg["Subject"]
    body = msg.get_content()
    assert "psi:       0.670" in body
    assert "week3" in body


def test_send_email_noop_on_empty():
    send_email("ops@example.com", [], env=SMTP_ENV,
               smtp_class=lambda *a, **k: (_ for _ in ()).throw(
                   AssertionError("must not connect")))


def test_send_email_missing_config_raises():
    with pytest.raises(SmtpConfigError, match="DRIFTWATCH_SMTP_HOST"):
        send_email("ops@example.com", [make_alert()], env={},
                   smtp_class=FakeSMTP)


def test_send_email_uses_ssl_class_when_configured():
    plain = FakeSMTP("h", 1)
    ssl = FakeSMTP("h", 1)
    env = {**SMTP_ENV, "DRIFTWATCH_SMTP_SSL": "1"}
    send_email("ops@example.com", [make_alert()], env=env,
               smtp_class=lambda *a, **k: plain,
               smtp_ssl_class=lambda *a, **k: ssl)
    assert len(ssl.sent) == 1 and not plain.sent
    assert not ssl.tls_started  # implicit TLS: no STARTTLS upgrade


def test_send_email_wraps_smtp_failure():
    class Broken(FakeSMTP):
        def send_message(self, msg):
            raise OSError("connection reset")
    with pytest.raises(NotifyError, match="sending alert email"):
        send_email("ops@example.com", [make_alert()], env=SMTP_ENV,
                   smtp_class=Broken)


# ------------------------------------------------- dispatch integration


def test_dispatch_webhook_and_email_hooks():
    calls = []

    def fake_webhook(url, alerts):
        calls.append(("webhook", url, [a.feature for a in alerts]))

    def fake_email(addr, alerts):
        calls.append(("email", addr, [a.feature for a in alerts]))

    hooks = [parse_alert_hook("webhook:https://x.example/h"),
             parse_alert_hook("email:ops@example.com")]
    errs = dispatch_alerts([make_alert("age")], hooks,
                           webhook_sender=fake_webhook,
                           email_sender=fake_email)
    assert errs == []
    assert calls == [("webhook", "https://x.example/h", ["age"]),
                     ("email", "ops@example.com", ["age"])]


def test_dispatch_failing_hook_does_not_block_others():
    def boom(url, alerts):
        raise OSError("unreachable")

    delivered = []
    hooks = [parse_alert_hook("webhook:https://x.example/h"),
             parse_alert_hook("email:ops@example.com")]
    errs = dispatch_alerts(
        [make_alert()], hooks,
        webhook_sender=boom,
        email_sender=lambda addr, alerts: delivered.append(addr))
    assert len(errs) == 1 and "webhook hook failed" in errs[0]
    assert delivered == ["ops@example.com"]  # email still attempted


def test_dispatch_default_senders_are_real_ones():
    # with no alerts, the real senders must not touch the network
    hooks = [parse_alert_hook("webhook:https://x.example/h"),
             parse_alert_hook("email:ops@example.com")]
    assert dispatch_alerts([], hooks) == []
