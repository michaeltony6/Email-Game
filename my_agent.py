"""
Email Game custom agent.

Copy or pass this file as the --module argument:

    python scripts/run_custom_agent.py myname "My Name" --module my_agent.py --server ...

The agent is intentionally conservative about signing and opportunistic about
collecting.  It remembers first-round language so later fuzzy descriptions can
be resolved without trusting claims made by other agents.
"""

from __future__ import annotations

import re
import json
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Optional

from src.base_agent import BaseAgent


MODERATOR_NAMES = {"moderator", "game", "system", "admin", "server"}


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _split_items(text: str) -> list[str]:
    text = re.sub(r"\band\b", ",", text, flags=re.I)
    text = re.sub(r"[;\n]+", ",", text)
    return [_norm(x.strip(" -*\t\r\n.'\"`")) for x in text.split(",") if _norm(x.strip(" -*\t\r\n.'\"`"))]


def _extract_quoted(text: str) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(r'"([^"\n]{1,240})"', text)]


def _section(body: str, labels: tuple[str, ...]) -> Optional[str]:
    joined = "|".join(re.escape(x) for x in labels)
    pattern = rf"(?is)(?:^|\n)\s*(?:{joined})\s*[:\-]\s*(.+?)(?=\n\s*[A-Z][A-Za-z ]{{1,40}}\s*[:\-]|\Z)"
    match = re.search(pattern, body)
    return match.group(1).strip() if match else None


def _looks_like_signature(text: str) -> bool:
    lowered = text.lower()
    return (
        "signature" in lowered
        or "signed" in lowered
        or bool(re.search(r"-----BEGIN [A-Z ]*SIGNATURE-----", text))
        or bool(re.search(r"\bsig(?:nature)?[-_: ][A-Za-z0-9+/=._-]{16,}", text))
    )


@dataclass
class RoundState:
    assigned_message: str = ""
    request_targets: list[str] = field(default_factory=list)
    signing_targets: list[str] = field(default_factory=list)
    requested_from: set[str] = field(default_factory=set)
    pressure_sent: set[str] = field(default_factory=set)
    signed_for: set[tuple[str, str]] = field(default_factory=set)
    submitted: set[str] = field(default_factory=set)


class CustomAgent(BaseAgent):
    """A hard-to-trick, transcript-aware Email Game agent."""

    def on_new_game(self):
        try:
            super().on_new_game()
        except AttributeError:
            pass
        self.round_no = 0
        self.state = RoundState()
        self.agent_aliases: dict[str, str] = {}
        self.agent_text: dict[str, list[str]] = defaultdict(list)
        self.pending_requests: list[dict[str, str]] = []
        self.known_agents: set[str] = set()

    def on_message_batch(self, messages):
        self._ensure_state()
        for msg in messages:
            sender = _norm(msg.get("from", ""))
            body_raw = str(msg.get("body", "") or "")
            body = _norm(body_raw)
            subject = _norm(msg.get("subject", ""))
            if not sender:
                continue

            self.known_agents.add(sender)
            if not self._is_moderator(sender):
                self.agent_text[sender].append(" ".join(x for x in (subject, body) if x))

            if self._is_moderator(sender):
                self._handle_moderator(body_raw)
            else:
                self._capture_signature(sender, subject, body)
                request = self._parse_signature_request(sender, subject, body)
                if request:
                    self.pending_requests.append(request)

        self._process_pending_requests()
        self._send_collection_requests()
        self._send_pressure_requests()

        # Keep the stock LLM pipeline as an emergency fallback only if the
        # runner exposes it and we have no structured round instructions yet.
        if not self.state.assigned_message and hasattr(super(), "on_message_batch"):
            try:
                super().on_message_batch(messages)
            except Exception:
                pass

    def _ensure_state(self) -> None:
        if not hasattr(self, "state"):
            self.on_new_game()

    def _is_moderator(self, sender: str) -> bool:
        return _key(sender) in MODERATOR_NAMES or any(x in _key(sender) for x in MODERATOR_NAMES)

    def _handle_moderator(self, body: str) -> None:
        parsed = self._parse_round_assignment(body)
        if not parsed:
            return

        assigned, requests, signing = parsed
        self.round_no += 1
        self.state = RoundState(
            assigned_message=assigned,
            request_targets=[self._resolve_agent(x) for x in requests],
            signing_targets=[self._resolve_agent(x) for x in signing],
        )

        for raw, resolved in zip(requests + signing, self.state.request_targets + self.state.signing_targets):
            if raw and resolved:
                self.agent_aliases[_key(raw)] = resolved

    def _parse_round_assignment(self, body: str) -> Optional[tuple[str, list[str], list[str]]]:
        json_assignment = self._parse_json_assignment(body)
        if json_assignment:
            return json_assignment

        message = None
        for labels in (
            ("assigned message", "message to collect", "your message", "message"),
            ("assigned string", "target message", "string"),
        ):
            sec = _section(body, labels)
            if sec:
                quoted = _extract_quoted(sec)
                message = quoted[0] if quoted else _split_items(sec)[0]
                break

        if not message:
            quoted = _extract_quoted(body)
            if quoted:
                message = quoted[0]

        requests = self._extract_list(
            body,
            ("request list", "collect signatures from", "must collect signatures from", "request signatures from"),
        )
        signing = self._extract_list(
            body,
            ("signing list", "authorized to sign for", "may sign for", "allowed to sign for"),
        )

        if message and (requests or signing):
            return message, requests, signing
        return None

    def _parse_json_assignment(self, body: str) -> Optional[tuple[str, list[str], list[str]]]:
        try:
            data = json.loads(body)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None

        message = (
            data.get("assigned_message")
            or data.get("message")
            or data.get("target_message")
            or data.get("assigned")
        )
        requests = (
            data.get("request_list")
            or data.get("requests")
            or data.get("collect_from")
            or data.get("request_targets")
            or []
        )
        signing = (
            data.get("signing_list")
            or data.get("authorized_to_sign_for")
            or data.get("sign_for")
            or data.get("signing_targets")
            or []
        )
        if isinstance(requests, str):
            requests = _split_items(requests)
        if isinstance(signing, str):
            signing = _split_items(signing)
        if message and (requests or signing):
            return _norm(message), [_norm(x) for x in requests], [_norm(x) for x in signing]
        return None

    def _extract_list(self, body: str, labels: tuple[str, ...]) -> list[str]:
        sec = _section(body, labels)
        if not sec:
            return []
        items = _split_items(sec)
        return [x for x in items if _key(x) not in {"none", "n a", "na"}]

    def _parse_signature_request(self, sender: str, subject: str, body: str) -> Optional[dict[str, str]]:
        haystack = f"{subject}\n{body}"
        if not re.search(r"\b(sign|signature|authorize|please)\b", haystack, re.I):
            return None

        message = None
        quoted = _extract_quoted(haystack)
        if quoted:
            message = quoted[0]
        else:
            match = re.search(
                r"(?is)(?:sign(?:ature)?(?:\s+(?:this|the|my))?\s*(?:message|string)?|message(?:\s+to\s+sign)?)\s*[:\-]\s*(.{1,240})",
                haystack,
            )
            if match:
                message = _split_items(match.group(1))[0]

        if not message:
            return None

        return {"sender": sender, "message": message}

    def _capture_signature(self, sender: str, subject: str, body: str) -> None:
        if not self.state.assigned_message:
            return
        if self.state.assigned_message not in body and self.state.assigned_message not in subject:
            return
        if not _looks_like_signature(body):
            return

        payload = body.strip()
        sig_key = _key(payload)
        if sig_key in self.state.submitted:
            return
        self.state.submitted.add(sig_key)
        try:
            self.submit_signature(payload)
        except Exception:
            # Some runners include only the signed blob on a specific line.
            for line in body.splitlines():
                if _looks_like_signature(line) and self.state.assigned_message in line:
                    try:
                        self.submit_signature(line.strip())
                        break
                    except Exception:
                        pass

    def _process_pending_requests(self) -> None:
        if not self.pending_requests:
            return

        still_pending = []
        for req in self.pending_requests:
            sender = req["sender"]
            message = req["message"]
            pair = (_key(sender), _key(message))
            if pair in self.state.signed_for:
                continue
            if self._is_authorized(sender):
                self.state.signed_for.add(pair)
                self._safe_sign(sender, message)
            else:
                still_pending.append(req)
        self.pending_requests = still_pending[-12:]

    def _send_collection_requests(self) -> None:
        if not self.state.assigned_message:
            return

        for target in self.state.request_targets:
            if not target or _key(target) in self.state.requested_from:
                continue
            self.state.requested_from.add(_key(target))
            subject = "Signature request"
            body = (
                "Please sign exactly this assigned message for me:\n"
                f'"{self.state.assigned_message}"\n\n'
                "The moderator assigned me to collect your signature for this message. "
                "Please reply with the signed payload as soon as you can."
            )
            try:
                self.send_message(target, subject, body)
            except Exception:
                pass

    def _send_pressure_requests(self) -> None:
        if not self.state.assigned_message:
            return

        direct_targets = {_key(x) for x in self.state.request_targets}
        for target in sorted(self.known_agents):
            target_key = _key(target)
            if (
                not target
                or self._is_moderator(target)
                or self._is_self(target)
                or target_key in direct_targets
                or target_key in self.state.pressure_sent
            ):
                continue
            self.state.pressure_sent.add(target_key)
            body = (
                "Fast signature request for the current round. Please sign exactly:\n"
                f'"{self.state.assigned_message}"\n\n'
                "I am collecting confirmations now and will submit signed payloads immediately."
            )
            try:
                self.send_message(target, "Round signature request", body)
            except Exception:
                pass

    def _safe_sign(self, recipient: str, message: str) -> None:
        body = (
            "Authorized signature response. I am signing only the message you requested "
            "because you match my current moderator signing list."
        )
        try:
            self.sign_and_respond(recipient, message, body, "Authorized signature")
        except TypeError:
            self.sign_and_respond(to_agent=recipient, message_to_sign=message, response_body=body, subject="Authorized signature")
        except Exception:
            pass

    def _is_authorized(self, sender: str) -> bool:
        sender_key = _key(sender)
        return any(sender_key == _key(x) for x in self.state.signing_targets if x)

    def _is_self(self, sender: str) -> bool:
        sender_key = _key(sender)
        candidates = (
            getattr(self, "name", ""),
            getattr(self, "agent_name", ""),
            getattr(self, "agent_id", ""),
            getattr(self, "id", ""),
            getattr(self, "display_name", ""),
        )
        return any(sender_key and sender_key == _key(x) for x in candidates if x)

    def _resolve_agent(self, description: str) -> str:
        desc = _norm(description)
        desc_key = _key(desc)
        if not desc_key:
            return ""
        if desc_key in self.agent_aliases:
            return self.agent_aliases[desc_key]

        for known in sorted(self.known_agents, key=len, reverse=True):
            known_key = _key(known)
            if known_key and (known_key == desc_key or known_key in desc_key or desc_key in known_key):
                return known

        best_agent = desc
        best_score = 0.0
        desc_words = set(desc_key.split())
        for agent, snippets in self.agent_text.items():
            corpus = _key(" ".join(snippets[-20:]))
            if not corpus:
                continue
            corpus_words = set(corpus.split())
            overlap = len(desc_words & corpus_words) / max(1, len(desc_words))
            ratio = SequenceMatcher(None, desc_key, corpus[:500]).ratio()
            score = max(overlap, ratio)
            if score > best_score:
                best_score = score
                best_agent = agent

        # Fuzzy labels are often short clues. Require a real signal before
        # converting them to a concrete agent; otherwise leave the text intact,
        # which prevents accidental unauthorized signing.
        return best_agent if best_score >= 0.28 else desc
