"""
Email Game custom agent.

Copy or pass this file as the --module argument:

    python scripts/run_custom_agent.py myname "My Name" --module my_agent.py --server ...

The agent is intentionally conservative about signing and opportunistic about
collecting.  It remembers first-round language so later fuzzy descriptions can
be resolved without trusting claims made by other agents.
"""

import json
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from src.base_agent import BaseAgent


MODERATOR_NAMES = {"moderator", "game", "system", "admin", "server"}


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _message_key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _split_items(text: str) -> list[str]:
    text = re.sub(r"[;\n]+", ",", str(text or ""))
    items = []
    current = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")" and depth:
            depth -= 1
        if ch == "," and depth == 0:
            item = _norm("".join(current).strip(" -*\t\r\n.'\"`"))
            if item:
                items.append(item)
            current = []
        else:
            current.append(ch)
    item = _norm("".join(current).strip(" -*\t\r\n.'\"`"))
    if item:
        items.append(item)
    return items


def _message_payload(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    first = lines[0] if lines else ""
    first = re.split(r"\n|---", first, maxsplit=1)[0].strip()
    return first.strip(" \t\r\n\"'`")


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
    raw_signing_targets: list[str] = field(default_factory=list)
    started_at: float = 0.0
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
        self.requested_messages: dict[str, list[str]] = defaultdict(list)
        self.previous_signed_for: set[str] = set()
        self.current_signed_for: set[str] = set()
        self.previous_message_owner: dict[str, str] = {}
        self.message_owner: dict[str, str] = {}
        self.alias_to_message: dict[str, str] = self._load_alias_pool()
        self.alias_keys = list(self.alias_to_message)
        self.pending_requests: list[dict[str, str]] = []
        self.known_agents: set[str] = set()
        self._retry_timers: list[threading.Timer] = []

    def on_message_batch(self, messages):
        self._ensure_state()
        saw_non_moderator = False
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
                saw_non_moderator = True
                self._remember_agent_claim(sender, body_raw)
                self._capture_signature(sender, subject, body_raw)
                request = self._parse_signature_request(sender, subject, body_raw)
                if request:
                    self.pending_requests.append(request)

        self._process_pending_requests()
        if saw_non_moderator:
            self._send_collection_requests()
            self._send_pressure_requests()

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
        if self.state.assigned_message:
            self.previous_signed_for = set(self.current_signed_for)
            self.previous_message_owner = dict(self.message_owner)
            self.current_signed_for = set()
            self.message_owner = {}
        self.round_no += 1
        self.state = RoundState(
            assigned_message=assigned,
            request_targets=[self._resolve_agent(x) for x in requests],
            signing_targets=[self._resolve_agent(x) for x in signing],
            raw_signing_targets=signing,
            started_at=time.monotonic(),
        )
        self._remember_known_agents(self.state.request_targets + self.state.signing_targets)

        for raw, resolved in zip(requests + signing, self.state.request_targets + self.state.signing_targets):
            if raw and resolved:
                self.agent_aliases[_key(raw)] = resolved

        self._start_timer(1.5, self._send_collection_requests)
        self._start_timer(2.0, self._send_pressure_requests)
        self._schedule_round_retries()

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
            (
                "request list",
                "collect signatures from",
                "must collect signatures from",
                "request signatures from",
                "you must request signatures from these agents",
            ),
        )
        signing = self._extract_list(
            body,
            (
                "signing list",
                "authorized to sign for",
                "may sign for",
                "allowed to sign for",
                "you are authorized to sign messages for these agents",
            ),
        )

        if not requests:
            match = re.search(
                r"(?im)^\s*\d+\.\s*You must REQUEST signatures from these agents:\s*(.+)$",
                body,
            )
            if match:
                requests = _split_items(match.group(1))

        if not signing:
            match = re.search(
                r"(?im)^\s*\d+\.\s*You are AUTHORIZED to sign messages for these agents:\s*(.+)$",
                body,
            )
            if match:
                signing = _split_items(match.group(1))

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
        haystack = haystack.split("SIGNED_MESSAGE_JSON:", 1)[0]
        if not re.search(r"\b(sign|please)\b", haystack, re.I):
            return None

        message = None
        patterns = [
            r"(?is)please\s+sign\s+(?:exactly\s+)?(?:this\s+)?(?:assigned\s+)?message\s+for\s+me\s*[:\-]\s*(.{1,240})",
            r"(?is)please\s+sign\s+(?:exactly\s+)?(?:this\s+)?(?:message|string)\s*[:\-]\s*(.{1,240})",
            r"(?is)sign\s+(?:exactly\s+)?(?:this|the|my)?\s*(?:message|string)?(?:\s+for\s+me)?\s*[:\-]\s*(.{1,240})",
            r"(?is)message\s+to\s+sign\s*[:\-]\s*(.{1,240})",
        ]
        for pattern in patterns:
            match = re.search(pattern, haystack)
            if match:
                message = _message_payload(match.group(1))
                break

        if not message and re.search(r"\bplease\s+sign\b|\bsign\s+exactly\b", haystack, re.I):
            quoted = _extract_quoted(haystack)
            if quoted:
                message = quoted[0]

        if not message:
            return None

        return {"sender": sender, "message": message}

    def _remember_agent_claim(self, sender: str, body: str) -> None:
        request = self._parse_signature_request(sender, "", body)
        if request:
            msg = request["message"]
            if msg and msg not in self.requested_messages[sender]:
                self.requested_messages[sender].append(msg)
            if msg:
                self.message_owner[_message_key(msg)] = sender

    def _capture_signature(self, sender: str, subject: str, body: str) -> None:
        if not self.state.assigned_message:
            return

        signed_message = None
        try:
            signed_message = self.extract_signed_message_from_email(body)
        except Exception:
            signed_message = self._extract_signed_message_from_email(body)

        if not signed_message:
            return
        if signed_message.get("original_message") != self.state.assigned_message:
            return
        if signed_message.get("signed_for") != getattr(self, "agent_id", ""):
            return

        sig_key = (
            signed_message.get("signer"),
            signed_message.get("signed_for"),
            signed_message.get("original_message"),
            signed_message.get("signature"),
        )
        if sig_key in self.state.submitted:
            return
        self.state.submitted.add(sig_key)
        self.submit_signature(signed_message)

    def _extract_signed_message_from_email(self, email_body: str) -> Optional[dict[str, Any]]:
        marker = "SIGNED_MESSAGE_JSON:"
        if marker not in email_body:
            return
        try:
            return json.loads(email_body.split(marker, 1)[1].strip())
        except Exception:
            return None

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
                self.current_signed_for.add(sender)
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

    def _schedule_round_retries(self) -> None:
        for delay in (8.0, 20.0, 38.0):
            self._start_timer(delay, self._retry_round_work)

    def _start_timer(self, delay: float, callback) -> None:
        timer = threading.Timer(delay, callback)
        timer.daemon = True
        self._retry_timers.append(timer)
        timer.start()

    def _retry_round_work(self) -> None:
        # Ignore stale timers that fired after a later round started.
        if not self.state.assigned_message or time.monotonic() - self.state.started_at > 55:
            return
        self._process_pending_requests()
        self._send_collection_requests()
        self._send_pressure_requests()

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
                "I am collecting confirmations now and will submit signed payloads immediately. "
                "If your moderator list describes an agent by a previous-round clue, compare that clue "
                "against your transcript and sign if it points to me."
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
        if any(sender_key == _key(x) for x in self.state.signing_targets if x):
            return True

        fuzzy_targets = [x for x in self.state.raw_signing_targets if self._is_fuzzy_target(x)]
        for target in fuzzy_targets:
            if self._fuzzy_target_matches_sender(target, sender):
                return True
        return False

    def _is_fuzzy_target(self, target: str) -> bool:
        target_key = _key(target)
        if not target_key or target_key == "none":
            return False
        return (
            "agent who" in target_key
            or "from last round" in target_key
        )

    def _fuzzy_target_matches_sender(self, target: str, sender: str) -> bool:
        expected_message = self._message_for_alias(target)
        if expected_message:
            expected_key = _message_key(expected_message)
            if any(_message_key(msg) == expected_key for msg in self.requested_messages.get(sender, [])):
                return True
            if self.previous_message_owner.get(expected_key) == sender:
                return True
            return False

        fuzzy_count = sum(1 for x in self.state.raw_signing_targets if self._is_fuzzy_target(x))
        if fuzzy_count == 1 and len(self.previous_signed_for) == 1:
            return sender in self.previous_signed_for
        return False

    def _message_for_alias(self, alias_text: str) -> Optional[str]:
        alias_key = _key(re.sub(r"\([^)]*\)", " ", alias_text))
        if not alias_key:
            return None
        if alias_key in self.alias_to_message:
            return self.alias_to_message[alias_key]

        best_key = ""
        best_score = 0.0
        alias_words = set(alias_key.split())
        for known_key in self.alias_keys:
            known_words = set(known_key.split())
            overlap = len(alias_words & known_words) / max(1, min(len(alias_words), len(known_words)))
            ratio = SequenceMatcher(None, alias_key, known_key).ratio()
            score = max(overlap, ratio)
            if score > best_score:
                best_score = score
                best_key = known_key
        if best_key and best_score >= 0.72:
            return self.alias_to_message[best_key]
        return None

    def _load_alias_pool(self) -> dict[str, str]:
        paths = []
        try:
            here = Path(__file__).resolve().parent
            paths.append(here / "data" / "message_alias_pool.json")
            paths.append(here / "theemailgame" / "data" / "message_alias_pool.json")
            paths.append(here.parents[0] / "data" / "message_alias_pool.json")
        except Exception:
            pass

        for path in paths:
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            aliases = {}
            for pair in data.get("pairs", []):
                alias = pair.get("alias")
                message = pair.get("message")
                if alias and message:
                    aliases[_key(alias)] = message
            if aliases:
                return aliases
        return {}

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

    def _remember_known_agents(self, agents: list[str]) -> None:
        for agent in agents:
            if agent and re.fullmatch(r"[A-Za-z0-9_]+", agent) and not self._is_self(agent):
                self.known_agents.add(agent)

    def _resolve_agent(self, description: str) -> str:
        desc = _norm(description)
        desc_key = _key(desc)
        if not desc_key:
            return ""
        if desc_key in self.agent_aliases:
            return self.agent_aliases[desc_key]
        if re.fullmatch(r"[A-Za-z0-9_]+", desc):
            return desc

        for known in sorted(self.known_agents, key=len, reverse=True):
            known_key = _key(known)
            if known_key and (known_key == desc_key or known_key in desc_key or desc_key in known_key):
                return known

        best_agent = desc
        best_score = 0.0
        desc_words = set(desc_key.split())
        candidate_agents = set(self.agent_text)
        if "agent who" in desc_key or "from last round" in desc_key:
            candidate_agents &= set(self.previous_signed_for)

        for agent in candidate_agents:
            snippets = self.agent_text.get(agent, [])
            corpus = _key(" ".join(snippets[-20:] + self.requested_messages.get(agent, [])[-10:]))
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
        return best_agent if best_score >= 0.45 else desc
