"""Closed procedure registry and template-aware compilation (paper §3.3).

For a detected protocol family ``f`` the registry ``R_f`` is finite and closed:
only the registered template identifiers are admissible and there is no
free-form fallback. Candidate ``c`` refers to a specification ``rho(c)`` that
carries its procedure template and exploit-specific operands; compiling it keeps
the two binding provenances apart:

    B_env  = Resolve_env(M_W, C_A)   -> resolve_env_bindings()
    B_proc = Resolve_proc(rho(c))    -> resolve_proc_operands()

``Resolve_env`` supplies target-dependent values only when supported by ``M_W``
or authorized by ``C_A``; ``Resolve_proc`` supplies exploit-specific operands
only. Every rendered procedure is an argv array whose ``argv[0]`` belongs to
``APPROVED_TOOLS`` — adapters serialize and transport, never shell text.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

from src.memory.episodic import procedure_id
from src.memory.world_state import WorldState
from src.pipeline.case_contract import normalize_protocol_family

#: ``R_f`` — registered procedure templates per protocol family (paper §3.3).
TEMPLATES: dict[str, tuple[str, ...]] = {
    "http": (
        "http-request", "http-put-range", "http-batch", "jolokia-exec", "jdbc-url", "form-auth",
    ),
    "websocket": ("websocket-frame",),
    "ssh": ("ssh-channel",),
}

#: binding names ``Resolve_env(M_W, C_A)`` may supply: target-derived values only.
#: ``Resolve_env`` may also supply ``USERNAME``/``PASSWORD`` when ``M_W`` holds a
#: *verified* credential (paper §3.2), which is how an authenticated procedure
#: gets its secret without the spec hard-coding one.
ENV_BINDINGS: frozenset[str] = frozenset({"TARGET_IP", "RPORT", "SCHEME"})

#: The approved tool set (paper §3.3, Theorem 5): every rendered ``argv[0]`` must
#: come from here, and the registry is the only producer of dispatched argv.
APPROVED_TOOLS: frozenset[str] = frozenset({
    "curl", "ssh", "nmap", "nuclei", "msfconsole", "nc", "ncat", "wget",
    "searchsploit", "nikto", "gobuster", "dirb", "wfuzz", "hydra",
    "medusa", "sqlmap",
    # ``ws-frame`` is the first-party WebSocket transport adapter (paper §3.3:
    # "procedure adapters implement transport and serialization"). curl can open
    # a ws:// URL but its CLI has no option that sends an application frame
    # (`curl_ws_send()` is library-only), so the registered ``websocket-frame``
    # template — whose whole point is one frame — cannot be dispatched with curl.
    "ws-frame",
})

#: ports whose default target scheme is ``https``.
_TLS_PORTS = frozenset({"443", "8443"})

_PLACEHOLDER_RE = re.compile(r"\{([A-Z][A-Z0-9_]*)\}")

#: ``argv`` shape per registered template. ``argv[0]`` is the transport tool
#: (fixed by the template, never by a binding); ``{NAME}`` tokens are bindings.
_TEMPLATE_ARGV: dict[str, tuple[str, ...]] = {
    "http-request": (
        "curl", "-sS", "-i", "-X", "{METHOD}", "{SCHEME}://{TARGET_IP}:{RPORT}{PATH}",
    ),
    "http-put-range": (
        "curl", "-sS", "-X", "PUT", "-H", "Content-Range: bytes {RANGE}",
        "--data-binary", "{BODY}", "{SCHEME}://{TARGET_IP}:{RPORT}{PATH}",
    ),
    "http-batch": (
        "curl", "-sS", "-X", "POST", "-H", "Content-Type: application/json",
        "--data-binary", "{BATCH}", "{SCHEME}://{TARGET_IP}:{RPORT}{PATH}",
    ),
    # The frame is the point of this template: the transport must be able to send
    # one, and must not add an ``Origin`` the advisory's precondition forbids.
    "websocket-frame": (
        "ws-frame", "ws://{TARGET_IP}:{RPORT}{PATH}",
        "--subprotocol", "{SUBPROTOCOL}", "--frame", "{FRAME}", "--wait", "3",
    ),
    "ssh-channel": (
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
        "-p", "{RPORT}", "{USERNAME}@{TARGET_IP}", "{CHANNEL_COMMAND}",
    ),
    "jolokia-exec": (
        "curl", "-sS", "-X", "POST", "-H", "Content-Type: application/json",
        "--data-binary", "{M_BEAN}", "{SCHEME}://{TARGET_IP}:{RPORT}{PATH}",
    ),
    "jdbc-url": (
        "curl", "-sS", "-X", "POST", "--data-urlencode", "url={JDBC_URL}",
        "{SCHEME}://{TARGET_IP}:{RPORT}{PATH}",
    ),
    "form-auth": (
        "curl", "-sS", "-i", "-X", "POST", "--data-urlencode", "username={USERNAME}",
        "--data-urlencode", "password={PASSWORD}", "{SCHEME}://{TARGET_IP}:{RPORT}{PATH}",
    ),
}

#: constants that belong to the registered template rather than to ``rho(c)``.
_TEMPLATE_DEFAULTS: dict[str, dict[str, str]] = {
    "http-request": {"METHOD": "GET"},
    # A handshake need not offer a subprotocol; the transport omits the header
    # when the procedure declares none.
    "websocket-frame": {"SUBPROTOCOL": ""},
}


class RegistrationError(ValueError):
    """Raised when a procedure is outside the closed registry or the tool set."""


@dataclass
class CompiledProcedure:
    """``P`` — one registered template rendered with split binding provenance."""

    template_id: str
    family: str
    argv: list[str]
    bindings_env: dict[str, str]      # B_env  = Resolve_env(M_W, C_A)
    bindings_proc: dict[str, str]     # B_proc = Resolve_proc(rho(c))
    proc_id: str                      # id(P) = H(Normalize(P))

    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "family": self.family,
            "argv": list(self.argv),
            "bindings_env": dict(self.bindings_env),
            "bindings_proc": dict(self.bindings_proc),
            "proc_id": self.proc_id,
        }


def allowed_templates(family: str) -> tuple[str, ...]:
    """``R_f`` — the registered template ids of *family* (empty when closed)."""
    return TEMPLATES.get(_text(family).lower(), ())


def is_registered(family: str, template_id: str) -> bool:
    """True iff *template_id* is a member of ``R_f`` for *family*."""
    return _text(template_id) in allowed_templates(family)


def render_template(template_id: str, bindings: Mapping[str, str], *,
                    raw_path: bool = False, headers: Mapping[str, str] | None = None,
                    basic_auth: str = "", request_target: str = "") -> list[str]:
    """Render one registered template into an argv array.

    ``argv[0]`` is checked against ``APPROVED_TOOLS``; a missing binding is a
    :class:`RegistrationError` rather than an unresolved placeholder.

    ``raw_path`` marks a procedure whose operand path must reach the target
    verbatim (path-traversal exploitation depends on the dot segments surviving
    the client): the transport then sends the request without normalising it.

    ``headers`` and ``basic_auth`` are the procedure's own transport
    requirements (a real advisory may need a browser origin or default
    credentials); header values may use the same bindings as the argv, so a
    target-derived value such as ``{TARGET_IP}``/``{RPORT}`` stays resolved by
    ``Resolve_env`` rather than baked into the spec.
    """
    token = _text(template_id)
    argv_template = _TEMPLATE_ARGV.get(token)
    if argv_template is None:
        raise RegistrationError(f"unregistered procedure template {template_id!r}")
    values = {str(name): _text(value) for name, value in dict(bindings or {}).items()}
    defaults = _TEMPLATE_DEFAULTS.get(token, {})
    argv = [
        _render_token(item, template_id=token, values=values, defaults=defaults)
        for item in argv_template
    ]
    if raw_path:
        argv = _with_raw_path(argv)
    argv = _with_transport_options(argv, headers=headers, basic_auth=basic_auth,
                                   values={**defaults, **values}, request_target=request_target)
    _assert_approved_tool(argv)
    return argv


def _pin_env_bindings(declared: Mapping[str, str], env_names: tuple[str, ...],
                      contract: Any) -> dict[str, str]:
    """Env bindings a procedure pins to a value the contract authorizes."""
    authorized: dict[str, set[str]] = {
        "TARGET_IP": {str(item) for item in (getattr(contract, "allowed_hostnames", ()) or ())},
        "RPORT": {str(item) for item in (getattr(contract, "allowed_ports", ()) or ())},
        "SCHEME": {str(item) for item in (getattr(contract, "allowed_schemes", ()) or ())},
    }
    target_host = str(getattr(contract, "target_host", "") or "")
    if target_host:
        authorized["TARGET_IP"].add(target_host)
    pinned: dict[str, str] = {}
    for name in env_names:
        value = _text(declared.get(name) or "")
        allowed = authorized.get(name)
        if value and allowed and value in allowed:
            pinned[name] = value
    return pinned


def _option_binding_names(extra: Mapping[str, Any]) -> set[str]:
    """Placeholder names the procedure's own options use (credentials, headers)."""
    names: set[str] = set()
    texts = [str(extra.get("basic_auth") or ""), str(extra.get("request_target") or "")]
    for value in dict(extra.get("headers") or {}).items():
        texts.extend([str(value[0]), str(value[1])])
    for text in texts:
        names.update(_PLACEHOLDER_RE.findall(text))
    return names


def _with_transport_options(argv: list[str], *, headers: Mapping[str, str] | None,
                            basic_auth: str, values: Mapping[str, str],
                            request_target: str = "") -> list[str]:
    """Insert a procedure's own ``curl`` options (credentials, extra headers).

    The options are attached to the transport, never to the URL, so the scope
    check still reads the destination from the same token it always did.
    """
    if not argv or os.path.basename(argv[0]) != "curl":
        return list(argv)
    def render_option(text: str) -> str:
        return _PLACEHOLDER_RE.sub(
            lambda match: values.get(match.group(1), match.group(0)), _text(text))

    options: list[str] = []
    if _text(basic_auth):
        options += ["-u", render_option(basic_auth)]
    if _text(request_target):
        # Some advisories are *about* the request line itself (an invalid
        # request-target that carries a fragment), which a URL cannot express.
        options += ["--request-target", render_option(request_target)]
    for name, value in dict(headers or {}).items():
        rendered = render_option(value)
        if _text(name) and rendered:
            options += ["-H", f"{_text(name)}: {rendered}"]
    if not options:
        return list(argv)
    return [argv[0], *options, *argv[1:]]


def _with_raw_path(argv: list[str]) -> list[str]:
    """Insert the transport's "do not normalise the path" switch before the URL.

    Only ``curl`` is affected today (``--path-as-is``); a template whose argv
    carries no URL token is returned unchanged, and a switch that is already
    present is never duplicated.
    """
    if not argv or os.path.basename(argv[0]) != "curl" or "--path-as-is" in argv:
        return list(argv)
    for index, item in enumerate(argv):
        if "://" in item:
            return [*argv[:index], "--path-as-is", *argv[index:]]
    return list(argv)


def resolve_env_bindings(
    world_state: WorldState | None,
    contract: Any,
    *,
    state: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """``Resolve_env(M_W, C_A)`` — target-derived bindings only.

    Values come from the contract first, then the run state mapping, then ``M_W``;
    nothing else may enter ``B_env``.
    """
    supplied = dict(state or {})
    host = _text(getattr(contract, "target_host", "")) or _text(supplied.get("target_ip"))
    port = _text(getattr(contract, "target_port", "")) or _text(supplied.get("target_port"))
    if world_state is not None:
        if not host:
            hosts = world_state.nodes("host")
            host = hosts[0][1] if hosts else ""
        if not port:
            ports = world_state.nodes("port")
            port = ports[0][1] if ports else ""

    bindings: dict[str, str] = {}
    if host:
        bindings["TARGET_IP"] = host
        bindings["SCHEME"] = _scheme(contract, supplied.get("scheme"), port)
    if port:
        bindings["RPORT"] = port
    credential = _verified_credential(world_state, port)
    if credential is not None:
        bindings["USERNAME"] = credential.username
        bindings["PASSWORD"] = credential.password
    return bindings


def _verified_credential(world_state: WorldState | None, port: str):
    """The credential ``M_W`` holds for this target, if any is verified.

    Only verified credentials enter ``B_env``: an unverified guess must never be
    rendered into a dispatched procedure.
    """
    if world_state is None:
        return None
    verified = [item for item in getattr(world_state, "credentials", []) or [] if item.verified]
    if not verified:
        return None
    for item in verified:
        if str(getattr(item, "target_service", "") or "") in {str(port), ""}:
            return item
    return verified[0]


def resolve_proc_operands(candidate: Any) -> dict[str, str]:
    """``Resolve_proc(rho(c))`` — operands declared by the candidate's spec.

    Reads scalar ``candidate.bindings`` entries (or ``{"value"|"default": ...}``
    specs) plus the explicit ``candidate.extra["operands"]`` mapping, which wins.
    Nested binding groups such as ``{"env": {...}}`` are not operands.
    """
    operands: dict[str, str] = {}
    for name, spec in (getattr(candidate, "bindings", None) or {}).items():
        value = _operand_value(spec)
        if value is not None:
            operands[str(name)] = value
    extra = getattr(candidate, "extra", None) or {}
    declared = extra.get("operands") if isinstance(extra, Mapping) else None
    if isinstance(declared, Mapping):
        for name, value in declared.items():
            text = _text(value)
            if text:
                operands[str(name)] = text
    return operands


def resolve_family(
    candidate: Any,
    *,
    contract: Any,
    world_state: WorldState | None = None,
) -> str:
    """The protocol family whose ``R_f`` must admit the candidate's template.

    Order: the family the candidate declares, the family ``M_W`` records for the
    contract's (host, port), the contract's declared families, then the single
    family registering the template id.
    """
    template_id = _text(getattr(candidate, "template_id", ""))
    extra = getattr(candidate, "extra", None)
    if isinstance(extra, Mapping):
        for key in ("protocol_family", "family"):
            family = _family_label(extra.get(key))
            if family:
                return family

    host = _text(getattr(contract, "target_host", ""))
    port = _text(getattr(contract, "target_port", ""))
    if world_state is not None and host and port.isdigit():
        service = world_state.service(host, int(port))
        family = _family_label(getattr(service, "protocol_family", ""))
        if family:
            return family

    declared = [_family_label(value) for value in getattr(contract, "protocol_families", ()) or ()]
    declared = [family for family in declared if family]
    for family in declared:
        if is_registered(family, template_id):
            return family
    if declared:
        return declared[0]

    for family, template_ids in TEMPLATES.items():
        if template_id in template_ids:
            return family
    return ""


def compile_procedure(
    candidate: Any,
    *,
    contract: Any,
    world_state: WorldState | None = None,
    state: Mapping[str, Any] | None = None,
) -> CompiledProcedure:
    """Compile ``rho(c)`` against the closed registry for the detected family.

    Raises :class:`RegistrationError` when the candidate carries no template id,
    when the template is not registered for the family, or when the rendered
    ``argv[0]`` falls outside ``APPROVED_TOOLS``.
    """
    template_id = _text(getattr(candidate, "template_id", ""))
    if not template_id:
        raise RegistrationError(
            f"candidate {getattr(candidate, 'candidate_id', '')!r} carries no procedure template id"
        )
    family = resolve_family(candidate, contract=contract, world_state=world_state)
    if not is_registered(family, template_id):
        raise RegistrationError(
            f"procedure template {template_id!r} is not registered for protocol family "
            f"{family or 'unknown'!r}"
        )

    env_names, operand_names = _binding_names(template_id)
    resolved_env = resolve_env_bindings(world_state, contract, state=state)
    env = {name: value for name, value in resolved_env.items() if name in env_names}
    declared = resolve_proc_operands(candidate)
    proc = {name: declared[name] for name in operand_names if name in declared}
    extra = getattr(candidate, "extra", None) or {}
    extra = extra if isinstance(extra, Mapping) else {}
    # An operand the candidate does not declare may still be supplied by
    # ``M_W``/``C_A`` (a verified credential, a discovered value), and the
    # procedure's own options may name bindings the template never mentions.
    from_env = {name: value for name, value in resolved_env.items()
                if name not in proc and (name in operand_names or name in _option_binding_names(extra))}
    # ``Resolve_proc(rho(c))`` may also *select* among the values ``C_A`` already
    # authorizes (a lab's proof endpoint can sit on another authorized port than
    # the exploit's); it can never introduce a value outside the contract.
    env = {**env, **from_env, **_pin_env_bindings(declared, env_names, contract)}
    argv = render_template(
        template_id, {**env, **proc},
        raw_path=bool(extra.get("raw_path")),
        headers=extra.get("headers") if isinstance(extra.get("headers"), Mapping) else None,
        basic_auth=str(extra.get("basic_auth") or ""),
        request_target=str(extra.get("request_target") or ""),
    )
    return CompiledProcedure(
        template_id=template_id,
        family=family,
        argv=argv,
        bindings_env=env,
        bindings_proc=proc,
        proc_id=procedure_id(argv),
    )


# ── Rendering internals ─────────────────────────────────────────────────────


def _binding_names(template_id: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(env_names, operand_names)`` of one template, in argv order."""
    env: list[str] = []
    operands: list[str] = []
    for item in _TEMPLATE_ARGV.get(_text(template_id), ()):
        for name in _PLACEHOLDER_RE.findall(item):
            target = env if name in ENV_BINDINGS else operands
            if name not in target:
                target.append(name)
    return tuple(env), tuple(operands)


def _render_token(
    item: str,
    *,
    template_id: str,
    values: Mapping[str, str],
    defaults: Mapping[str, str],
) -> str:
    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in values:
            return values[name]
        if name in defaults:
            return defaults[name]
        raise RegistrationError(f"template {template_id!r} requires binding {name}")

    return _PLACEHOLDER_RE.sub(substitute, item)


def _assert_approved_tool(argv: list[str]) -> None:
    tool = os.path.basename(argv[0]) if argv else ""
    if tool not in APPROVED_TOOLS:
        raise RegistrationError(f"procedure tool {tool!r} is outside the approved tool set")


def _operand_value(spec: Any) -> str | None:
    if isinstance(spec, Mapping):
        for key in ("value", "default"):
            if key in spec:
                text = _text(spec[key])
                if text:
                    return text
        return None
    return _text(spec) or None


def _scheme(contract: Any, declared: Any, port: str) -> str:
    candidates = [_text(declared)]
    candidates.extend(_text(item) for item in getattr(contract, "allowed_schemes", ()) or ())
    for value in candidates:
        if value in ("http", "https", "ws", "wss"):
            return value
    return "https" if port in _TLS_PORTS else "http"


def _family_label(value: Any) -> str:
    label = normalize_protocol_family(value)
    return "" if label in ("", "unknown") else label


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()
