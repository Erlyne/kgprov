"""
Bedrock client wrapper.

Four responsibilities, kept out of the pipeline phases so that
provenance and inference concerns do not entangle:

  1. Persistence. Every request and response is written to a content-
     addressed log. A run in replay mode reads from the log and issues
     zero API calls, which is what makes the pipeline reproducible
     despite greedy decoding not being deterministic.

  2. Fault tolerance. Retry with exponential backoff on throttling and
     transient faults. Exhausted calls raise; each phase in
     pipeline/phases.py catches the exception, prints an ERROR line and
     continues.

  3. Structured output, by two different mechanisms:

       * ``complete()`` sends the schema nowhere. It prompts for JSON and
         validates afterwards. The model is free to emit anything; a
         violation is caught, not prevented. This is not constrained
         decoding.

       * ``complete_constrained()`` sends the schema to the provider as a
         forced tool call (``toolConfig`` + ``toolChoice``). Enumerated
         fields are enforced at decode time: the tokens spelling an
         out-of-enum value are masked, so a violation cannot occur.

     Only the second path enforces an admissible set at decode time.
     ``complete()`` is retained for unconstrained calls only.

  4. Bounded agency. ``run_agent()`` drives a multi-turn tool-use loop in
     which the model chooses whether and what to call. Termination is
     enforced by this module through a tool-call budget and a read
     timeout on each request, never requested of the model in the prompt.

The wrapper exposes token usage so cost can be tracked per phase and per
run, and so the model-tier comparison can be priced.
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any



@dataclass
class LLMResponse:
    text: str
    parsed: Any | None
    input_tokens: int
    output_tokens: int
    model_id: str
    from_cache: bool = False
    raw: dict = field(default_factory=dict)


@dataclass
class AgentRun:
    """Outcome of one bounded agent invocation.

    ``n_tool_calls`` is the field that distinguishes an agent from a
    lookup: a Tier 2 run with zero tool calls never used its tools.
    """
    submitted: Any | None
    tool_calls: list
    n_tool_calls: int
    n_model_turns: int
    budget_exhausted: bool
    input_tokens: int
    output_tokens: int
    mode: str = "agent"
    from_cache: bool = False


class SchemaValidationError(Exception):
    pass


_RETRYABLE = {"ThrottlingException", "ServiceUnavailableException",
              "ModelTimeoutException", "InternalServerException"}


class BedrockClient:
    def __init__(self, region: str, response_log_dir: str,
                 replay_mode: bool = False, max_retries: int = 5):
        self.replay_mode = replay_mode
        self.max_retries = max_retries
        self.log_dir = Path(response_log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if replay_mode:
            self._client = None
        else:
            import boto3
            from botocore.config import Config
            # Explicit read/connect timeouts prevent an indefinite hang
            # on a broken TCP connection (default boto3 has no read timeout).
            # The retry loops below handle retryable failures on top of this.
            self._client = boto3.client(
                "bedrock-runtime", region_name=region,
                config=Config(
                    retries={"max_attempts": 0},
                    connect_timeout=10,
                    read_timeout=90,
                ))
        # Notional totals: every response, cached or live. This is what
        # the run would cost from cold.
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        # Billed totals: live API calls only. This is what this invocation
        # spent. Without them a replay looks identical in price to a cold
        # run.
        self.billed_input_tokens = 0
        self.billed_output_tokens = 0
        self.n_calls_live = 0
        self.n_calls_cached = 0
        # Per-model totals, because a blended price is wrong once two
        # models are used.
        self.by_model: dict[str, dict] = {}

    # -- persistence -------------------------------------------------

    def _account(self, tok_in: int, tok_out: int, from_cache: bool,
                 model_id: str = "unknown") -> None:
        """Record token usage: notional vs billed, and per model.

        A cached response cost nothing this run, so counting it toward
        spend makes a replay indistinguishable in price from a cold run.
        And tokens must be attributed to the model that produced them, or
        a two-arm run cannot be priced at all.
        """
        m = self.by_model.setdefault(
            model_id, {"input": 0, "output": 0, "billed_input": 0,
                       "billed_output": 0, "live": 0, "cached": 0})
        m["input"] += tok_in
        m["output"] += tok_out
        m["cached" if from_cache else "live"] += 1
        if not from_cache:
            m["billed_input"] += tok_in
            m["billed_output"] += tok_out
        self.total_input_tokens += tok_in
        self.total_output_tokens += tok_out
        if from_cache:
            self.n_calls_cached += 1
        else:
            self.n_calls_live += 1
            self.billed_input_tokens += tok_in
            self.billed_output_tokens += tok_out

    def _key(self, model_id: str, system: str, user: str,
             schema: dict | None) -> str:
        payload = json.dumps({"m": model_id, "s": system, "u": user,
                              "sc": schema}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def _key_constrained(self, model_id: str, system: str, user: str,
                         tool_name: str, tool_schema: dict) -> str:
        """Cache key for a forced-tool call.

        The ``mode`` discriminator keeps a constrained and an
        unconstrained call with the same prompt in separate cache entries.
        Without it, a replay could serve an unconstrained response to a
        run that records decode-time enforcement.
        """
        payload = json.dumps({"mode": "constrained", "m": model_id,
                              "s": system, "u": user, "t": tool_name,
                              "ts": tool_schema}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def _key_agent(self, model_id: str, system: str, user: str,
                   tool_specs: list) -> str:
        payload = json.dumps({"mode": "agent", "m": model_id, "s": system,
                              "u": user, "tools": tool_specs},
                             sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def _log_path(self, key: str) -> Path:
        return self.log_dir / f"{key}.json"

    def _load(self, key: str) -> dict | None:
        p = self._log_path(key)
        if p.exists():
            return json.loads(p.read_text())
        return None

    def _save(self, key: str, record: dict) -> None:
        self._log_path(key).write_text(json.dumps(record, indent=2))

    # -- invocation --------------------------------------------------

    def complete(self, model_id: str, system: str, user: str,
                 *, schema: dict | None = None, temperature: float = 0.0,
                 max_tokens: int = 2048) -> LLMResponse:
        key = self._key(model_id, system, user, schema)
        cached = self._load(key)
        if cached is not None:
            return self._to_response(cached, model_id, from_cache=True,
                                     schema=schema)

        if self.replay_mode:
            raise RuntimeError(
                f"replay_mode: no logged response for key {key[:12]}. "
                "The pipeline reached a call not present in the response log.")

        record = self._invoke_with_retry(
            model_id, system, user, temperature, max_tokens)
        self._save(key, record)
        return self._to_response(record, model_id, from_cache=False,
                                 schema=schema)

    # -- constrained decoding ---------------------------------------

    @staticmethod
    def tool_spec(name: str, description: str, schema: dict) -> dict:
        return {"toolSpec": {"name": name, "description": description,
                             "inputSchema": {"json": schema}}}

    def complete_constrained(self, model_id: str, system: str, user: str,
                             *, tool_name: str, tool_schema: dict,
                             description: str = "Submit the result.",
                             temperature: float = 0.0,
                             max_tokens: int = 2048) -> LLMResponse:
        """One forced tool call. Enumerated fields hold at decode time.

        ``toolChoice`` naming a specific tool obliges the model to emit
        exactly one call to it with arguments conforming to
        ``tool_schema``. The result is read from the ``toolUse`` block;
        no JSON is scraped out of free text.

        A post-check still runs at the call site. Constrained decoding
        governs shape, never correctness: it guarantees the returned
        value is *in* the enum, not that it is the *right* member of it.
        """
        key = self._key_constrained(model_id, system, user,
                                    tool_name, tool_schema)
        cached = self._load(key)
        if cached is not None:
            return self._to_constrained_response(cached, model_id, True)

        if self.replay_mode:
            raise RuntimeError(
                f"replay_mode: no logged response for key {key[:12]}. "
                "The pipeline reached a call not present in the log.")

        kwargs = dict(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"temperature": temperature,
                             "maxTokens": max_tokens},
            toolConfig={
                "tools": [self.tool_spec(tool_name, description,
                                         tool_schema)],
                "toolChoice": {"tool": {"name": tool_name}},
            },
        )
        if system:
            kwargs["system"] = [{"text": system}]

        resp = self._converse_with_retry(**kwargs)
        blocks = resp["output"]["message"]["content"]
        tool_input = None
        for b in blocks:
            if "toolUse" in b and b["toolUse"]["name"] == tool_name:
                tool_input = b["toolUse"]["input"]
                break
        if tool_input is None:
            # toolChoice makes this a provider-side failure, not a model
            # one. Surface it rather than silently degrading to text.
            raise SchemaValidationError(
                f"forced tool '{tool_name}' produced no toolUse block; "
                f"stopReason={resp.get('stopReason')}")

        usage = resp.get("usage", {})
        record = {
            "mode": "constrained",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "stop_reason": resp.get("stopReason"),
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
        }
        self._save(key, record)
        return self._to_constrained_response(record, model_id, False)

    def _to_constrained_response(self, record: dict, model_id: str,
                                 from_cache: bool) -> LLMResponse:
        self._account(record.get("input_tokens", 0),
                      record.get("output_tokens", 0), from_cache,
                      model_id)
        return LLMResponse(
            text=json.dumps(record["tool_input"]),
            parsed=record["tool_input"],
            input_tokens=record.get("input_tokens", 0),
            output_tokens=record.get("output_tokens", 0),
            model_id=model_id, from_cache=from_cache, raw=record)

    # -- bounded agency ---------------------------------------------

    def run_agent(self, model_id: str, system: str, user: str, *,
                  tools: list[dict], executor,
                  submit_tool_name: str, submit_schema: dict,
                  submit_description: str = "Submit the final result.",
                  max_tool_calls: int = 6,
                  temperature: float = 0.0,
                  max_tokens: int = 2048) -> "AgentRun":
        """Multi-turn tool-use loop with controller-enforced termination.

        The model decides whether to call a tool, which one, with what
        arguments, and whether to call again. ``executor(name, args) ->
        str`` performs the call. No ``toolChoice`` is set while the loop
        runs, so choosing not to search at all remains a possible outcome.

        The budget is enforced here rather than requested in the prompt.
        On exhaustion the run ends and submission is forced.

        Returns an AgentRun carrying the submitted payload and the full
        tool-call trace, which is what the provenance layer records.
        """
        key = self._key_agent(model_id, system, user, tools)
        cached = self._load(key)
        if cached is not None:
            self._account(cached.get("input_tokens", 0),
                          cached.get("output_tokens", 0), True,
                          model_id)
            return AgentRun(**{**cached, "from_cache": True})

        if self.replay_mode:
            raise RuntimeError(
                f"replay_mode: no logged agent run for key {key[:12]}.")

        messages = [{"role": "user", "content": [{"text": user}]}]
        sys_block = [{"text": system}] if system else None
        calls: list[dict] = []
        tok_in = tok_out = 0
        budget_exhausted = False

        while True:
            kwargs = dict(
                modelId=model_id, messages=messages,
                inferenceConfig={"temperature": temperature,
                                 "maxTokens": max_tokens},
                toolConfig={"tools": tools},
            )
            if sys_block:
                kwargs["system"] = sys_block
            resp = self._converse_with_retry(**kwargs)
            u = resp.get("usage", {})
            tok_in += u.get("inputTokens", 0)
            tok_out += u.get("outputTokens", 0)

            content = resp["output"]["message"]["content"]
            messages.append({"role": "assistant", "content": content})

            if resp.get("stopReason") != "tool_use":
                break

            # A turn may contain several toolUse blocks: the model is free
            # to request parallel calls. Bedrock requires a toolResult for
            # every one of them in the following message, so answering
            # only the first fails the next request with
            # "Expected toolResult blocks at messages.N.content".
            tool_blocks = [b for b in content if "toolUse" in b]
            if not tool_blocks:
                break

            results = []
            for b in tool_blocks:
                tu = b["toolUse"]
                if len(calls) >= max_tool_calls:
                    # Budget reached mid-turn. Every outstanding toolUse
                    # still needs an answer, or the conversation is
                    # malformed -- so decline explicitly rather than
                    # dropping the block.
                    budget_exhausted = True
                    result = ("Call budget exhausted. No further searches "
                              "are permitted; submit your answer.")
                else:
                    try:
                        result = executor(tu["name"], tu.get("input", {}))
                    except Exception as e:             # noqa: BLE001
                        result = f"tool error: {type(e).__name__}: {e}"
                    calls.append({"name": tu["name"],
                                  "input": tu.get("input", {}),
                                  "result_excerpt": str(result)[:400]})
                results.append({"toolResult": {
                    "toolUseId": tu["toolUseId"],
                    "content": [{"text": str(result)}]}})

            messages.append({"role": "user", "content": results})
            if budget_exhausted:
                break

        # Forced submission. Only here is toolChoice applied.
        messages.append({"role": "user", "content": [{
            "text": "Submit your final answer now."}]})
        kwargs = dict(
            modelId=model_id, messages=messages,
            inferenceConfig={"temperature": temperature,
                             "maxTokens": max_tokens},
            toolConfig={
                "tools": [self.tool_spec(submit_tool_name,
                                         submit_description, submit_schema)],
                "toolChoice": {"tool": {"name": submit_tool_name}},
            },
        )
        if sys_block:
            kwargs["system"] = sys_block
        final = self._converse_with_retry(**kwargs)
        u = final.get("usage", {})
        tok_in += u.get("inputTokens", 0)
        tok_out += u.get("outputTokens", 0)

        submitted = None
        for b in final["output"]["message"]["content"]:
            if "toolUse" in b and b["toolUse"]["name"] == submit_tool_name:
                submitted = b["toolUse"]["input"]
                break

        record = {
            "mode": "agent",
            "submitted": submitted,
            "tool_calls": calls,
            "n_tool_calls": len(calls),
            "n_model_turns": len(calls) + 2,
            "budget_exhausted": budget_exhausted,
            "input_tokens": tok_in,
            "output_tokens": tok_out,
        }
        self._save(key, record)
        self._account(tok_in, tok_out, False, model_id)
        return AgentRun(**record)

    @staticmethod
    def _error_code(e: Exception) -> str:
        """Botocore error Code, or "" when the exception carries none.

        Botocore sets ``response`` to None on some transport failures, so a
        ``hasattr(e, "response")`` guard is not sufficient: calling
        ``.get`` on it would raise AttributeError inside the retry handler
        instead of retrying.
        """
        resp = getattr(e, "response", None)
        if not isinstance(resp, dict):
            return ""
        err = resp.get("Error")
        return err.get("Code", "") if isinstance(err, dict) else ""

    def _converse_with_retry(self, **kwargs) -> dict:
        last = None
        for attempt in range(self.max_retries):
            try:
                return self._client.converse(**kwargs)
            except Exception as e:                     # noqa: BLE001
                last = e
                ename = type(e).__name__
                code = self._error_code(e)
                if code in _RETRYABLE or ename in {
                        "ReadTimeoutError", "ConnectTimeoutError",
                        "EndpointConnectionError", "ConnectionClosedError"}:
                    time.sleep(min(2 ** attempt + 0.1 * attempt, 30))
                    continue
                raise
        raise RuntimeError(f"exhausted {self.max_retries} retries: {last}")

    def _invoke_with_retry(self, model_id, system, user,
                           temperature, max_tokens) -> dict:
        messages = [{"role": "user", "content": [{"text": user}]}]
        kwargs = dict(
            modelId=model_id,
            messages=messages,
            inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
        )
        if system:
            kwargs["system"] = [{"text": system}]

        last = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.converse(**kwargs)
                text = resp["output"]["message"]["content"][0]["text"]
                usage = resp.get("usage", {})
                return {
                    "text": text,
                    "input_tokens": usage.get("inputTokens", 0),
                    "output_tokens": usage.get("outputTokens", 0),
                }
            except Exception as e:
                last = e
                ename = type(e).__name__
                code = self._error_code(e)
                # botocore socket/connection timeouts have no error Code
                # dict; match by exception class name so they are retried.
                if code in _RETRYABLE or ename in {
                        "ReadTimeoutError", "ConnectTimeoutError",
                        "EndpointConnectionError", "ConnectionClosedError"}:
                    time.sleep(min(2 ** attempt + 0.1 * attempt, 30))
                    continue
                raise
        raise RuntimeError(f"exhausted {self.max_retries} retries: {last}")

    def _to_response(self, record: dict, model_id: str,
                     from_cache: bool, schema: dict | None) -> LLMResponse:
        text = record["text"]
        self._account(record.get("input_tokens", 0),
                      record.get("output_tokens", 0), from_cache,
                      model_id)
        parsed = None
        if schema is not None:
            parsed = self._parse_and_validate(text, schema)
        return LLMResponse(
            text=text, parsed=parsed,
            input_tokens=record.get("input_tokens", 0),
            output_tokens=record.get("output_tokens", 0),
            model_id=model_id, from_cache=from_cache, raw=record)

    # -- structured output ------------------------------------------

    @staticmethod
    def _parse_and_validate(text: str, schema: dict) -> Any:
        """Extract JSON and check required keys are present.

        Lightweight: full JSON-Schema validation is unnecessary for the
        shapes here; the aim is to fail fast and retry.
        """
        s = text.strip()
        if s.startswith("```"):
            s = s.split("```", 2)[1]
            if s.startswith("json"):
                s = s[4:]
            s = s.strip()
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            raise SchemaValidationError(f"not JSON: {e}") from e

        root = schema.get("required", [])
        if isinstance(obj, dict):
            missing = [k for k in root if k not in obj]
            if missing:
                raise SchemaValidationError(f"missing keys: {missing}")
        return obj

    def cost_summary(self, price_per_1k_in: float,
                     price_per_1k_out: float) -> dict:
        """Both figures, always, and labelled.

        ``notional_usd`` is what the run would cost from a cold cache and
        is the number to report as the cost of the experiment.
        ``billed_usd`` is what this invocation actually spent. They differ
        by exactly the replayed calls.
        """
        def price(ti, to):
            return round(ti / 1000 * price_per_1k_in +
                         to / 1000 * price_per_1k_out, 4)

        return {
            # what the experiment costs from cold
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "notional_usd": price(self.total_input_tokens,
                                  self.total_output_tokens),
            # what this invocation actually spent
            "billed_input_tokens": self.billed_input_tokens,
            "billed_output_tokens": self.billed_output_tokens,
            "billed_usd": price(self.billed_input_tokens,
                                self.billed_output_tokens),
            "n_calls_live": self.n_calls_live,
            "n_calls_cached": self.n_calls_cached,
            "cache_hit_rate": (
                round(self.n_calls_cached /
                      (self.n_calls_live + self.n_calls_cached), 4)
                if (self.n_calls_live + self.n_calls_cached) else None),
            # alias of notional_usd
            "est_usd": price(self.total_input_tokens,
                             self.total_output_tokens),
        }
