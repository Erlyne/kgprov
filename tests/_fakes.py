"""Fake Bedrock provider, shared by the constrained-decoding and Tier 2
tests. Records every request so assertions can check what was actually
sent, and replays a scripted response list.
"""

from __future__ import annotations


class FakeBedrock:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def converse(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("fake provider ran out of responses")
        return self.responses.pop(0)


def tool_use(name, inp, stop="tool_use", tokens=(10, 5)):
    return {
        "output": {"message": {"content": [
            {"toolUse": {"toolUseId": f"tu_{name}", "name": name,
                         "input": inp}}]}},
        "stopReason": stop,
        "usage": {"inputTokens": tokens[0], "outputTokens": tokens[1]},
    }


def text(msg, tokens=(10, 5)):
    return {
        "output": {"message": {"content": [{"text": msg}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": tokens[0], "outputTokens": tokens[1]},
    }
