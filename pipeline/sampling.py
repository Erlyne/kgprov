"""Per-key sampling for the expensive arms of the 2x2.

The strong arm costs 3x the weak arm per token (runner._PRICE), so only
the strong cells are sampled. The primary (weak) arm feeds the main
results and is never sampled.

Admission is a hash of (seed, key), not a draw from a random stream.
Two properties follow:

  * Order-independent. The same key is admitted or not regardless of
    which document it turned up in, or how many keys preceded it. A
    resumed run therefore samples the identical subset, and the
    content-addressed response log (partitioned per run id) replays it.
  * Shared across cells. The bounded and unbounded strong cells key off
    the same decision, so the strong row of the 2x2 sits on one
    population rather than two overlapping ones. Escalation is triggered
    by Tier 1 failure, so skipping an arm's Tier 1 for a key skips its
    Tier 2 for that key by construction.

The cap is a cost limit, not the sampler. It is the one order-dependent
part: if it binds, the realised sample is a document-order prefix of the
rate-selected set rather than the whole of it. The manifest records
whether it bound.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

#: Dedup keys per document, used *only* to turn a target sample size into
#: an admission rate. Measured at 2,291 keys over the 500-document dev
#: split. An inaccurate estimate costs sample size, never correctness: the
#: realised count is logged and is the number that gets reported.
DEFAULT_KEYS_PER_DOC = 4.6

#: Nothing is sampled unless asked. A `None` sampler means census, and a
#: sampler with no arms is the same thing said differently.
CENSUS = None


def _unit(seed: int, *parts: object) -> float:
    """Map (seed, key) into [0, 1) uniformly.

    A hash rather than an RNG stream: `random.Random(seed).random()` in a
    loop returns a different value for the same key depending on how many
    keys came before it, which is the order dependence this module
    avoids.
    """
    payload = "|".join([str(seed)] + ["" if p is None else str(p)
                                      for p in parts])
    h = hashlib.sha1(payload.encode("utf-8")).digest()[:8]
    return int.from_bytes(h, "big") / float(1 << 64)


def rate_for_target(target: int, n_docs: int,
                    keys_per_doc: float = DEFAULT_KEYS_PER_DOC) -> float:
    """Admission rate that lands ~`target` keys over `n_docs` documents."""
    projected = max(1.0, n_docs * keys_per_doc)
    return min(1.0, target / projected)


@dataclass
class ArmSampler:
    """Decides, per (arm, dedup key), whether that arm runs at all.

    `arms` are the sampled arms. Every other arm is a census -- notably
    the primary arm, whose decision propagates downstream to
    `mention.cui`: sampling it would leave mentions with no grounding at
    all, which is a different experiment.
    """

    arms: frozenset[str]
    rate: float
    seed: int = 0
    #: Hard ceiling on distinct keys admitted per arm. 0 = no ceiling.
    cap: int = 0
    _admitted: dict[str, set[tuple]] = field(default_factory=dict,
                                             init=False, repr=False)
    _rejected: dict[str, set[tuple]] = field(default_factory=dict,
                                             init=False, repr=False)
    _cap_bound: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.arms = frozenset(self.arms)
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {self.rate}")
        if self.cap < 0:
            raise ValueError(f"cap must be >= 0, got {self.cap}")
        for arm in self.arms:
            self._admitted[arm] = set()
            self._rejected[arm] = set()

    def samples(self, arm: str) -> bool:
        """Is this arm sampled, or a census?"""
        return arm in self.arms

    def admits(self, arm: str, key: tuple) -> bool:
        """Does `arm` run for this dedup key?

        Idempotent: a key recurring in a later document gets the same
        answer, and is not charged twice against the cap.
        """
        if arm not in self.arms:
            return True
        key = tuple(key)
        if key in self._admitted[arm]:
            return True
        if key in self._rejected[arm]:
            return False
        if _unit(self.seed, arm, *key) >= self.rate:
            self._rejected[arm].add(key)
            return False
        if self.cap and len(self._admitted[arm]) >= self.cap:
            # Rate-selected but over budget. Recorded as bound so the
            # manifest can say the sample is a prefix, not the whole set.
            self._cap_bound.add(arm)
            self._rejected[arm].add(key)
            return False
        self._admitted[arm].add(key)
        return True

    def admitted(self, arm: str) -> set[tuple]:
        return set(self._admitted.get(arm, ()))

    def summary(self) -> dict:
        """What to put in the run manifest."""
        return {
            "sampled_arms": sorted(self.arms),
            "seed": self.seed,
            "rate": round(self.rate, 6),
            "cap": self.cap or None,
            "keys_admitted": {a: len(self._admitted[a])
                              for a in sorted(self.arms)},
            "keys_rejected": {a: len(self._rejected[a])
                              for a in sorted(self.arms)},
            # If this is true for an arm, the rate did not do the
            # selecting on its own and the sample is order-dependent.
            "cap_binding": {a: (a in self._cap_bound)
                            for a in sorted(self.arms)},
        }
