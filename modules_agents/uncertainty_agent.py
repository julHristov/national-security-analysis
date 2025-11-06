# modules_agents/uncertainty_agent.py
"""
UncertaintyQuantifier (basic Bayesian fusion):
- Комбинира независими evidence sources и дава постериорно доверие.
- Това е лек, интерпретируем слой (не е full Bayesian MCMC).
"""

import math


class UncertaintyAgent:
    def __init__(self, prior: float = 0.5):
        self.prior = prior  # prior belief that a scenario is valid

    def combine_evidence(self, evidences: dict):
        """
        evidences: dict of {source_name: score_in_[0,1]}
        We compute likelihood as product of scores (simplified).
        posterior = (likelihood * prior) / normalization
        For stability we use log-odds aggregation.
        """
        # convert to log-odds
        eps = 1e-9
        log_odds = math.log(self.prior / (1 - self.prior + eps) + eps)
        for name, s in evidences.items():
            s = max(min(s, 0.999999), 1e-9)
            # treat s as likelihood ratio proxy: log(s/(1-s))
            log_odds += math.log(s / (1 - s + eps))
        post = 1 / (1 + math.exp(-log_odds))
        # estimate simple uncertainty = 1 - max_evidence_variance proxy
        vals = list(evidences.values()) if evidences else [self.prior]
        var = (sum((v - (sum(vals) / len(vals))) ** 2 for v in vals) / max(1, len(vals)))
        uncertainty = min(1.0, var * 4)  # scaling heuristic
        return {"posterior": post, "uncertainty": uncertainty, "evidence": evidences}
