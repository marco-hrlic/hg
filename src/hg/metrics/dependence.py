"""Error-dependence analysis (protocol section 9).

The protocol is explicit that the naive independence null

    P(all wrong) ~ prod_j p_j

"is useful descriptively but is not a credible model of real ensemble errors",
and that the real question is whether observed clean consensus failure

    remains unusually large after accounting for individual model quality
    and ordinary correlated difficulty.

So this module provides three layers, weakest to strongest:

1. `independence_null` -- the product of marginal error rates. Kept because it
   is the number people expect to see, and because the gap between it and the
   observed rate is a useful headline. It is not the inferential baseline.
2. `pairwise_error_correlation` -- rho_jk = Corr(E_j, E_k). Descriptive, but it
   localizes *which* pairs are dependent, which matters for telling lineage
   sharing from data sharing.
3. `fit_item_difficulty` + `excess_consensus` -- the model that actually
   answers the question:

       logit P(E_ij = 1) = alpha_j + u_i

   a one-parameter-per-item, one-per-member logistic model (a Rasch model).
   Conditional on u_i the members are independent, so the model's predicted
   consensus rate is exactly "what ordinary correlated difficulty plus unequal
   model quality would produce". `excess_consensus` compares the observed rate
   to that prediction. Excess above this baseline is the part that shared
   difficulty does not explain -- the candidate systematic residual.

Fitting notes
-------------
The joint MLE of a Rasch model is not consistent as items grow with a fixed
member count, and items where every member is right or every member is wrong
push u_i to +-infinity. Both are handled by ridge penalties on alpha and u,
with the penalty strength reported in the fit so a reader can see how much
regularization is doing. This is a diagnostic baseline, not a publication-grade
random-effects fit; `notes` on the returned object says so, and a proper
mixed-effects fit (integrating u_i out) is the natural upgrade once the design
is settled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .matrix import ErrorMatrix


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def independence_null(matrix: ErrorMatrix, status: str | Sequence[str] | None = None) -> float:
    """prod_j p_j over the selected item subset.

    Descriptive only. Treating this as the null is what makes ordinary shared
    difficulty look like a mysterious common-mode failure.
    """
    mask = matrix.item_mask(status)
    if not mask.any():
        return float("nan")
    p = matrix.errors[mask].mean(axis=0)
    return float(np.prod(p))


def pairwise_error_correlation(matrix: ErrorMatrix) -> np.ndarray:
    """rho_jk = Corr(E_j, E_k), an (n_members, n_members) array.

    Members with zero error variance (never wrong, or always wrong) yield NaN
    rather than a spurious 0: an undefined correlation is not independence.
    """
    e = matrix.errors.astype(float)
    n_m = e.shape[1]
    out = np.full((n_m, n_m), np.nan)
    sd = e.std(axis=0)
    centered = e - e.mean(axis=0)
    for j in range(n_m):
        for k in range(n_m):
            if sd[j] == 0 or sd[k] == 0:
                continue
            out[j, k] = float(
                (centered[:, j] * centered[:, k]).mean() / (sd[j] * sd[k])
            )
    return out


def mean_pairwise_correlation(matrix: ErrorMatrix) -> float:
    rho = pairwise_error_correlation(matrix)
    n = rho.shape[0]
    off = rho[~np.eye(n, dtype=bool)]
    if np.all(np.isnan(off)):
        return float("nan")
    return float(np.nanmean(off))


def within_between_correlation(matrix: ErrorMatrix) -> dict[str, float]:
    """Mean rho_jk split by whether the pair shares a model family.

    The decomposition HD and DS are designed to produce: within-family
    correlation that exceeds between-family correlation is representation-level
    dependence, which is what family diversity can remove.
    """
    rho = pairwise_error_correlation(matrix)
    fams = matrix.families
    within: list[float] = []
    between: list[float] = []
    for j in range(len(fams)):
        for k in range(j + 1, len(fams)):
            v = rho[j, k]
            if np.isnan(v):
                continue
            (within if fams[j] == fams[k] else between).append(float(v))
    return {
        "within_family": float(np.mean(within)) if within else float("nan"),
        "between_family": float(np.mean(between)) if between else float("nan"),
        "n_within_pairs": len(within),
        "n_between_pairs": len(between),
    }


@dataclass(slots=True)
class DependenceFit:
    """Fitted logit P(E_ij=1) = alpha_j + u_i."""

    alpha: np.ndarray  # (n_members,) member severity; higher = more error-prone
    u: np.ndarray  # (n_items,) item difficulty
    member_ids: list[str]
    item_ids: list[str]
    n_iter: int
    converged: bool
    loglik: float
    ridge_alpha: float
    ridge_u: float
    notes: str = ""

    def error_probabilities(self) -> np.ndarray:
        """(n_items, n_members) fitted P(E_ij = 1)."""
        return _sigmoid(self.u[:, None] + self.alpha[None, :])

    def predicted_consensus(self, item_mask: np.ndarray | None = None) -> float:
        """E[P(all wrong)] under conditional independence given u_i.

        This is the baseline the observed consensus rate should be judged
        against: it already contains unequal model quality (alpha_j) and
        ordinary correlated difficulty (u_i).
        """
        p = self.error_probabilities()
        if item_mask is not None:
            p = p[np.asarray(item_mask, dtype=bool)]
        if p.shape[0] == 0:
            return float("nan")
        return float(np.prod(p, axis=1).mean())

    def predicted_disagreement(self, item_mask: np.ndarray | None = None) -> float:
        """E[1 - P(all wrong) - P(all right)] under the same assumption.

        Only equals the disagreement rate for binary tasks, where "not all
        wrong and not all right" is exactly "the members split".
        """
        p = self.error_probabilities()
        if item_mask is not None:
            p = p[np.asarray(item_mask, dtype=bool)]
        if p.shape[0] == 0:
            return float("nan")
        all_wrong = np.prod(p, axis=1)
        all_right = np.prod(1.0 - p, axis=1)
        return float((1.0 - all_wrong - all_right).mean())

    def to_dict(self) -> dict[str, object]:
        return {
            "member_severity": {
                m: float(a) for m, a in zip(self.member_ids, self.alpha)
            },
            "item_difficulty_sd": float(np.std(self.u)),
            "item_difficulty_mean": float(np.mean(self.u)),
            "n_iter": self.n_iter,
            "converged": self.converged,
            "loglik": self.loglik,
            "ridge_alpha": self.ridge_alpha,
            "ridge_u": self.ridge_u,
            "notes": self.notes,
        }


def fit_item_difficulty(
    matrix: ErrorMatrix,
    *,
    ridge_alpha: float = 1e-3,
    ridge_u: float = 1e-2,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> DependenceFit:
    """Fit alpha_j and u_i by penalized IRLS-style coordinate ascent.

    Alternating Newton steps: with the other block fixed, each block's
    log-likelihood is concave and separable, so a single Newton update per
    coordinate is cheap and stable. `ridge_u` is what keeps items with a
    degenerate response pattern (all-wrong / all-right) finite.

    Identifiability: alpha and u trade off by a constant, so u is centered at
    every iteration and alpha absorbs the level.
    """
    e = matrix.errors.astype(float)
    n_i, n_m = e.shape
    if n_i == 0 or n_m == 0:
        raise ValueError("cannot fit a dependence model on an empty matrix")

    # Start from marginal log-odds so the first step is already close.
    overall = np.clip(e.mean(), 1e-3, 1 - 1e-3)
    alpha = np.array(
        [
            math.log(
                np.clip(e[:, j].mean(), 1e-3, 1 - 1e-3)
                / (1 - np.clip(e[:, j].mean(), 1e-3, 1 - 1e-3))
            )
            for j in range(n_m)
        ]
    )
    u = np.zeros(n_i)

    def loglik(alpha_: np.ndarray, u_: np.ndarray) -> float:
        z = u_[:, None] + alpha_[None, :]
        # log sigma(z) = -log1p(exp(-z)), stable form
        ll = np.where(z >= 0, -np.log1p(np.exp(-z)), z - np.log1p(np.exp(z)))
        ll_not = np.where(-z >= 0, -np.log1p(np.exp(z)), -z - np.log1p(np.exp(-z)))
        total = float((e * ll + (1 - e) * ll_not).sum())
        total -= 0.5 * ridge_alpha * float(np.dot(alpha_, alpha_))
        total -= 0.5 * ridge_u * float(np.dot(u_, u_))
        return total

    prev = loglik(alpha, u)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        # --- alpha block (one Newton step per member) ---
        p = _sigmoid(u[:, None] + alpha[None, :])
        grad = (e - p).sum(axis=0) - ridge_alpha * alpha
        hess = -(p * (1 - p)).sum(axis=0) - ridge_alpha
        alpha = alpha - grad / np.where(hess == 0, -1e-8, hess)

        # --- u block ---
        p = _sigmoid(u[:, None] + alpha[None, :])
        grad_u = (e - p).sum(axis=1) - ridge_u * u
        hess_u = -(p * (1 - p)).sum(axis=1) - ridge_u
        u = u - grad_u / np.where(hess_u == 0, -1e-8, hess_u)

        # Re-center u; alpha absorbs the shift (alpha + u is what is identified).
        shift = float(u.mean())
        u = u - shift
        alpha = alpha + shift

        cur = loglik(alpha, u)
        if abs(cur - prev) < tol * (1.0 + abs(prev)):
            converged = True
            prev = cur
            break
        prev = cur

    return DependenceFit(
        alpha=alpha,
        u=u,
        member_ids=list(matrix.member_ids),
        item_ids=list(matrix.item_ids),
        n_iter=it,
        converged=converged,
        loglik=prev,
        ridge_alpha=ridge_alpha,
        ridge_u=ridge_u,
        notes=(
            "Penalized joint MLE of a Rasch-style model. Diagnostic baseline: "
            "joint MLE is not consistent as items grow at fixed ensemble size, "
            "and ridge_u keeps degenerate response patterns finite. Upgrade to a "
            "mixed-effects fit integrating u_i out before publishing an effect."
        ),
    )


@dataclass(slots=True)
class RandomEffectsFit:
    """Hierarchical error model with u_i integrated out (protocol section 12).

        E_ij ~ Bernoulli(sigma(alpha_j + u_i)),   u_i ~ N(0, sigma_u^2)

    This is the baseline `excess_consensus` should be judged against, and it is
    the reason `fit_item_difficulty` alone is not enough. The joint MLE
    estimates one u_i per item from only E observations, so the estimated
    spread of u is inflated by its own sampling noise -- and since predicted
    consensus is convex in u, an inflated spread *over*-predicts consensus
    failure. A real residual would then be masked, in the exact direction that
    matters. Marginalizing over u_i estimates sigma_u once, from all items, and
    removes that bias.

    The integral is evaluated by Gauss-Hermite quadrature, which is exact for
    smooth integrands at modest node counts and needs no sampling.
    """

    alpha: np.ndarray  # (n_members,) member severity
    sigma_u: float  # SD of the shared item-difficulty factor
    member_ids: list[str]
    n_iter: int
    converged: bool
    loglik: float
    n_nodes: int = 40

    def _nodes(self) -> tuple[np.ndarray, np.ndarray]:
        return _gh_nodes(self.n_nodes)

    def error_probabilities(self) -> np.ndarray:
        """(n_nodes, n_members) P(E=1 | u = sigma_u * t_q)."""
        t, _ = self._nodes()
        return _sigmoid(self.sigma_u * t[:, None] + self.alpha[None, :])

    def predicted_consensus(self, members: Sequence[int] | None = None) -> float:
        """E_u[prod_j P(E_ij = 1 | u)] -- the model's consensus-wrong rate."""
        t, w = self._nodes()
        p = self.error_probabilities()
        if members is not None:
            p = p[:, list(members)]
        return float(np.sum(w * np.prod(p, axis=1)))

    def predicted_all_right(self, members: Sequence[int] | None = None) -> float:
        t, w = self._nodes()
        p = self.error_probabilities()
        if members is not None:
            p = p[:, list(members)]
        return float(np.sum(w * np.prod(1.0 - p, axis=1)))

    def predicted_disagreement(self) -> float:
        return 1.0 - self.predicted_consensus() - self.predicted_all_right()

    def predicted_survival(
        self, member_subsets: Sequence[Sequence[int]]
    ) -> float:
        """Mean predicted consensus-wrong rate over the given member subsets.

        Lets the fitted model produce its own survival curve, evaluated over
        exactly the subsets the observed curve averaged over. That is the
        comparison the experiment needs, because under this model -- one shared
        difficulty factor, conditional independence given it -- C_k must fall to
        zero as k grows. It cannot produce a floor. So a *growing* gap between
        the observed curve and this one is the signature of something the shared
        difficulty axis does not explain, which is what an information-limited
        core would look like.
        """
        if not member_subsets:
            return float("nan")
        return float(
            np.mean([self.predicted_consensus(list(s)) for s in member_subsets])
        )

    def implied_pairwise_rho(self) -> float:
        """Mean rho_jk implied by a single shared factor of width sigma_u.

        Useful as a sanity check: if the observed mean rho is far above this,
        one shared difficulty axis does not account for the dependence, and the
        excess is not an artifact of the fit.
        """
        t, w = self._nodes()
        p = self.error_probabilities()
        n_m = p.shape[1]
        if n_m < 2:
            return float("nan")
        marg = np.sum(w[:, None] * p, axis=0)
        rhos: list[float] = []
        for j in range(n_m):
            for k in range(j + 1, n_m):
                joint = float(np.sum(w * p[:, j] * p[:, k]))
                cov = joint - marg[j] * marg[k]
                denom = math.sqrt(
                    marg[j] * (1 - marg[j]) * marg[k] * (1 - marg[k])
                )
                if denom > 0:
                    rhos.append(cov / denom)
        return float(np.mean(rhos)) if rhos else float("nan")

    def to_dict(self) -> dict[str, object]:
        return {
            "member_severity": {m: float(a) for m, a in zip(self.member_ids, self.alpha)},
            "sigma_u": self.sigma_u,
            "n_iter": self.n_iter,
            "converged": self.converged,
            "loglik": self.loglik,
            "n_nodes": self.n_nodes,
            "implied_mean_rho": self.implied_pairwise_rho(),
            "notes": (
                "Marginal MLE, u_i integrated out by Gauss-Hermite quadrature. "
                "Assumes a single shared item-difficulty factor and conditional "
                "independence given it; that is the null the residual is measured "
                "against, not a claim about the true error structure."
            ),
        }


def _gh_nodes(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes/weights for E[f(Z)], Z ~ N(0, 1). Weights sum to 1."""
    t, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / w.sum()
    return t, w


def fit_random_effects(
    matrix: ErrorMatrix,
    *,
    n_nodes: int = 40,
    max_iter: int = 500,
    tol: float = 1e-9,
    step0: float = 0.5,
) -> RandomEffectsFit:
    """Marginal MLE of (alpha, sigma_u) by gradient ascent with backtracking.

    Analytic gradients (no finite differences), m+1 parameters, so plain
    ascent with a backtracking line search converges in well under a second on
    a 5000 x 5 matrix. sigma_u is parameterized as log sigma_u to keep it
    positive without a constrained optimizer.
    """
    e = matrix.errors.astype(float)
    n_i, n_m = e.shape
    if n_i == 0 or n_m == 0:
        raise ValueError("cannot fit a dependence model on an empty matrix")
    t, w = _gh_nodes(n_nodes)

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, float]:
        return theta[:n_m], float(np.exp(theta[n_m]))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        alpha, sigma = unpack(theta)
        # p: (n_nodes, n_members)
        p = _sigmoid(sigma * t[:, None] + alpha[None, :])
        p = np.clip(p, 1e-12, 1 - 1e-12)
        # log f_iq = sum_j e_ij log p_qj + (1-e_ij) log(1-p_qj)
        log_f = e @ np.log(p).T + (1.0 - e) @ np.log1p(-p).T  # (n_items, n_nodes)
        mx = log_f.max(axis=1, keepdims=True)
        f = np.exp(log_f - mx)
        wf = f * w[None, :]
        li = wf.sum(axis=1)  # scaled marginal likelihood per item
        ll = float((np.log(li) + mx[:, 0]).sum())

        # posterior weights over nodes, per item
        post = wf / li[:, None]  # (n_items, n_nodes)
        resid = e[:, None, :] - p[None, :, :]  # (n_items, n_nodes, n_members)
        g_alpha = np.einsum("iq,iqj->j", post, resid)
        g_sigma_raw = np.einsum("iq,q,iqj->", post, t, resid)
        # chain rule for log-parameterization
        g_log_sigma = g_sigma_raw * sigma
        grad = np.concatenate([g_alpha, [g_log_sigma]])
        return ll, grad

    # Init: marginal log-odds per member, sigma_u from the observed mean
    # pairwise correlation so the search starts in the right neighbourhood.
    marg = np.clip(e.mean(axis=0), 1e-3, 1 - 1e-3)
    alpha0 = np.log(marg / (1 - marg))
    rho0 = mean_pairwise_correlation(matrix)
    sigma0 = 1.0 if math.isnan(rho0) or rho0 <= 0 else max(0.2, min(3.0, 2.5 * rho0))
    theta = np.concatenate([alpha0, [math.log(sigma0)]])

    ll, grad = objective(theta)
    step = step0
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        gnorm = float(np.linalg.norm(grad))
        if gnorm < 1e-7:
            converged = True
            break
        # backtracking ascent
        accepted = False
        for _ in range(40):
            cand = theta + step * grad / max(gnorm, 1e-12)
            ll_c, grad_c = objective(cand)
            if ll_c > ll:
                improvement = ll_c - ll
                theta, ll, grad = cand, ll_c, grad_c
                step = min(step * 1.3, 5.0)
                accepted = True
                if improvement < tol * (1.0 + abs(ll)):
                    converged = True
                break
            step *= 0.5
        if not accepted or converged:
            converged = converged or not accepted
            break

    alpha, sigma_u = unpack(theta)
    return RandomEffectsFit(
        alpha=alpha,
        sigma_u=sigma_u,
        member_ids=list(matrix.member_ids),
        n_iter=it,
        converged=converged,
        loglik=ll,
        n_nodes=n_nodes,
    )


def excess_consensus(
    matrix: ErrorMatrix,
    fit: DependenceFit | None = None,
    *,
    clean_only: bool = True,
    re_fit: "RandomEffectsFit | None" = None,
) -> dict[str, float]:
    """Observed consensus failure against three progressively fairer baselines.

    `excess_over_hierarchical` is the protocol's actual quantity of interest:
    how much clean consensus failure survives after unequal model quality
    (alpha_j) and ordinary correlated item difficulty (u_i) are accounted for.
    It uses the marginal fit, not the joint one, because the joint fit's
    per-item difficulty estimates are noisy at ensemble size 5 and their noise
    inflates predicted consensus -- masking exactly the residual being looked
    for.

    The other two are reported for context and are not the inferential
    baseline:

    * `independence_null` -- prod_j p_j. What people expect to see; ignores
      shared difficulty entirely, so any correlated difficulty reads as a
      mysterious common-mode failure.
    * `difficulty_model_joint` -- the joint Rasch fit's prediction. Kept
      because comparing it to the marginal fit shows how much the overfitting
      bias is worth on this matrix.

    Interpretation caveat, deliberately kept next to the number: the
    hierarchical null absorbs *any* dependence that behaves like a single
    shared item-level factor, and genuine shared representational bias behaves
    exactly like that. So a near-zero excess does not prove the residual is
    benign -- it says one shared difficulty axis plus unequal skill reproduces
    it. Distinguishing "one shared difficulty axis" from "an information-limited
    core" is the job of the diversity axis in `hg.diversity.survival`, not of
    this model.
    """
    mask = matrix.clean_mask if clean_only else None
    sub = matrix.select_items(mask) if mask is not None else matrix
    n_sub = sub.n_items
    if n_sub == 0:
        return {
            "observed": float("nan"),
            "independence_null": float("nan"),
            "difficulty_model_joint": float("nan"),
            "hierarchical_model": float("nan"),
            "excess_over_independence": float("nan"),
            "excess_over_difficulty": float("nan"),
            "excess_over_hierarchical": float("nan"),
            "ratio_over_hierarchical": float("nan"),
            "sigma_u": float("nan"),
            "n_items": 0,
        }

    observed = float(sub.all_wrong().mean())
    naive = independence_null(matrix, "unanimous" if clean_only else None)

    # Both models are fitted on the same subset the observed rate is measured
    # on. Fitting on all items and evaluating on the clean subset would mix
    # contested-item difficulty into the baseline.
    if fit is None:
        fit = fit_item_difficulty(sub)
    joint_pred = fit.predicted_consensus()
    if re_fit is None:
        re_fit = fit_random_effects(sub)
    hier_pred = re_fit.predicted_consensus()

    return {
        "observed": observed,
        "independence_null": naive,
        "difficulty_model_joint": joint_pred,
        "hierarchical_model": hier_pred,
        "excess_over_independence": observed - naive,
        "excess_over_difficulty": observed - joint_pred,
        "excess_over_hierarchical": observed - hier_pred,
        "ratio_over_hierarchical": (
            observed / hier_pred
            if hier_pred and not math.isnan(hier_pred) and hier_pred > 0
            else float("inf") if observed > 0 else float("nan")
        ),
        "sigma_u": re_fit.sigma_u,
        "hierarchical_converged": re_fit.converged,
        "n_items": n_sub,
    }


def dependence_report(matrix: ErrorMatrix) -> dict[str, object]:
    """Everything section 9 asks for, in one block."""
    joint = fit_item_difficulty(matrix)
    hier = fit_random_effects(matrix)
    rho = pairwise_error_correlation(matrix)
    observed_rho = mean_pairwise_correlation(matrix)
    implied_rho = hier.implied_pairwise_rho()
    return {
        "mean_pairwise_rho": observed_rho,
        "rho_matrix": [[None if np.isnan(v) else float(v) for v in row] for row in rho],
        "member_ids": list(matrix.member_ids),
        "within_between": within_between_correlation(matrix),
        "joint_fit": joint.to_dict(),
        "hierarchical_fit": hier.to_dict(),
        # If observed rho materially exceeds what a single shared difficulty
        # factor implies, the dependence is not one axis of item difficulty and
        # any excess consensus failure is not an artifact of the fit.
        "rho_excess_over_single_factor": (
            float("nan")
            if math.isnan(observed_rho) or math.isnan(implied_rho)
            else observed_rho - implied_rho
        ),
        "excess_clean": excess_consensus(matrix, clean_only=True),
        "excess_all": excess_consensus(matrix, clean_only=False),
        "predicted_disagreement_clean": hier.predicted_disagreement(),
    }
