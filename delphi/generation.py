"""
Conditional autoregressive generation for Delphi-family models.

Works with any model exposing the shared Delphi interface (`Delphi`,
`DelphiMultiStream`): `model(batch) -> (logits_dict, _)`, `predicted_domains`,
`domain_to_int`, `domain_offsets`, `embed._domain_vocab_sizes`, `config.domains`.

Difference from `Delphi.generate`: besides a fixed prefix (at_birth tokens --
sex, genetic PCs, HLA, ...), the caller can pass *pending* real tokens with
real ages (e.g. lifestyle), which are interleaved into the generated timeline
at their true ages instead of being put up-front. Sampling follows the same
competing-exponential-risks process as `Delphi.generate`: every predicted class
draws t ~ Exp(rate=exp(logit)) and the earliest one wins. If the next pending
token comes before the winning sampled time, the pending token is inserted
instead and the step's sample is discarded -- exact under the model's
piecewise-constant hazards (memorylessness), since hazards are re-evaluated
from the updated context at the next step anyway.
"""
from typing import Optional

import torch

from data.dataset import DelphiBatch

PADDING_AGE = -10000.0


@torch.no_grad()
def generate_conditioned(
    model,
    prefix: DelphiBatch,
    pending_global_ids: Optional[torch.Tensor] = None,
    pending_domain_ids: Optional[torch.Tensor] = None,
    pending_ages: Optional[torch.Tensor] = None,
    max_new_tokens: int = 256,
    max_age: torch.Tensor | float = 85 * 365.25,
    termination_domain: Optional[str] = "death",
    block_termination_until: Optional[torch.Tensor] = None,
    apply_no_repeat: bool = True,
) -> DelphiBatch:
    """
    Autoregressively extend `prefix` with sampled events, interleaving pending real tokens.

    Args:
        prefix: conditioning sequence, **right-aligned** (padding at the front, as
            produced by DelphiCollateFn), so position -1 is each row's last real token.
        pending_*: [B, L] real tokens to insert at their own ages, sorted by age
            ascending per row; unused slots must have age = +inf.
        max_age: scalar or [B] tensor (days); a row stops once its next event would
            fall beyond it.
        termination_domain: sampling a token of this domain ends the row (e.g. death).
        block_termination_until: [B] ages (days). While a row's current age is below
            it, `termination_domain` logits are masked (-inf), i.e. the row is
            forced to survive until then. Use it to condition on having been alive
            at the time the real pending tokens were recorded. Note this masks the
            hazard rather than doing exact survival conditioning (which would also
            re-weight the other events towards healthier trajectories).
        apply_no_repeat: mask already-seen tokens of `no_repeat` domains.

    Returns:
        DelphiBatch with the full sequence [B, T_prefix + n_steps]. Positions after a
        row stops are padding (domain=padding, token=0, age=PADDING_AGE).
    """
    device = model.device
    prefix = prefix.to(device)
    global_ids = prefix.global_token_ids.clone()
    domain_ids = prefix.domain_ids.clone()
    ages = prefix.ages.clone().float()
    B = global_ids.shape[0]

    if pending_ages is None:
        pending_global_ids = torch.zeros(B, 1, dtype=torch.long)
        pending_domain_ids = torch.zeros(B, 1, dtype=torch.long)
        pending_ages = torch.full((B, 1), float("inf"))
    pending_global_ids = pending_global_ids.to(device)
    pending_domain_ids = pending_domain_ids.to(device)
    pending_ages = pending_ages.to(device).float()
    L = pending_ages.shape[1]
    ptr = torch.zeros(B, dtype=torch.long, device=device)
    rows = torch.arange(B, device=device)

    max_age = torch.as_tensor(max_age, dtype=torch.float32, device=device).expand(B)
    padding_int = model.domain_to_int["padding"]
    term_int = model.domain_to_int[termination_domain] if termination_domain else None

    # Layout of the concatenated predicted-domain softmax.
    predicted = list(model.predicted_domains)
    vocab_sizes = [model.embed._domain_vocab_sizes[d] for d in predicted]
    boundaries = torch.tensor([0] + list(torch.tensor(vocab_sizes).cumsum(0).tolist()), device=device)
    slot_domain_int = torch.tensor([model.domain_to_int[d] for d in predicted], device=device)
    slot_offset = torch.tensor([model.domain_offsets[model.domain_to_int[d]] for d in predicted], device=device)
    V = int(boundaries[-1])

    # Global id <-> softmax column, for the running no-repeat / termination masks.
    def _cols_of(global_ids_flat, domain_ids_flat):
        col = torch.full_like(global_ids_flat, -1)
        for k, d in enumerate(predicted):
            m = domain_ids_flat == model.domain_to_int[d]
            col[m] = global_ids_flat[m] - slot_offset[k] + boundaries[k]
        return col

    no_repeat_cols = torch.zeros(V, dtype=torch.bool, device=device)
    term_cols = torch.zeros(V, dtype=torch.bool, device=device)
    for k, d in enumerate(predicted):
        if apply_no_repeat and getattr(model.config.domains[d], "no_repeat", False):
            no_repeat_cols[boundaries[k]:boundaries[k + 1]] = True
        if d == termination_domain:
            term_cols[boundaries[k]:boundaries[k + 1]] = True

    seen = torch.zeros(B, V, dtype=torch.bool, device=device)
    col0 = _cols_of(global_ids.flatten(), domain_ids.flatten()).view(B, -1)
    for b, c in (col0 >= 0).nonzero(as_tuple=False).tolist():
        seen[b, col0[b, c]] = True
    seen &= no_repeat_cols

    terminated = ages.max(dim=1).values > max_age
    if term_int is not None:
        terminated |= (domain_ids == term_int).any(dim=1)

    for _ in range(max_new_tokens):
        if terminated.all():
            break

        cur = DelphiBatch(
            global_token_ids=global_ids, domain_ids=domain_ids, ages=ages,
            subject_ids=prefix.subject_ids,
            continuous_data=prefix.continuous_data,
            continuous_positions=prefix.continuous_positions,
            eval_mask=None,
        )
        logits_dict, _ = model(cur)
        logits = torch.cat([logits_dict[d][:, -1, :] for d in predicted], dim=-1).float()  # [B, V]
        logits = logits.masked_fill(seen, float("-inf"))

        cur_age = ages[:, -1]
        if block_termination_until is not None:
            blocked = (cur_age < block_termination_until.to(device)).unsqueeze(1) & term_cols.unsqueeze(0)
            logits = logits.masked_fill(blocked, float("-inf"))

        u = torch.rand_like(logits).clamp_min(1e-12)
        t_cand = -torch.exp(-logits) * u.log()
        t_next, winner = t_cand.min(dim=-1)
        slot = torch.bucketize(winner, boundaries[1:], right=True)
        s_global = slot_offset[slot] + (winner - boundaries[slot])
        s_domain = slot_domain_int[slot]
        s_age = cur_age + t_next

        # Pending real token comes first -> insert it instead of the sampled event.
        has_pending = ptr < L
        p_idx = ptr.clamp(max=L - 1)
        p_age = torch.where(has_pending, pending_ages[rows, p_idx], torch.full_like(s_age, float("inf")))
        use_pending = has_pending & (p_age <= s_age) & (p_age <= max_age)

        new_global = torch.where(use_pending, pending_global_ids[rows, p_idx], s_global)
        new_domain = torch.where(use_pending, pending_domain_ids[rows, p_idx], s_domain)
        new_age = torch.where(use_pending, p_age, s_age)
        ptr = ptr + use_pending.long()

        should_pad = terminated | (new_age > max_age)
        new_global = torch.where(should_pad, torch.zeros_like(new_global), new_global)
        new_domain = torch.where(should_pad, torch.full_like(new_domain, padding_int), new_domain)
        new_age = torch.where(should_pad, torch.full_like(new_age, PADDING_AGE), new_age)

        global_ids = torch.cat([global_ids, new_global.unsqueeze(1)], dim=1)
        domain_ids = torch.cat([domain_ids, new_domain.unsqueeze(1)], dim=1)
        ages = torch.cat([ages, new_age.unsqueeze(1)], dim=1)

        sampled_real = ~should_pad & ~use_pending
        seen[rows[sampled_real], winner[sampled_real]] |= no_repeat_cols[winner[sampled_real]]

        just_died = (new_domain == term_int) if term_int is not None else torch.zeros_like(should_pad)
        terminated = terminated | should_pad | just_died

    return DelphiBatch(
        global_token_ids=global_ids, domain_ids=domain_ids, ages=ages,
        subject_ids=prefix.subject_ids,
        continuous_data=prefix.continuous_data,
        continuous_positions=prefix.continuous_positions,
        eval_mask=None,
    )
