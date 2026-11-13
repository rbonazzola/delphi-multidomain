# Attention schemes

An attention scheme sets which tokens each token can attend to. You set it with `--attention_scheme`.
The code is in `delphi/model.py` (`AttentionMaskBuilder`) and `train.py` (`parse_attention_scheme`).

## Syntax

A scheme is a comma-separated list of rules. Each rule has the form `<domains>:<type>`.

- `<domains>` is one domain name, a group in brackets (`[d1,d2,...]`), or an alias (see below).
- `<type>` is one of:

| Type | A token at age *a* can attend to a token at age *b* when... |
|------|-------------------------------------------------------------|
| `causal(mask_ties=True)` | *b* < *a* (strictly earlier) |
| `causal` | *b* ≤ *a* (earlier or same age, including itself) |
| `bidirectional` | always (ages are ignored) |

Both tokens must belong to the domains of the rule. A rule never connects a domain inside the rule to a domain outside it.

Example:

```
"[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)"
```

## How the mask is built

1. The mask starts empty: no token can attend to any token.
2. Each rule **adds** the pairs it allows. Rules never remove pairs. The final mask is the union of all rules, so the order of the rules has no effect.
3. Padding tokens (age < 0) are removed from all pairs.
4. A token with no allowed target attends to itself only (fallback to prevent NaN).

The mask uses **token ages**, not the order of the domains. For example, with `all:causal(mask_ties=True)`, a disease token at age 50 can attend to a lifestyle token at age 40, and a lifestyle token at age 60 can attend to a disease token at age 50.

## Aliases

| Alias | Expands to |
|-------|-----------|
| `all` | Every domain in the model, including the `padding` and `no_event` pseudo-domains |
| `at_birth` | Every domain with `at_birth: true` in its config (resolved when the model is built) |
| a `group` or `parent` name | Every domain with that `group` in its config; for a domain with no `group`, its `parent`. For example, `hla_alleles` expands to `hla_a`, `hla_b`, ... in `config/domain_config_per_hla_locus.yaml`. See [domain_config.md](domain_config.md) |

> ⚠️ **No-event tokens.** No-event tokens belong to the `no_event` domain. `all` includes it. An explicit list such as `[sex,diseases,lifestyle,death]` does not include it. If a scheme does not mention `no_event` (directly or through `all`), no-event tokens attend only to themselves and no token attends to them. Use `all` for the causal rule, or add `no_event` to the list.

A domain that is in no rule has the same problem: its tokens attend only to themselves.

## Worked example

Take one subject with these tokens:

| Token | Domain | Age |
|-------|--------|-----|
| H | hla_alleles | 0 |
| S | sex | 0 |
| L | lifestyle | 40 |
| D1 | diseases | 50 |
| D2 | diseases | 50 |
| N | no_event | 55 |

In the tables, rows attend to columns. `✓` = allowed, `s` = self-attention fallback only, `·` = not allowed.

### Scheme A — `all:causal(mask_ties=True)`

| from \ to | H | S | L | D1 | D2 | N |
|-----------|---|---|---|----|----|---|
| **H**  | s | · | · | · | · | · |
| **S**  | · | s | · | · | · | · |
| **L**  | ✓ | ✓ | · | · | · | · |
| **D1** | ✓ | ✓ | ✓ | · | · | · |
| **D2** | ✓ | ✓ | ✓ | · | · | · |
| **N**  | ✓ | ✓ | ✓ | ✓ | ✓ | · |

Notes:
- H and S have age 0, so `mask_ties=True` blocks H ↔ S. They have no target and fall back to self-attention.
- D1 and D2 have the same age, so they do not see each other.
- No token attends to itself, except through the fallback.

### Scheme B — `[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)`

| from \ to | H | S | L | D1 | D2 | N |
|-----------|---|---|---|----|----|---|
| **H**  | ✓ | ✓ | · | · | · | · |
| **S**  | ✓ | ✓ | · | · | · | · |
| **L**  | ✓ | ✓ | · | · | · | · |
| **D1** | ✓ | ✓ | ✓ | · | · | · |
| **D2** | ✓ | ✓ | ✓ | · | · | · |
| **N**  | ✓ | ✓ | ✓ | ✓ | ✓ | · |

The bidirectional rule adds the H/S block (with the diagonal). All other rows are the same as in scheme A. So the at-birth tokens contextualise each other, and later tokens still see them through the causal rule.

The same result, without naming the domains:

```
"[at_birth]:bidirectional,all:causal(mask_ties=True)"
```

Note: `at_birth` also includes `genetic_pcs` when that domain is active.

## Presets (`config/attention_schemes.yaml`)

You can pass a preset name instead of a scheme string:

```bash
python train.py --attention_scheme hla_bidirectional ...
```

Each preset has this form:

```yaml
hla_bidirectional:
  description: HLA + sex form a bidirectional block; clinical domains causal
  domains: diseases,death,lifestyle,hla_alleles,sex
  scheme: "[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)"
```

`train.py` reads only the `scheme` field. The `domains` field is documentation: you must still pass `--domains` yourself.

Available presets:

| Preset | Scheme |
|--------|--------|
| `base`, `hla_no_self_attention`, `cv_drugs_causal`, `cns_drugs_causal`, `cv_cns_drugs_causal` | `all:causal(mask_ties=True)` |
| `base_pcs`, `hla_no_self_attention_pcs` | `[genetic_pcs,sex]:bidirectional,all:causal(mask_ties=True)` |
| `base_with_pcs`, `hla_pcs_causal` | `[genetic_pcs,lifestyle,sex]:bidirectional,all:causal(mask_ties=True)` |
| `hla_bidirectional`, `cv_cns_drugs_hla_bidirectional` | `[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)` |
| `hla_bidirectional_pcs` | `[hla_alleles,sex,genetic_pcs]:bidirectional,all:causal(mask_ties=True)` |
| `hla_pcs_bidirectional` | `[hla_alleles,sex,genetic_pcs,lifestyle]:bidirectional,all:causal(mask_ties=True)` |

Presets with the same scheme differ only in their `domains` field. Read the YAML file for the current list.

## One scheme per layer

`--attention_scheme` accepts 1 value or `n_layer` values. With 1 value, all layers use the same scheme. With `n_layer` values, layer *i* uses value *i*:

```bash
python train.py --n_layer 3 --attention_scheme \
  "[at_birth]:bidirectional,all:causal(mask_ties=True)" \
  "all:causal(mask_ties=True)" \
  "all:causal(mask_ties=True)"
```

Each value can be a scheme string or a preset name.
