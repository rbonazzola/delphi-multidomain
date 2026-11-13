# Domain configuration

A domain config YAML defines the domains that the model can use: where their data is, how each one is embedded, and how it is treated during training. The code is in `delphi/model.py` (`DomainConfig`) and `utils/utils.py` (`load_domain_config`).

To create the data files of a domain, see [data_preparation.md](data_preparation.md).

## Basic usage

```yaml
# config/domain_config_default.yaml (excerpt)
diseases:
  projector: embed
  path: diseases
  predict: true
  no_repeat: true
  dropout_mode: token
  dropout_rate: 0.1
  group: core

sex:
  projector: embed
  path: sex
  at_birth: true
  group: core

genetic_pcs:
  type: continuous
  projector: linear
  path: genetic_pcs
  at_birth: true
  input_size: 40
  n_latent_tokens: 5
```

Give the YAML with `--domain_config_yaml` and select the active domains with `--domains`:

```bash
python train.py --domain_config_yaml config/domain_config_default.yaml \
    --domains diseases,death,lifestyle,sex ...
```

- A domain in the YAML but not in `--domains` is not used.
- `--domains` also accepts `group` names: `--domains core,genetic_pcs` expands `core` to all the domains with `group: core`.
- `padding` and `no_event` are reserved pseudo-domains. The loader adds them. Do not put them in `--domains`. The YAML can contain only `no_event: {predict: true}` (all other fields are ignored).

## Field reference

### Data

| Field | Default | Description |
|-------|---------|-------------|
| `path` | `None` | Domain directory, relative to `<data_root>/tokens/` (or absolute) |
| `type` | `categorical` | `categorical` (token IDs) or `continuous` (a vector of values per subject) |
| `at_birth` | `false` | Set all token ages to 0 (sex, HLA, genetic PCs). Cannot be used with `age_jitter` |
| `subdomain` | `None` | Keep only the tokens with this value in `subdomain_column` of `token_metadata.csv` ([Hierarchy](#hierarchy-parent-and-abstract)) |
| `subdomain_column` | `locus` | Metadata column for `subdomain` |
| `token_value_column` | `None` | Merge tokens that have the same value in this metadata column ([Token collapsing](#token-collapsing-token_value_column)). Cannot be used with `subdomain` |

### Embedding (projector)

| Field | Default | Description |
|-------|---------|-------------|
| `projector` | `embed` | `embed`: learned row in the global embedding table. `linear` / `mlp`: project a continuous vector. `pretrained`: fixed external embeddings + learned linear projection |
| `input_size` | `None` | Continuous domains: length of the input vector. Must be equal to the number of entries in `tokenizer.yaml` |
| `n_latent_tokens` | `None` | Continuous domains: number of sequence positions that the projected vector fills (default 1) |
| `projector_rank` | `None` | `linear`: bottleneck dimension of a low-rank projector. `None` = full rank |
| `n_layers`, `n_hidden` | `None` | `mlp`: number of layers and hidden width |
| `pretrained_path` | `None` | `pretrained`: `.pt` tensor `[vocab_size, d_ext]` |
| `freeze` | `false` | `pretrained`: keep the external embeddings fixed and train only the projection |

### Training behaviour

| Field | Default | Description |
|-------|---------|-------------|
| `predict` | `false` | Compute logits and loss for this domain |
| `no_repeat` | `false` | Set logits of tokens that the subject already has to −∞ ([details](#no_repeat)) |
| `age_jitter` | `false` | Training only: add uniform random noise to the token ages |
| `age_jitter_min`, `age_jitter_max` | −20 y, +40 y | Noise range, in days. Strings with units are also accepted (`-20y`, `6m`, `1000d`). The ages are clipped at 0 |
| `dropout_mode` | `None` | Training only: `token`, `block` or `block_and_token` ([Domain dropout](#domain-dropout)) |
| `dropout_rate` | `0.0` | Drop probability. If it is 0, `dropout_mode` is ignored |
| `token_dropout_rate` | `0.0` | `block_and_token`: token drop probability for subjects whose block is not dropped |

### Organisation

| Field | Default | Description |
|-------|---------|-------------|
| `group` | `None` | Short name for logs and run summaries. It is also an alias in `--domains` and in attention schemes. Not inherited through `parent` |
| `parent` | `None` | Inherit fields from another domain in the same YAML |
| `abstract` | `false` | Template only: the domain is not created, but other domains can inherit from it |

## `no_repeat`

Use `no_repeat: true` for domains where each token appears at most once per subject by construction, e.g. diseases recorded only at first diagnosis. After each forward pass, the logits for tokens already seen are set to −∞, before the loss is computed. Without it, the model learns to suppress the logit of a disease once it has appeared, which is a property of the data, not a biological signal.

## Domain groups (`group`)

`config/domain_config_default.yaml` sets these groups:

| Group | Domains |
|-------|---------|
| `core` | diseases, death, lifestyle, sex |
| `drugs` | cv_drugs, ns_drugs |
| `hla` | hla_alleles |
| *(none)* | genetic_pcs — shown by its own name |

A group is used in three places:
- Logs and run summaries show the group name instead of the list of domains.
- `--domains core` expands to all domains of the group.
- In an attention scheme, the group name expands to its domains (see [attention_schemes.md](attention_schemes.md)). If a domain has both `group` and `parent`, only `group` is an alias.

## Extending a config (`extends`)

A YAML can inherit from another file:

```yaml
# config/my_experiment.yaml
extends: domain_config_default.yaml

diseases:
  predict: false        # change one field; all other fields are inherited

hla_a:
  parent: hla_alleles   # within-file inheritance still works
  subdomain: hla_a
  predict: true
  group: hla            # group is not inherited through parent; set it explicitly
```

- The child file is merged on top of the base file, field by field within each domain. Fields that the child does not declare keep the base value.
- Chains work (A extends B extends C).
- The `extends` path is relative to the file that contains it.

## Hierarchy (`parent` and `abstract`)

A child domain copies all the fields of its parent, except `parent`, `subdomain`, `subdomain_column`, `predict`, `group` and `abstract`. Fields in the child replace the parent's values.

Typical use: one abstract HLA domain with the shared settings, and one child per locus.

```yaml
hla_alleles:
  abstract: true            # not a real domain; template only
  projector: embed
  path: hla_alleles
  at_birth: true

hla_a:
  parent: hla_alleles
  subdomain: hla_a
  subdomain_column: locus   # not inherited; default is "locus"
  group: hla

hla_b:
  parent: hla_alleles
  subdomain: hla_b
  group: hla
```

Each child loads only the tokens with its `subdomain` value in `token_metadata.csv`, and its token IDs are renumbered from 0. The children of the same parent share the block-dropout draw (see below).

In attention schemes, each domain is reachable through one alias: its `group` if it has one, otherwise its `parent`. In the example above, the alias is `hla` (from `group`), not `hla_alleles`. Remove `group` from the children if you want to write `hla_alleles` in the scheme (as in `config/domain_config_per_hla_locus.yaml`).

## Overriding fields from the CLI (`--dcfg`)

Change fields at run time, without editing the YAML:

```bash
python train.py --dcfg diseases.predict=True hla_alleles.dropout_rate=0.2 lifestyle.age_jitter=False ...
python train.py --dcfg lifestyle.age_jitter_min=-20y lifestyle.age_jitter_max=10y ...
```

- Format: `DOMAIN.FIELD=VALUE`. Values are parsed as YAML (`True`, `0.2`, `null`, strings).
- One `--dcfg` can contain several overrides, separated by spaces. A single string with spaces (for example, one TSV cell from `sarray_params`) is also split.
- An unknown domain or field is an error.
- The overrides are applied after `extends` and `parent` are resolved, so they change only that domain (not its children).

## Domain dropout

Training only. A dropped token is replaced by padding.

| `dropout_mode` | Effect |
|----------------|--------|
| `token` | Each token of the domain is dropped with probability `dropout_rate` |
| `block` | The whole domain is dropped for a subject with probability `dropout_rate` |
| `block_and_token` | Like `block`; for subjects that keep the domain, each token is then dropped with probability `token_dropout_rate` |

For `block` modes, domains with the same `parent` (e.g. per-locus HLA domains) share one draw per subject: either all loci are dropped or none.

## Pretrained token embeddings (`projector: pretrained`)

Use external embeddings (e.g. ESM2 protein embeddings of HLA alleles) instead of learning them from scratch:

```yaml
hla_a:
  parent: hla_alleles
  subdomain: hla_a
  projector: pretrained
  pretrained_path: embeddings/hla/esm2_t33_650M_UR50D/hla_a.pt
  freeze: false
```

- The `.pt` file is a tensor `[vocab_size, d_ext]`. Row *i* is the embedding of token *i* in the domain vocabulary (after `subdomain` renumbering).
- A learned linear layer projects `d_ext` to `n_embd`, so `d_ext` can be any size. This layer is always trained from scratch.
- `freeze: true` trains only the projection. `freeze: false` also fine-tunes the embeddings.
- Children that do not set `projector` use the parent's value, so one config can mix pretrained and learned loci.

To create the `.pt` files, see [data_preparation.md](data_preparation.md#5-pretrained-embeddings-optional).

## Token collapsing (`token_value_column`)

Merge fine-grained tokens into coarser ones inside one domain, without child domains:

```yaml
hla_alleles:
  projector: embed
  path: hla_alleles
  at_birth: true
  token_value_column: allele_1field   # 2-field alleles → 1-field groups
```

At load time, `TokenDomain` reads `token_metadata.csv`, groups the `token_id`s by their value in `token_value_column`, and gives each unique value a new contiguous ID. The vocabulary then has one entry per value (`HLA-A*01`, `HLA-A*02`, …). Tokens with the same value become the same token.
