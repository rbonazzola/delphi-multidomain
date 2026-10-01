"""
Tests for utils/backfill_model_spec.py's config recovery on runs whose MLflow
'domains' param was truncated at 6000 chars. Builds fake runs (params + artifact
dir) around a real model's state_dict; needs only tokenizer.yaml files, runs on CPU.
"""
from types import SimpleNamespace

import pytest
import yaml

from delphi.model import DomainConfig
from tests.test_model_spec import _delphi
from utils.backfill_model_spec import (
    RecoveryError, overrides_for, recover_domains, recover_run,
)
from utils.ckpt_utils import _REPO_DIR, model_spec

TOKENS = _REPO_DIR / "data" / "transforms" / "tokens"


def _fake_run(params, name="fold1__noevent", tags=None):
    return SimpleNamespace(
        info=SimpleNamespace(run_id="r" * 32, run_name=name, experiment_id="0"),
        data=SimpleNamespace(params=params, tags=tags or {}),
    )


def _domains_param(model, n_chars=None):
    """The 'domains' param as Trainer logs it (repr of the config dicts, with absolute
    paths from another machine), optionally truncated like MLflow does."""
    domains = {
        k: {**vars(v), "path": f"/gpfs/elsewhere/delphi-refactor/data/transforms/tokens/{k}" if v.path else None}
        for k, v in model.config.domains.items()
    }
    s = repr(domains)
    return s[:n_chars] if n_chars else s


def _params(model, domains_str):
    cfg = model.config
    return {
        "domains": domains_str, "attention_scheme": repr(cfg.attention_scheme),
        "n_layer": str(cfg.n_layer), "n_head": str(cfg.n_head), "n_embd": str(cfg.n_embd),
        "block_size": str(cfg.block_size), "seed": str(cfg.seed),
        "no_event_token_rate": str(cfg.no_event_token_rate), "test_fold": "3",
    }


@pytest.fixture
def run_yaml(tmp_path):
    """Artifact dir holding the (unresolved) domain config yaml the run was launched with."""
    (tmp_path / "domain_config_test_backfill.yaml").write_text(yaml.safe_dump({
        "diseases":    {"path": str(TOKENS / "diseases"), "predict": True},
        "hla_alleles": {"path": str(TOKENS / "hla_alleles"), "dropout_mode": "block", "dropout_rate": 0.2},
        "sex":         {"path": str(TOKENS / "sex"), "at_birth": True},
    }))
    return tmp_path


def test_full_param_is_used_verbatim_and_paths_relocated(tmp_path):
    model = _delphi()
    run = _fake_run(_params(model, _domains_param(model)))

    rec = recover_run(run, tmp_path, model.state_dict(), rules=[])

    assert rec.source == "param"
    spec = model_spec(rec.model)
    assert spec["config"]["domains"]["diseases"]["path"] == "data/transforms/tokens/diseases"
    assert rec.data_config["no_event_token_rate"] == 5
    assert rec.data_config["test_fold"] == 3


def test_truncated_param_recovers_tail_from_yaml(run_yaml):
    model = _delphi()
    full = _domains_param(model)
    cut = full.index("'hla_alleles'") + 40          # hla_alleles' dict is cut in half
    params = _params(model, full[:cut]) | {"domain_list": "diseases,hla_alleles,sex"}

    domains, source, notes = recover_domains(_fake_run(params), run_yaml, model.state_dict(), [], {})

    assert source == "partial"
    assert domains["diseases"].predict is True                 # verbatim from the param
    assert domains["hla_alleles"].dropout_rate == 0.2          # from the yaml
    assert "hla_alleles" in notes[0] and "diseases" not in notes[0]


def test_rules_only_touch_domains_recovered_from_yaml(run_yaml):
    model = _delphi()
    full = _domains_param(model)
    params = _params(model, full[: full.index("'hla_alleles'") + 40]) | {"domain_list": "diseases,hla_alleles,sex"}
    domain_ovs, data = overrides_for("fold1__noevent", [
        (__import__("re").compile("noevent"),
         ["no_event.predict=True", "diseases.predict=False", "data.no_event_token_rate=7"]),
    ])

    domains, _, notes = recover_domains(_fake_run(params), run_yaml, model.state_dict(), domain_ovs, {})

    assert domains["no_event"].predict is True     # recovered from the yaml -> rule applies
    assert domains["diseases"].predict is True     # recorded in the run's own param -> rule ignored
    assert data == {"no_event_token_rate": 7}
    assert "no_event.predict=True" in notes[1] and "diseases.predict" not in notes[1]


def test_wrong_domain_set_fails_strict_verification(run_yaml):
    model = _delphi()
    full = _domains_param(model)
    params = _params(model, full[: full.index("'hla_alleles'") + 40]) | {"domain_list": "diseases,sex"}

    with pytest.raises(RecoveryError, match="state_dict mismatch"):
        recover_run(_fake_run(params), run_yaml, model.state_dict(), rules=[])


def test_multistream_without_no_event_rate_requires_rule(tmp_path):
    from tests.test_model_spec import _multistream

    model = _multistream()
    domains = {k: {**vars(v), "path": str(v.path) if v.path else None} for k, v in model._config.domains.items()}
    params = {
        "domains": repr(domains), "arch": "MultiStream([hla_alleles,sex]:(h2d16l1),[diseases,sex]:(h2d16l2)):(h2d16l1)",
        "attention_scheme": model.attention_scheme, "block_size": "32", "seed": "3",
    }

    with pytest.raises(RecoveryError, match="no_event_token_rate"):
        recover_run(_fake_run(params), tmp_path, model.state_dict(), rules=[])

    rule = [(__import__("re").compile("."), ["data.no_event_token_rate=5"])]
    rec = recover_run(_fake_run(params), tmp_path, model.state_dict(), rules=rule)
    assert rec.data_config["no_event_token_rate"] == 5
    assert isinstance(rec.model.embed, type(model.embed))


def test_domain_config_fields_roundtrip():
    # sanity: vars(DomainConfig) is what Trainer's asdict-based param logging produces
    assert set(vars(DomainConfig())) >= {"path", "predict", "dropout_rate"}


@pytest.mark.parametrize("logged", [
    "/gpfs/elsewhere/delphi-refactor/data/transforms/tokens/diseases",
    "../data/transforms/tokens/diseases",
    "data/transforms/tokens/diseases",
])
def test_relocate_paths_logged_on_other_machines_or_cwds(logged):
    from utils.backfill_model_spec import _relocate
    assert _relocate(logged) == str(TOKENS / "diseases")


def test_data_missing_here_builds_with_fallback_vocab_but_spec_keeps_run_paths(tmp_path, monkeypatch):
    import utils.backfill_model_spec as bf

    model = _delphi()
    synth = "/gpfs/elsewhere/delphi-refactor/data/synthetic/cohort1/fold1/tokens"
    domains = {k: {**vars(v), "path": f"{synth}/{k}" if v.path else None} for k, v in model.config.domains.items()}
    run = _fake_run(_params(model, repr(domains)))

    with pytest.raises(RecoveryError, match="not found"):
        recover_run(run, tmp_path, model.state_dict(), rules=[])

    monkeypatch.setattr(bf, "VOCAB_FALLBACK", TOKENS)
    rec = recover_run(run, tmp_path, model.state_dict(), rules=[])
    assert rec.spec_paths["diseases"] == "data/synthetic/cohort1/fold1/tokens/diseases"
    assert "padding" not in rec.spec_paths


def test_partial_run_on_other_cohort_records_data_root_paths_for_yaml_domains(run_yaml):
    model = _delphi()
    full = _domains_param(model)
    params = _params(model, full[: full.index("'hla_alleles'") + 40]) | {
        "domain_list": "diseases,hla_alleles,sex", "data_root": "data/synthetic/cohort1/fold1",
    }
    spec_paths = {}
    recover_domains(_fake_run(params), run_yaml, model.state_dict(), [], spec_paths)
    assert spec_paths["hla_alleles"] == "data/synthetic/cohort1/fold1/tokens/hla_alleles"


def test_dates_stay_strings():
    from utils.backfill_model_spec import _parse_param
    assert _parse_param("date_cutoff", "2020-01-01") == "2020-01-01"
    assert _parse_param("no_event_token_rate", "5.0") == 5.0
    assert _parse_param("block_size", "auto") == 512
    _, data = overrides_for("x", [(__import__("re").compile("."), ["data.date_cutoff=2020-01-01"])])
    assert data == {"date_cutoff": "2020-01-01"}


def test_log_domain_params_adds_missing_keys_to_finished_run(tmp_path):
    """domains.* params are added to an already finished run; existing keys are kept,
    and one with a different value is reported as a conflict, never overwritten."""
    import mlflow
    from utils.backfill_model_spec import log_domain_params
    from utils.ckpt_utils import domain_params

    model = _delphi()
    spec = model_spec(model)
    expected = domain_params(spec)

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file://{tmp_path}")
    run = client.create_run(client.create_experiment("backfill_test"))
    client.log_param(run.info.run_id, "domains.sex", expected["domains.sex"])   # already there, same value
    client.log_param(run.info.run_id, "domains.diseases", "predict=False")      # stale / conflicting
    client.set_terminated(run.info.run_id)
    run = client.get_run(run.info.run_id)

    n, conflicts = log_domain_params(client, run, spec, apply=False)            # dry run: nothing written
    assert (n, conflicts) == (len(expected) - 2, ["domains.diseases"])
    assert set(client.get_run(run.info.run_id).data.params) == {"domains.sex", "domains.diseases"}

    n, conflicts = log_domain_params(client, run, spec, apply=True)
    params = client.get_run(run.info.run_id).data.params
    assert n == len(expected) - 2 and conflicts == ["domains.diseases"]
    assert params == expected | {"domains.diseases": "predict=False"}


def test_write_spec_json_only_when_missing(tmp_path):
    import json
    from utils.backfill_model_spec import write_spec_json
    from utils.ckpt_utils import MODEL_SPEC_ARTIFACT

    spec = model_spec(_delphi())
    assert write_spec_json(tmp_path, spec, apply=False) == "missing"
    assert not (tmp_path / MODEL_SPEC_ARTIFACT).exists()                  # dry run writes nothing

    assert write_spec_json(tmp_path, spec, apply=True) == "written"
    assert json.loads((tmp_path / MODEL_SPEC_ARTIFACT).read_text()) == spec

    (tmp_path / MODEL_SPEC_ARTIFACT).write_text('{"kept": true}')
    assert write_spec_json(tmp_path, spec, apply=True) == "exists"        # never overwritten
    assert json.loads((tmp_path / MODEL_SPEC_ARTIFACT).read_text()) == {"kept": True}
