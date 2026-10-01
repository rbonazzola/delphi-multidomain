"""
Round-trip tests for the model spec stored in checkpoints (utils/ckpt_utils.py):
model -> model_spec -> checkpoint -> model_from_spec must rebuild the same
architecture, for both Delphi and DelphiMultiStream. Needs only tokenizer.yaml
files (for vocab sizes); runs on CPU.
"""
import io

import pytest
import torch
import yaml

from delphi.model import Delphi, DelphiConfig, DomainConfig
from delphi.multi_stream_model import DelphiMultiStream, DelphiMultiStreamConfig
from utils.ckpt_utils import _REPO_DIR, load_model_from_checkpoint, model_spec

TOKEN_PATH = _REPO_DIR / "data" / "transforms" / "tokens"


def _domains():
    return {
        "padding":     DomainConfig(path=None),
        "diseases":    DomainConfig(path=TOKEN_PATH / "diseases", predict=True),
        "hla_alleles": DomainConfig(path=TOKEN_PATH / "hla_alleles", dropout_mode="block", dropout_rate=0.2),
        "sex":         DomainConfig(path=TOKEN_PATH / "sex", at_birth=True),
    }


def _delphi():
    cfg = DelphiConfig(
        n_layer=2, n_head=2, n_embd=16, domains=_domains(), block_size=32,
        attention_scheme="[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)",
        no_event_token_rate=5, seed=3,
    )
    return Delphi(cfg)


def _multistream():
    cfg = DelphiMultiStreamConfig(domains=_domains(), block_size=32, seed=3)
    return DelphiMultiStream.from_scheme_string(
        "MultiStream([hla_alleles,sex]:(h2d16l1),[diseases,sex]:(h2d16l2)):(h2d16l1)",
        "[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)",
        cfg,
    )


def _save_and_load(model) -> dict:
    """Serialize like MLFlowLogger.build_state_dict does, then load back with weights_only."""
    buf = io.BytesIO()
    torch.save({"model_spec": model_spec(model), "state_dict": model.state_dict()}, buf)
    buf.seek(0)
    return torch.load(buf, map_location="cpu", weights_only=True)


@pytest.mark.parametrize("build", [_delphi, _multistream], ids=["delphi", "multistream"])
def test_roundtrip_rebuilds_identical_model(build):
    model = build()
    ckpt = _save_and_load(model)

    rebuilt = load_model_from_checkpoint(ckpt)  # strict load: same architecture

    assert type(rebuilt) is type(model)
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), rebuilt.state_dict().items()):
        assert k1 == k2
        assert torch.equal(v1, v2)
    assert rebuilt.domain_to_int == model.domain_to_int
    assert rebuilt.domain_offsets == model.domain_offsets


@pytest.mark.parametrize("build", [_delphi, _multistream], ids=["delphi", "multistream"])
def test_spec_is_plain_and_repo_relative(build):
    spec = model_spec(build())

    yaml.safe_dump(spec)  # only plain types: no dataclasses, no PosixPath
    assert spec["config"]["domains"]["diseases"]["path"] == "data/transforms/tokens/diseases"
    assert spec["config"]["domains"]["hla_alleles"]["dropout_rate"] == 0.2


def test_config_overrides_and_compiled_model():
    model = _delphi()
    spec = model_spec(torch.compile(model))  # unwraps torch.compile's OptimizedModule
    assert spec["model_class"] == "Delphi"

    ckpt = _save_and_load(model)
    rebuilt = load_model_from_checkpoint(ckpt, block_size=128)
    assert rebuilt.config.block_size == 128
    assert rebuilt.config.seed == 3


def test_unknown_model_returns_none():
    assert model_spec(torch.nn.Linear(2, 2)) is None


def test_domain_params_roundtrip_and_run_setup():
    """domains.* params carry exactly the non-default fields, parse back, and decode to the
    same RunSetup as the legacy brace-matching of the raw `domains` param."""
    from dataclasses import asdict

    from utils.ckpt_utils import domain_params
    from utils.mlflow_utils import domain_params_from_run_params, get_run_setup

    model = _delphi()
    params = domain_params(model)
    assert domain_params(_save_and_load(model)["model_spec"]) == params  # from a checkpoint's spec
    assert domain_params(None) == {}

    assert "domains.padding" not in params
    assert set(params) == {f"domains.{d}" for d in model.config.domains if d != "padding"}
    doms = domain_params_from_run_params(params)
    assert doms["hla_alleles"]["dropout_mode"] == "block"
    assert doms["hla_alleles"]["dropout_rate"] == 0.2
    assert doms["diseases"]["predict"] is True
    assert "freeze" not in doms["diseases"]  # default fields are omitted

    base = {"attention_scheme": ["[hla_alleles,sex]:bidirectional,all:causal(mask_ties=True)"], "n_embd": "16", "n_head": "2", "n_layer": "2"}
    legacy = get_run_setup(base | {"domains": str(asdict(model.config)["domains"])})
    assert get_run_setup(base | params) == legacy
    assert legacy.label() == "2field__drop20__bidir"
