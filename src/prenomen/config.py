"""Optional YAML configuration for :class:`~prenomen.inference.Prenomen`.

Configuration is Python first: every setting is a constructor argument. This module
exists because an endpoint, a model name and a list of dictionaries are the kind of
thing people like to keep in a file and share, and because the error messages for a
mistyped key are worth having in one place.

    p = Prenomen.from_config("prenomen.yaml", max_workers=4)

Keyword arguments override the file. See ``examples/prenomen.yaml``.
"""

import os
from typing import Any

import yaml


def load_yaml(path) -> dict:
    """Parse one YAML config, reporting the usual mistakes as ValueError.

    A missing path, a syntax error and an empty document all have to surface as the
    one exception the caller can render as a single line instead of a traceback.
    """
    try:
        with open(path, "rb") as f:
            loaded = yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"Config file not found: {path}") from None
    except OSError as exc:
        raise ValueError(f"{path}: cannot be read: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: is not valid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        found = "an empty file" if loaded is None else type(loaded).__name__
        raise ValueError(f"{path}: expected a YAML mapping, found {found}")
    return loaded


def require(cfg: dict, path: str, where: str):
    """Fetch a mandatory (possibly dotted) config key, naming the file if absent."""
    node = cfg
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f"{where}: missing required `{path}` setting")
        node = node[key]
    return node


def section(cfg: dict, name: str, where: str) -> dict:
    """An optional block of settings, empty when absent.

    ``name:`` with nothing under it parses as None, which is what commenting a
    block out leaves behind. That reads as "no settings here", not as a crash.
    """
    value = cfg.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(
            f"{where}: `{name}:` must be a block of settings, "
            f"found {type(value).__name__}"
        )
    return value


# Settings accepted under `inference:`, mapped to constructor arguments of the same
# name. Listed rather than passed through, so a typo is reported instead of ignored.
_INFERENCE_KEYS = (
    "dict_bypass_threshold",
    "max_workers",
    "timeout",
    "max_retries",
    "cache_size",
    "template_dir",
    "preload",
)


def _dictionaries(raw, where: str):
    """Read the ``dictionaries:`` list: names of shipped ones, or full specs."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(
            f"{where}: `dictionaries:` must be a list of names or specs, found "
            f"{type(raw).__name__}. For example:\n"
            f"  dictionaries:\n    - wgnd\n    - name: my_list\n"
            f'      path: "~/names.tsv"\n      name_col: "name"'
        )
    from .dictionaries import DictionarySpec

    out = []
    for position, item in enumerate(raw):
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            try:
                out.append(DictionarySpec(**item))
            except TypeError as exc:
                raise ValueError(
                    f"{where}: dictionaries[{position}]: {exc}"
                ) from exc
        else:
            raise ValueError(
                f"{where}: dictionaries[{position}] must be a name or a mapping, "
                f"found {type(item).__name__}"
            )
    return out


def config_to_kwargs(cfg: dict, where: str) -> dict[str, Any]:
    """Translate a parsed config into Prenomen constructor arguments."""
    api = section(cfg, "api", where)
    if not api.get("base_url"):
        raise ValueError(
            f"{where}: needs an `api:` section naming the endpoint:\n"
            f"  api:\n"
            f'    base_url: "https://api.openai.com/v1"\n'
            f'    api_key_env: "OPENAI_API_KEY"'
        )
    kwargs: dict[str, Any] = {
        "model": require(cfg, "model.name", where),
        "base_url": api["base_url"],
        "api_key": api.get("api_key"),
        "api_key_env": api.get("api_key_env"),
        "parameters": section(cfg, "parameters", where),
        "dictionaries": _dictionaries(cfg.get("dictionaries"), where),
    }
    if cfg.get("dictionary_dir") is not None:
        kwargs["dictionary_dir"] = os.path.expanduser(str(cfg["dictionary_dir"]))

    inference = section(cfg, "inference", where)
    unknown = sorted(set(inference) - set(_INFERENCE_KEYS))
    if unknown:
        raise ValueError(
            f"{where}: unknown `inference:` setting(s) {unknown}. Accepted: "
            f"{list(_INFERENCE_KEYS)}"
        )
    kwargs.update({k: v for k, v in inference.items() if k in _INFERENCE_KEYS})
    return kwargs
