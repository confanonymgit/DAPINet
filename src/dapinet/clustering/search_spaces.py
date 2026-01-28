import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import optuna

SearchSpaceFn = Callable[[optuna.Trial], dict[str, Any]]


def load_search_spaces_config() -> dict[str, Any]:
    path = Path(__file__).parent / "search_spaces.json"
    with path.open("r") as f:
        return json.load(f)


SEARCH_SPACES_CONFIG = load_search_spaces_config()


def create_search_space_fn(params_config: dict[str, Any]) -> SearchSpaceFn:
    def _search_space(trial: optuna.Trial) -> dict[str, Any]:
        params = {}
        for param_name, config in params_config.items():
            param_type = config.get("type")
            if param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, config["choices"])
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name, config["low"], config["high"], log=config.get("log", False)
                )
            elif param_type == "float":
                params[param_name] = trial.suggest_float(
                    param_name, config["low"], config["high"], log=config.get("log", False)
                )
        return params

    return _search_space


def _build_search_spaces() -> dict[str, SearchSpaceFn]:
    spaces = {}
    for algo, params in SEARCH_SPACES_CONFIG.items():
        if not params:
            spaces[algo] = None
        else:
            spaces[algo] = create_search_space_fn(params)
    return spaces


SEARCH_SPACES = _build_search_spaces()
