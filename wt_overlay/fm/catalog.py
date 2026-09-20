"""Pinned Air RB aircraft identities and their unmodified FM sources."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "fm"
COUNTRIES = {"usa": "美国", "germany": "德国", "ussr": "苏联", "britain": "英国",
             "japan": "日本", "china": "中国", "italy": "意大利", "france": "法国",
             "sweden": "瑞典", "israel": "以色列"}


def aircraft_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


@dataclass(frozen=True)
class Aircraft:
    id: str
    name: str
    name_en: str
    country: str
    br: float
    path: Path
    sha256: str
    source_url: str
    revision: str

    @property
    def label(self) -> str:
        return f"{COUNTRIES[self.country]} · {self.name} · {self.br:.1f}"


@lru_cache(maxsize=1)
def aircraft_catalog() -> tuple[Aircraft, ...]:
    data = json.loads((DATA_DIR / "catalog.json").read_text(encoding="utf-8"))
    revision = data["revision"]
    prefix = f"https://github.com/{data['repository']}/blob/{revision}/"
    return tuple(Aircraft(p["id"], p["name"], p["name_en"], p["country"], p["br"],
                          DATA_DIR / p["fm_file"], p["fm_sha256"],
                          prefix + p["fm_source_path"], revision) for p in data["aircraft"])


@lru_cache(maxsize=1)
def _identities() -> dict[str, Aircraft]:
    identities = {}
    for aircraft in aircraft_catalog():
        key = aircraft_key(aircraft.id)
        if key in identities:
            raise ValueError(f"ambiguous aircraft identity: {aircraft.id}")
        identities[key] = aircraft
    return identities


def find_aircraft(identity: str | None) -> Aircraft | None:
    return _identities().get(aircraft_key(identity)) if identity else None
