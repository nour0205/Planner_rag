from pathlib import Path
import json


CATALOG_PATH = Path("data/document_catalog.json")


def _load_catalog() -> list[dict]:
    if not CATALOG_PATH.exists():
        return []

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_catalog(items: list[dict]) -> None:
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CATALOG_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def upsert_document_entry(entry: dict) -> None:
    items = _load_catalog()

    updated = False
    for i, item in enumerate(items):
        if item.get("document_id") == entry.get("document_id"):
            items[i] = entry
            updated = True
            break

    if not updated:
        items.append(entry)

    _save_catalog(items)


def list_document_entries() -> list[dict]:
    return _load_catalog()


def get_document_entry(document_id: str) -> dict | None:
    items = _load_catalog()
    for item in items:
        if item.get("document_id") == document_id:
            return item
    return None