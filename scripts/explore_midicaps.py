#!/usr/bin/env python3
"""Explore MidiCaps metadata to understand genre tag structure."""

import json
from collections import Counter

print("Loading train.json (JSONL format)...")
items = []
with open("data/raw/midicaps/train.json") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))

print(f"Items loaded: {len(items)}")
item = items[0]

print(f"Item type: {type(item)}")
if isinstance(item, dict):
    print(f"Item keys: {list(item.keys())}")
    for k, v in item.items():
        print(f"  {k}: {repr(v)[:200]}")

print("\n--- Genre tag analysis ---")
genre_counter = Counter()
for entry in items:
    if not isinstance(entry, dict):
        continue
    for field in ("genre", "genres", "tag", "tags", "genre_tag"):
        val = entry.get(field)
        if val:
            if isinstance(val, list):
                for v in val:
                    genre_counter[str(v).strip()] += 1
            else:
                for v in str(val).replace(";", ",").split(","):
                    v = v.strip()
                    if v:
                        genre_counter[v] += 1
            break

print(f"\nTotal genre tags: {sum(genre_counter.values())}")
print(f"Unique tags: {len(genre_counter)}")
print("\nTop 50 genre tags:")
for tag, count in genre_counter.most_common(50):
    print(f"  {tag}: {count}")

print("\n--- File reference analysis ---")
file_fields_found = Counter()
for entry in items[:200]:
    if not isinstance(entry, dict):
        continue
    for field in ("file", "filename", "path", "midi_path", "fname", "location", "midi_filename"):
        if field in entry:
            file_fields_found[field] += 1
print(f"File reference fields (first 200): {dict(file_fields_found)}")


