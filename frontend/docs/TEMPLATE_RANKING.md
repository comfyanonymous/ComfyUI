# Template Ranking System

Usage-based ordering for workflow templates with position bias normalization.

Scores are pre-computed and normalized offline and shipped as static JSON (mirrors `sorted-custom-node-map.json` pattern for node search).

## Sort Modes

| Mode           | Formula                                          | Description            |
| -------------- | ------------------------------------------------ | ---------------------- |
| `recommended`  | `usage × 0.5 + internal × 0.3 + freshness × 0.2` | Curated recommendation |
| `popular`      | `usage × 0.9 + freshness × 0.1`                  | Pure user-driven       |
| `newest`       | Date sort                                        | Existing               |
| `alphabetical` | Name sort                                        | Existing               |

Freshness computed at runtime from `template.date`: `1.0 / (1 + daysSinceAdded / 90)`, min 0.1.

## Data Files

**Usage scores** (generated from Mixpanel):

```json
// In templates/index.json, add to any template:
{
  "name": "some_template",
  "usage": 1000,
  ...
}
```

**Search rank** (set per-template in workflow_templates repo):

```json
// In templates/index.json, add to any template:
{
  "name": "some_template",
  "searchRank": 8,  // Scale 1-10, default 5
  ...
}
```

| searchRank | Effect                       |
| ---------- | ---------------------------- |
| 1-4        | Demote (bury in results)     |
| 5          | Neutral (default if not set) |
| 6-10       | Promote (boost in results)   |

## Position Bias Correction

Raw usage reflects true preference AND UI position bias. We use linear interpolation:

```
correction = 1 + (position - 1) / (maxPosition - 1)
normalizedUsage = rawUsage × correction
```

| Position | Boost |
| -------- | ----- |
| 1        | 1.0×  |
| 50       | 1.28× |
| 100      | 1.57× |
| 175      | 2.0×  |

Templates buried at the bottom get up to 2× boost to compensate for reduced visibility.

---
