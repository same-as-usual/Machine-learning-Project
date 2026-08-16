# ManipuLens browser extension (MV3)

Badges news-site headlines with a manipulation score. **Private by default**:
the 23 MB INT8 ONNX student model runs inside the browser via ONNX Runtime
Web — headline text never leaves your machine. Verify it yourself: open
devtools → Network in local mode and watch zero requests happen.

## Build

```bash
# 1. produce the model artifact (repo root)
make train-transformer distill export-onnx   # or the -smoke variants

# 2. bundle the extension (copies model + vocab, generates entity list, fetches ORT)
python apps/extension/build.py
```

## Install

1. `chrome://extensions` → enable **Developer mode**
2. **Load unpacked** → select `apps/extension/`
3. Visit a news site; badges (● green/amber/red) appear next to headlines

## Modes (popup)

| Mode | What happens |
|---|---|
| **Private (in-browser)** — default | WordPiece tokenize + entity-mask + INT8 ONNX inference in the service worker. Zero network. |
| Server API | `POST /score_batch` to a ManipuLens API (full per-dimension breakdown lives there) |

## Parity with training

- `lib/mask.js` mirrors `neutralize.py` — same entity list (generated from the
  same lexicon file), longest-first, word-boundary, case-insensitive.
- `lib/tokenizer.js` mirrors the HF BERT uncased tokenizer; parity is checked
  by `tools/test_tokenizer_parity.py` (JS ids == Python ids on a probe set).

## Notes

- Generated files (`assets/model.int8.onnx`, `assets/vocab.txt`,
  `assets/political_entities.js`, `lib/ort/`) are gitignored — run `build.py`.
- The "I disagree" feedback loop and per-dimension span highlighting are
  roadmap items (server mode already returns the full breakdown).
