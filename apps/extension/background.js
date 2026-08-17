/* ManipuLens service worker: scores headlines in one of two modes.
 *
 *   local  (default) — the INT8 ONNX student runs INSIDE the browser via
 *                      ONNX Runtime Web. Zero network calls; verifiable in
 *                      devtools. This is the privacy pillar.
 *   server           — POST /score_batch to a ManipuLens API instance.
 *
 * Mode + API URL live in chrome.storage.sync (see popup.js).
 *
 * This is a MODULE service worker (manifest: background.type = "module") using
 * the ort.bundle build: the classic ort.min.js loads its wasm glue via a
 * runtime dynamic import(), which Chrome forbids in MV3 service workers —
 * InferenceSession.create() would throw and no headline ever got a badge.
 * The .bundle. build inlines the glue; the .wasm itself arrives via fetch(),
 * which is allowed.
 */

import * as ort from "./lib/ort/ort.bundle.min.mjs";
import "./lib/tokenizer.js"; // attaches WordPieceTokenizer to globalThis
import "./lib/mask.js"; // attaches maskEntities to globalThis
import "./assets/political_entities.js"; // sets MANIPULENS_POLITICAL_ENTITIES

const { WordPieceTokenizer, maskEntities } = globalThis;

const MAX_LENGTH = 64;

ort.env.wasm.wasmPaths = chrome.runtime.getURL("lib/ort/");
ort.env.wasm.numThreads = 1; // no cross-origin isolation in extension SW

let localState = null; // {session, tokenizer}

async function getLocal() {
  if (!localState) {
    const [session, vocabText] = await Promise.all([
      ort.InferenceSession.create(chrome.runtime.getURL("assets/model.int8.onnx")),
      fetch(chrome.runtime.getURL("assets/vocab.txt")).then((r) => r.text()),
    ]);
    localState = { session, tokenizer: new WordPieceTokenizer(vocabText) };
  }
  return localState;
}

async function scoreLocal(headlines) {
  const { session, tokenizer } = await getLocal();
  const results = [];
  for (const headline of headlines) {
    const masked = maskEntities(headline); // neutrality by construction, as in training
    const { inputIds, attentionMask } = tokenizer.encode(masked, MAX_LENGTH);
    const feeds = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from(inputIds.map(BigInt)), [1, MAX_LENGTH]),
      attention_mask: new ort.Tensor(
        "int64",
        BigInt64Array.from(attentionMask.map(BigInt)),
        [1, MAX_LENGTH]
      ),
    };
    const out = await session.run(feeds);
    // model outputs are already sigmoid()ed (see OnnxWrapper in export_onnx.py)
    results.push({ manipulation_score: out.binary.data[0], intensity_score: out.intensity.data[0] });
  }
  return results;
}

async function scoreServer(headlines, apiUrl) {
  const resp = await fetch(`${apiUrl.replace(/\/$/, "")}/score_batch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ headlines }),
  });
  if (!resp.ok) throw new Error(`API ${resp.status}`);
  const body = await resp.json();
  return body.results.map((r) => ({ manipulation_score: r.manipulation_score }));
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type !== "manipulens:score") return false;
  (async () => {
    const { mode = "local", apiUrl = "http://localhost:8000" } = await chrome.storage.sync.get([
      "mode",
      "apiUrl",
    ]);
    try {
      const scores =
        mode === "server"
          ? await scoreServer(msg.headlines, apiUrl)
          : await scoreLocal(msg.headlines);
      sendResponse({ ok: true, scores, mode });
    } catch (e) {
      sendResponse({ ok: false, error: String(e) });
    }
  })();
  return true; // async response
});
