/* ManipuLens content script: finds headline-like elements, asks the service
 * worker for scores, and injects a small colored badge next to each.
 * Sends only headline TEXT — never URLs or page context. In local mode the
 * text never leaves the browser at all.
 */

(() => {
  "use strict";

  const MIN_LEN = 25;
  const MAX_LEN = 200;
  const MAX_PER_SCAN = 40;
  const MAX_TOTAL = 80;
  const SCAN_EVERY_MS = 3000;
  const MAX_SCANS = 20; // keep rescanning for ~1 min: cold model + late-rendered headlines
  const SEEN = new WeakSet();
  let badged = 0;
  let scanning = false;

  function candidates() {
    const els = [];
    for (const el of document.querySelectorAll("h1, h2, h3, a")) {
      if (SEEN.has(el)) continue;
      const text = (el.innerText || "").trim().replace(/\s+/g, " ");
      if (text.length < MIN_LEN || text.length > MAX_LEN) continue;
      if (el.tagName === "A" && text.split(" ").length < 5) continue;
      if (el.querySelector(".manipulens-badge")) continue;
      els.push({ el, text });
      if (els.length >= MAX_PER_SCAN) break;
    }
    return els;
  }

  function badge(score, mode) {
    const dot = document.createElement("span");
    dot.className = "manipulens-badge";
    const pct = Math.round(score * 100);
    const color = score >= 0.65 ? "#d03b3b" : score >= 0.35 ? "#eda100" : "#0ca30c";
    dot.textContent = ` ●`;
    dot.style.cssText = `color:${color};font-size:0.8em;cursor:help;`;
    dot.title =
      `ManipuLens: ${pct}% manipulation score (${mode} mode). ` +
      `Detects rhetorical techniques, not truth.`;
    return dot;
  }

  async function scan() {
    if (scanning || badged >= MAX_TOTAL) return;
    const found = candidates();
    if (!found.length) return;
    scanning = true;
    try {
      const resp = await chrome.runtime.sendMessage({
        type: "manipulens:score",
        headlines: found.map((f) => f.text),
      });
      // !resp.ok: model may still be warming up — leave elements unmarked so
      // the next scan retries them (fixes "works only after a refresh")
      if (!resp || !resp.ok) return;
      found.forEach((f, i) => {
        SEEN.add(f.el);
        const s = resp.scores[i];
        if (s) {
          f.el.appendChild(badge(s.manipulation_score, resp.mode));
          badged += 1;
        }
      });
    } catch {
      // extension reloaded / SW asleep — try again next scan
    } finally {
      scanning = false;
    }
  }

  scan();
  let scans = 0;
  const timer = setInterval(() => {
    scans += 1;
    if (scans >= MAX_SCANS || badged >= MAX_TOTAL) {
      clearInterval(timer);
      return;
    }
    scan();
  }, SCAN_EVERY_MS);
})();
