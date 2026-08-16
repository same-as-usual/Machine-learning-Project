/* ManipuLens content script: finds headline-like elements, asks the service
 * worker for scores, and injects a small colored badge next to each.
 * Sends only headline TEXT — never URLs or page context. In local mode the
 * text never leaves the browser at all.
 */

(() => {
  "use strict";

  const MIN_LEN = 25;
  const MAX_LEN = 200;
  const MAX_PER_PAGE = 40;
  const SEEN = new WeakSet();

  function candidates() {
    const els = [];
    for (const el of document.querySelectorAll("h1, h2, h3, a")) {
      if (SEEN.has(el)) continue;
      const text = (el.innerText || "").trim().replace(/\s+/g, " ");
      if (text.length < MIN_LEN || text.length > MAX_LEN) continue;
      if (el.tagName === "A" && text.split(" ").length < 5) continue;
      if (el.querySelector(".manipulens-badge")) continue;
      els.push({ el, text });
      if (els.length >= MAX_PER_PAGE) break;
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
    const found = candidates();
    if (!found.length) return;
    let resp;
    try {
      resp = await chrome.runtime.sendMessage({
        type: "manipulens:score",
        headlines: found.map((f) => f.text),
      });
    } catch {
      return; // extension reloaded / SW asleep — try again next scan
    }
    if (!resp || !resp.ok) return;
    found.forEach((f, i) => {
      SEEN.add(f.el);
      const s = resp.scores[i];
      if (s) f.el.appendChild(badge(s.manipulation_score, resp.mode));
    });
  }

  scan();
  setTimeout(scan, 3000); // catch late-rendering headlines
})();
