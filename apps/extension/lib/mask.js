/* Political-entity masking — MUST mirror src/manipulens/models/neutralize.py
 * so in-browser inference matches training-time preprocessing.
 * The entity list is generated into assets/political_entities.js by build.py
 * (single source of truth: labeling/lexicons/political_entities.txt).
 */

(function (root) {
  "use strict";

  const MASK_TOKEN = "entitytoken";
  let pattern = null;

  function escapeRe(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function getPattern() {
    if (!pattern) {
      const terms = (root.MANIPULENS_POLITICAL_ENTITIES || [])
        .slice()
        .sort((a, b) => b.length - a.length); // longest-first, like Python
      pattern = new RegExp("\\b(?:" + terms.map(escapeRe).join("|") + ")\\b", "gi");
    }
    return pattern;
  }

  root.maskEntities = function maskEntities(text) {
    if (!(root.MANIPULENS_POLITICAL_ENTITIES || []).length) return text;
    return text.replace(getPattern(), MASK_TOKEN);
  };
})(typeof self !== "undefined" ? self : globalThis);

if (typeof module !== "undefined" && module.exports) {
  module.exports = { maskEntities: globalThis.maskEntities };
}
