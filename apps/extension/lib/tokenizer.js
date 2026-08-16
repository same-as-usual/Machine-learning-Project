/* Minimal BERT WordPiece tokenizer (uncased) for the MiniLM student model.
 *
 * Mirrors HuggingFace BertTokenizer basic+wordpiece behavior:
 *   NFD normalize -> strip combining marks -> lowercase -> split on
 *   whitespace/punctuation (punct chars become single tokens) -> greedy
 *   longest-match WordPiece with "##" continuations.
 *
 * Parity with the Python tokenizer is verified by tools/test_tokenizer_parity
 * (same input -> identical input_ids).
 *
 * Plain script (no modules) so it loads via importScripts() in the MV3
 * service worker and via require() in Node for parity tests.
 */

(function (root) {
  "use strict";

  const PUNCT_RE = /[\p{P}\p{S}]/u; // punctuation + symbols, like HF's _is_punctuation
  const UNK = "[UNK]";
  const MAX_CHARS_PER_WORD = 100;

  function basicTokenize(text) {
    const clean = text
      .normalize("NFD")
      .replace(/[̀-ͯ]/g, "") // strip combining marks
      .toLowerCase();
    const tokens = [];
    let current = "";
    for (const ch of clean) {
      if (/\s/.test(ch)) {
        if (current) tokens.push(current);
        current = "";
      } else if (PUNCT_RE.test(ch)) {
        if (current) tokens.push(current);
        tokens.push(ch);
        current = "";
      } else {
        current += ch;
      }
    }
    if (current) tokens.push(current);
    return tokens;
  }

  function wordpiece(word, vocab) {
    if (word.length > MAX_CHARS_PER_WORD) return [UNK];
    const pieces = [];
    let start = 0;
    while (start < word.length) {
      let end = word.length;
      let piece = null;
      while (start < end) {
        let sub = word.slice(start, end);
        if (start > 0) sub = "##" + sub;
        if (vocab.has(sub)) {
          piece = sub;
          break;
        }
        end -= 1;
      }
      if (piece === null) return [UNK];
      pieces.push(piece);
      start = end;
    }
    return pieces;
  }

  class WordPieceTokenizer {
    /** @param {string} vocabText raw contents of vocab.txt (one token per line) */
    constructor(vocabText) {
      this.vocab = new Map();
      vocabText.split("\n").forEach((tok, i) => {
        // preserve trailing-whitespace-significant tokens; strip only \r
        const t = tok.replace(/\r$/, "");
        if (t.length) this.vocab.set(t, i);
      });
      this.vocabSet = new Set(this.vocab.keys());
      this.clsId = this.vocab.get("[CLS]");
      this.sepId = this.vocab.get("[SEP]");
      this.padId = this.vocab.get("[PAD]");
      this.unkId = this.vocab.get(UNK);
    }

    tokenize(text) {
      const out = [];
      for (const word of basicTokenize(text)) {
        for (const piece of wordpiece(word, this.vocabSet)) out.push(piece);
      }
      return out;
    }

    /** @returns {{inputIds: number[], attentionMask: number[]}} padded to maxLength */
    encode(text, maxLength) {
      const pieces = this.tokenize(text).slice(0, maxLength - 2);
      const ids = [
        this.clsId,
        ...pieces.map((p) => this.vocab.get(p) ?? this.unkId),
        this.sepId,
      ];
      const attentionMask = new Array(maxLength).fill(0);
      const inputIds = new Array(maxLength).fill(this.padId);
      for (let i = 0; i < ids.length; i++) {
        inputIds[i] = ids[i];
        attentionMask[i] = 1;
      }
      return { inputIds, attentionMask };
    }
  }

  root.WordPieceTokenizer = WordPieceTokenizer;
})(typeof self !== "undefined" ? self : globalThis);

if (typeof module !== "undefined" && module.exports) {
  module.exports = { WordPieceTokenizer: globalThis.WordPieceTokenizer };
}
