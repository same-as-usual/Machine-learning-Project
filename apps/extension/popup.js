const local = document.getElementById("modeLocal");
const server = document.getElementById("modeServer");
const apiRow = document.getElementById("apiRow");
const apiUrl = document.getElementById("apiUrl");

function render(mode) {
  (mode === "server" ? server : local).checked = true;
  apiRow.style.display = mode === "server" ? "block" : "none";
}

chrome.storage.sync.get(["mode", "apiUrl"]).then(({ mode = "local", apiUrl: url }) => {
  render(mode);
  apiUrl.value = url || "http://localhost:8000";
});

for (const radio of [local, server]) {
  radio.addEventListener("change", () => {
    const mode = server.checked ? "server" : "local";
    chrome.storage.sync.set({ mode });
    render(mode);
  });
}
apiUrl.addEventListener("change", () => chrome.storage.sync.set({ apiUrl: apiUrl.value.trim() }));
