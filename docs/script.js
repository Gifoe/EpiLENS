const copyButton = document.querySelector("#copy-bibtex");
const bibtex = document.querySelector("#bibtex code");

copyButton?.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(bibtex.textContent.trim());
    copyButton.textContent = "Copied";
    window.setTimeout(() => {
      copyButton.textContent = "Copy BibTeX";
    }, 1600);
  } catch {
    copyButton.textContent = "Select text to copy";
  }
});
