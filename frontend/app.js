document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("file-input");
  const uploadArea = document.getElementById("upload-area");
  const previewImage = document.getElementById("preview-image");
  const previewContainer = document.getElementById("preview-container");
  const analyzeButton = document.getElementById("analyze-button");
  const processingLabel = document.getElementById("processing-label");
  const breedStats = document.getElementById("breed-stats");
  const diseaseStats = document.getElementById("disease-stats");
  const breedInfo = document.getElementById("breed-info");
  const diseaseInfo = document.getElementById("disease-info");

  function setupScrollFade() {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            observer.unobserve(entry.target);
          }
        });
      },
      {
        threshold: 0.15,
      }
    );

    document.querySelectorAll(".fade-section").forEach((el) => observer.observe(el));
  }

  function setPreview(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewImage.classList.remove("hidden");
      const placeholder = previewContainer.querySelector(".preview-placeholder");
      if (placeholder) placeholder.classList.add("hidden");
      processingLabel.textContent = "Ready to analyze.";
    };
    reader.readAsDataURL(file);
  }

  function buildConfidenceRows(container, preds) {
    container.innerHTML = "";
    if (!preds || preds.length === 0) {
      const p = document.createElement("p");
      p.className = "empty-copy";
      p.textContent = "No predictions available.";
      container.appendChild(p);
      return;
    }

    preds.forEach((pred) => {
      const row = document.createElement("div");
      row.className = "confidence-row";

      const header = document.createElement("div");
      header.className = "confidence-header";
      const label = document.createElement("span");
      label.textContent = pred.label.replace(/_/g, " ");
      const value = document.createElement("span");
      const percent = Math.round(pred.score * 100);
      value.textContent = `${percent}%`;
      header.appendChild(label);
      header.appendChild(value);

      const barOuter = document.createElement("div");
      barOuter.className = "confidence-bar-outer";
      const barInner = document.createElement("div");
      barInner.className = "confidence-bar-inner";

      if (percent < 35) {
        barInner.classList.add("low");
      } else if (percent < 65) {
        barInner.classList.add("medium");
      }

      barOuter.appendChild(barInner);

      row.appendChild(header);
      row.appendChild(barOuter);

      container.appendChild(row);

      requestAnimationFrame(() => {
        const width = Math.max(percent / 100, 0.06);
        barInner.style.transform = `scaleX(${width})`;
      });
    });
  }

  function setNarrative(target, markdownText) {
    if (!markdownText) {
      target.textContent = "";
      return;
    }
    const text = markdownText.replace(/^\*\*(.*?)\*\*/gm, "$1");
    target.textContent = text;
  }

  async function analyze() {
    const file = fileInput.files[0];
    if (!file) {
      processingLabel.textContent = "Please upload an image first.";
      return;
    }

    analyzeButton.disabled = true;
    uploadArea.classList.add("scanning-active");
    processingLabel.textContent = "Analyzing image with hybrid CNN + ViT…";

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(errText || `Request failed: ${response.status}`);
      }

      const data = await response.json();
      buildConfidenceRows(breedStats, data.breed_predictions || []);
      buildConfidenceRows(diseaseStats, data.disease_predictions || []);
      setNarrative(breedInfo, data.breed_info || "");
      setNarrative(diseaseInfo, data.disease_info || "");

      processingLabel.textContent = "Analysis complete.";
    } catch (err) {
      console.error(err);
      processingLabel.textContent = "Something went wrong. Check the server logs.";
    } finally {
      analyzeButton.disabled = false;
      uploadArea.classList.remove("scanning-active");
    }
  }

  uploadArea.addEventListener("click", () => {
    fileInput.click();
  });

  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
  });

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      fileInput.files = e.dataTransfer.files;
      setPreview(file);
    }
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (file) {
      setPreview(file);
    }
  });

  analyzeButton.addEventListener("click", () => {
    analyze();
  });

  setupScrollFade();
});

