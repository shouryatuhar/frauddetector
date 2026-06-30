/* ============================================================
   Screen Recapture Detector — Frontend Logic
   ============================================================ */

(() => {
  'use strict';

  // --- DOM refs ---
  const dropZone       = document.getElementById('drop-zone');
  const fileInput      = document.getElementById('file-input');
  const previewWrapper = document.getElementById('preview-wrapper');
  const previewImg     = document.getElementById('preview-img');
  const previewVideo   = document.getElementById('preview-video');
  const previewLabel   = document.getElementById('preview-label');
  const cameraControls = document.getElementById('camera-controls');

  const btnPredict     = document.getElementById('btn-predict');
  const btnCamera      = document.getElementById('btn-camera');
  const btnCapture     = document.getElementById('btn-capture');
  const btnStopCam     = document.getElementById('btn-stop-cam');
  const btnReset       = document.getElementById('btn-reset');

  const loadingOverlay = document.getElementById('loading-overlay');
  const resultSection  = document.getElementById('result-section');

  const resultBadge    = document.getElementById('result-badge-label');
  const probValue      = document.getElementById('prob-value');
  const timeValue      = document.getElementById('time-value');
  const confValue      = document.getElementById('conf-value');
  const confFill       = document.getElementById('conf-fill');
  const gaugeEl        = document.getElementById('gauge');
  const gaugeValue     = document.getElementById('gauge-value');
  const explanationText = document.getElementById('explanation-text');
  const explanationIcon = document.getElementById('explanation-icon');

  let currentFile      = null;
  let cameraStream     = null;

  // --- Drag & Drop ---
  ['dragenter', 'dragover'].forEach(evt => {
    dropZone.addEventListener(evt, e => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(evt => {
    dropZone.addEventListener(evt, e => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', e => {
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
  });

  dropZone.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
  });

  // --- File handling ---
  function handleFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file (JPG, PNG, etc.)');
      return;
    }

    stopCamera();
    currentFile = file;

    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;
      previewImg.style.display = 'block';
      previewVideo.style.display = 'none';
      previewLabel.textContent = file.name;
      previewWrapper.classList.add('visible');
      btnPredict.disabled = false;
      resultSection.classList.remove('visible');
    };
    reader.readAsDataURL(file);
  }

  // --- Camera ---
  btnCamera.addEventListener('click', async () => {
    if (cameraStream) {
      stopCamera();
      return;
    }

    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
      });

      previewVideo.srcObject = cameraStream;
      previewVideo.style.display = 'block';
      previewImg.style.display = 'none';
      previewVideo.play();
      previewLabel.textContent = 'Camera';
      previewWrapper.classList.add('visible');
      cameraControls.classList.add('visible');
      btnCamera.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
        Stop Camera`;
      btnPredict.disabled = true;
      resultSection.classList.remove('visible');
    } catch (err) {
      alert('Could not access camera. Please allow camera permissions or use file upload.');
    }
  });

  btnCapture.addEventListener('click', () => {
    if (!cameraStream) return;

    const canvas = document.createElement('canvas');
    canvas.width = previewVideo.videoWidth;
    canvas.height = previewVideo.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(previewVideo, 0, 0);

    canvas.toBlob(blob => {
      currentFile = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });

      previewImg.src = canvas.toDataURL('image/jpeg');
      previewImg.style.display = 'block';
      previewVideo.style.display = 'none';
      previewLabel.textContent = 'Camera capture';
      btnPredict.disabled = false;
      stopCamera();
    }, 'image/jpeg', 0.92);
  });

  btnStopCam.addEventListener('click', stopCamera);

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(t => t.stop());
      cameraStream = null;
    }
    cameraControls.classList.remove('visible');
    btnCamera.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/>
        <circle cx="12" cy="13" r="4"/>
      </svg>
      Camera`;
  }

  // --- Predict ---
  btnPredict.addEventListener('click', async () => {
    if (!currentFile) return;

    loadingOverlay.classList.add('visible');
    resultSection.classList.remove('visible');
    btnPredict.disabled = true;

    const formData = new FormData();
    formData.append('image', currentFile);

    const startTime = performance.now();

    try {
      const resp = await fetch('/predict', { method: 'POST', body: formData });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || `Server error ${resp.status}`);
      }

      const data = await resp.json();
      const clientTime = performance.now() - startTime;

      loadingOverlay.classList.remove('visible');
      showResult(data, clientTime);
    } catch (err) {
      loadingOverlay.classList.remove('visible');
      alert('Prediction failed: ' + err.message);
    } finally {
      btnPredict.disabled = false;
    }
  });

  // --- Show Result ---
  function showResult(data, clientMs) {
    const prob = data.probability;
    const label = data.label;
    const inferMs = data.inference_ms;
    const isScreen = label === 'SCREEN';

    // Badge
    resultBadge.textContent = label;
    resultBadge.className = 'result-badge__label ' +
      (isScreen ? 'result-badge__label--screen' : 'result-badge__label--real');

    // Stats
    probValue.textContent = prob.toFixed(4);
    timeValue.textContent = Math.round(inferMs) + ' ms';

    // Confidence bar
    const confidence = isScreen ? prob : (1 - prob);
    const confPct = (confidence * 100).toFixed(1);
    confValue.textContent = confPct + '%';
    confFill.className = 'confidence-fill ' +
      (isScreen ? 'confidence-fill--screen' : 'confidence-fill--real');

    // Animate confidence bar (defer to trigger CSS transition)
    confFill.style.width = '0%';
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        confFill.style.width = confPct + '%';
      });
    });

    // Gauge
    const angle = Math.round(confidence * 360);
    const gaugeColor = isScreen ? '#ef4444' : '#22c55e';
    gaugeEl.style.setProperty('--gauge-color', gaugeColor);
    gaugeEl.style.setProperty('--gauge-angle', '0deg');
    gaugeValue.textContent = confPct + '%';
    gaugeValue.style.color = gaugeColor;

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        gaugeEl.style.setProperty('--gauge-angle', angle + 'deg');
      });
    });

    // Explanation
    explanationIcon.textContent = isScreen ? '🖥️' : '✅';

    let expText;
    if (confidence >= 0.9) {
      expText = isScreen
        ? 'Very high confidence that this image is a photo of a screen.'
        : 'Very high confidence that this is a genuine real-world photograph.';
    } else if (confidence >= 0.7) {
      expText = isScreen
        ? 'High confidence that this image is a photo of a screen.'
        : 'High confidence that this is a genuine photograph.';
    } else if (confidence >= 0.55) {
      expText = isScreen
        ? 'Moderate confidence that this image may be a photo of a screen.'
        : 'Moderate confidence that this appears to be a real photograph.';
    } else {
      expText = 'Low confidence — the model is uncertain about this image. Try a clearer photo.';
    }
    explanationText.textContent = expText;

    resultSection.classList.add('visible');

    // Scroll result into view
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // --- Reset ---
  btnReset.addEventListener('click', () => {
    stopCamera();
    currentFile = null;
    fileInput.value = '';
    previewImg.src = '';
    previewImg.style.display = 'none';
    previewVideo.style.display = 'none';
    previewWrapper.classList.remove('visible');
    resultSection.classList.remove('visible');
    btnPredict.disabled = true;
  });

})();
