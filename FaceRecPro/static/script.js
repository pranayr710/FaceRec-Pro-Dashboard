// FaceRec Pro | Core Intelligence Logic

// --- Navigation & Tab Switching ---
const navItems = document.querySelectorAll('.nav-item');
const tabContents = document.querySelectorAll('.tab-content');

navItems.forEach(item => {
    item.addEventListener('click', () => {
        const targetTab = item.getAttribute('data-tab');
        
        // Update nav
        navItems.forEach(i => i.classList.remove('active'));
        item.classList.add('active');
        
        // Update tabs
        tabContents.forEach(t => t.classList.remove('active'));
        document.getElementById(targetTab).classList.add('active');
        
        console.log(`Switched to tab: ${targetTab}`);
    });
});

// --- Live Feed Controls ---
const landmarkToggle = document.getElementById('landmark-toggle');
const toleranceSlider = document.getElementById('tolerance-slider');
const toleranceValue = document.getElementById('tolerance-value');

landmarkToggle.addEventListener('change', async () => {
    const formData = new FormData();
    formData.append('active', landmarkToggle.checked);
    await fetch('/toggle_landmarks', { method: 'POST', body: formData });
});

toleranceSlider.addEventListener('input', async (e) => {
    const val = e.target.value;
    toleranceValue.innerText = val;
    const formData = new FormData();
    formData.append('value', val);
    await fetch('/set_tolerance', { method: 'POST', body: formData });
});

// --- Comparison Lab ---
const cZone1 = document.getElementById('compare-zone-1');
const cZone2 = document.getElementById('compare-zone-2');
const cInput1 = document.getElementById('compare-input-1');
const cInput2 = document.getElementById('compare-input-2');
const cBtn = document.getElementById('compare-btn');

[cZone1, cZone2].forEach((zone, idx) => {
    zone.addEventListener('click', () => document.getElementById(`compare-input-${idx+1}`).click());
});

[cInput1, cInput2].forEach((input, idx) => {
    input.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (re) => {
                const preview = document.getElementById(`compare-preview-${idx+1}`);
                const placeholder = document.getElementById(`compare-placeholder-${idx+1}`);
                preview.src = re.target.result;
                preview.style.display = 'block';
                placeholder.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
    });
});

cBtn.addEventListener('click', async () => {
    if (!cInput1.files[0] || !cInput2.files[0]) {
        alert("Please upload both identities for comparison.");
        return;
    }
    
    cBtn.innerText = "Analyzing Biometrics...";
    cBtn.disabled = true;
    
    const formData = new FormData();
    formData.append('file1', cInput1.files[0]);
    formData.append('file2', cInput2.files[0]);
    
    try {
        const response = await fetch('/api/compare', { method: 'POST', body: formData });
        const data = await response.json();
        
        const resultCont = document.getElementById('compare-result-container');
        const badge = document.getElementById('compare-badge');
        const details = document.getElementById('compare-details');
        
        resultCont.style.display = 'block';
        if (data.match) {
            badge.innerText = "IDENTITY MATCHED";
            badge.style.background = "#10b981";
            badge.style.color = "white";
        } else {
            badge.innerText = "MISMATCH DETECTED";
            badge.style.background = "#ef4444";
            badge.style.color = "white";
        }
        
        details.innerHTML = `Biometric Distance: <b>${data.distance}</b> | Confidence: <b>${data.confidence}%</b>`;
    } catch (err) {
        alert("Verification failed. Ensure faces are clearly visible.");
    } finally {
        cBtn.innerText = "Run Verification Scan";
        cBtn.disabled = false;
    }
});

// --- Feature Expert (Landmarks) ---
const lZone = document.getElementById('landmark-upload-zone');
const lInput = document.getElementById('landmark-input');
const lPreview = document.getElementById('landmark-preview');
const lCanvas = document.getElementById('landmark-canvas');
const lOutput = document.getElementById('landmark-output');

lZone.addEventListener('click', () => lInput.click());

lInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    // Preview
    const reader = new FileReader();
    reader.onload = async (re) => {
        lPreview.src = re.target.result;
        lPreview.style.display = 'block';
        document.getElementById('landmark-placeholder').style.display = 'none';
        
        lOutput.innerText = "// Extracting biometric features...";
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/landmarks', { method: 'POST', body: formData });
            const data = await response.json();
            
            lOutput.innerText = JSON.stringify(data.landmarks, null, 2);
            drawLandmarks(data.landmarks);
        } catch (err) {
            lOutput.innerText = "// Extraction failed. Please try a clearer high-res photo.";
        }
    };
    reader.readAsDataURL(file);
});

function drawLandmarks(landmarks) {
    const ctx = lCanvas.getContext('2d');
    // Set canvas internal size to match image display size
    lCanvas.width = lPreview.clientWidth;
    lCanvas.height = lPreview.clientHeight;
    
    ctx.clearRect(0, 0, lCanvas.width, lCanvas.height);
    ctx.strokeStyle = '#10b981';
    ctx.fillStyle = '#10b981';
    ctx.lineWidth = 1;
    
    // Natural dimensions of the image vs displayed dimensions
    const scaleX = lCanvas.width / lPreview.naturalWidth;
    const scaleY = lCanvas.height / lPreview.naturalHeight;
    
    landmarks.forEach(face => {
        for (let feature in face) {
            const points = face[feature];
            ctx.beginPath();
            points.forEach((p, i) => {
                const px = p[0] * scaleX;
                const py = p[1] * scaleY;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
                ctx.fillRect(px - 1, py - 1, 2, 2);
            });
            ctx.stroke();
        }
    });
}

// --- Training & Enrollment ---
const enrollBtn = document.getElementById('enroll-btn');
const captureRemoteBtn = document.getElementById('capture-remote-btn');
const enrollNameInput = document.getElementById('enroll-name');
const enrollFileInput = document.createElement('input'); 
enrollFileInput.type = 'file'; enrollFileInput.accept = 'image/*';

// Wrap enroll logic
enrollBtn.addEventListener('click', () => enrollFileInput.click());

enrollFileInput.addEventListener('change', async (e) => {
   const file = e.target.files[0];
   const name = enrollNameInput.value;
   if(!name) { alert("Please enter identity name first."); return; }
   
   const formData = new FormData();
   formData.append('file', file);
   formData.append('name', name);
   
   const response = await fetch('/enroll', { method: 'POST', body: formData });
   const res = await response.json();
   alert(res.message);
   if(res.status === 'success') fetchEnrolledFaces();
});

captureRemoteBtn.addEventListener('click', async () => {
    const name = enrollNameInput.value;
    if(!name) { alert("Please enter identity name first."); return; }
    
    captureRemoteBtn.innerText = "Capturing...";
    const response = await fetch('/capture');
    const blob = await response.blob();
    
    const formData = new FormData();
    formData.append('file', blob, 'remote_capture.jpg');
    formData.append('name', name);
    
    const enrollRes = await fetch('/enroll', { method: 'POST', body: formData });
    const res = await enrollRes.json();
    alert(res.message);
    captureRemoteBtn.innerText = "Remote Capture";
    if(res.status === 'success') fetchEnrolledFaces();
});

// --- Polling & Data ---
async function fetchEnrolledFaces() {
    const response = await fetch('/enrolled');
    const data = await response.json();
    const list = document.getElementById('enrolled-list');
    list.innerHTML = '';
    data.faces.forEach(f => {
        const div = document.createElement('div');
        div.className = 'glass-card';
        div.style.padding = '0.5rem'; div.style.textAlign = 'center'; div.style.fontSize = '0.75rem';
        div.innerHTML = `<p>${f.name}</p>`;
        list.appendChild(div);
    });
}

async function fetchHistory() {
    const response = await fetch('/history');
    const data = await response.json();
    const log = document.getElementById('history-log');
    log.innerHTML = '';
    data.history.forEach(h => {
        const item = document.createElement('div');
        item.className = 'log-item';
        item.innerHTML = `
            <div class="log-avatar">${h.name[0]}</div>
            <div class="log-info">
                <div class="log-name">${h.name}</div>
                <div class="log-time">${h.timestamp}</div>
            </div>
            <div class="log-conf">${Math.round(h.confidence)}%</div>
        `;
        log.appendChild(item);
    });
}

// Stats loop
setInterval(() => {
    fetchHistory();
    fetch('/stats').then(r => r.json()).then(d => {
        document.getElementById('enrolled-count-stat').innerText = d.enrolled;
    });
}, 3000);

// Init
fetchEnrolledFaces();
fetchHistory();
