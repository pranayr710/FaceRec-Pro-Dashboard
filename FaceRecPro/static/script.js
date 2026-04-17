// State Management
const state = {
    activeTab: 'live',
    enrolledFaces: [],
    history: [],
    detections: []
};

// UI Elements
const tabs = document.querySelectorAll('.nav-item');
const sections = document.querySelectorAll('.dashboard-section');
const logList = document.getElementById('recognition-log');
const enrolledList = document.getElementById('enrolled-list');
const enrolledCount = document.getElementById('enrolled-count');
const enrollForm = document.getElementById('enroll-form');

// Tab Switching
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const target = tab.getAttribute('data-tab');
        state.activeTab = target;
        
        tabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        
        sections.forEach(s => s.classList.remove('active'));
        document.getElementById(`tab-${target}`).classList.add('active');
        
        if (target === 'training') fetchEnrolledFaces();
        if (target === 'log') fetchHistory();
    });
});

// Fetch Enrolled Faces
async function fetchEnrolledFaces() {
    try {
        const response = await fetch('/faces');
        const data = await response.json();
        state.enrolledFaces = data;
        renderEnrolled();
        enrolledCount.innerText = data.length;
    } catch (err) {
        console.error("Error fetching faces:", err);
    }
}

function renderEnrolled() {
    enrolledList.innerHTML = state.enrolledFaces.map(face => `
        <div style="display: flex; flex-direction: column; align-items: center; gap: 0.25rem;">
            <div style="width: 40px; height: 40px; background: var(--primary); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.875rem;">
                ${face.name[0]}
            </div>
            <div style="font-size: 0.65rem; color: var(--text-muted); text-align: center; max-width: 60px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                ${face.name}
            </div>
        </div>
    `).join('');
    
    const badge = document.getElementById('enrolled-count-badge');
    if (badge) badge.innerText = `${state.enrolledFaces.length} Total`;
}

// Enrollment - Remote Capture Logic
const toggleWebcamBtn = document.getElementById('toggle-webcam-enroll');
const webcamContainer = document.getElementById('enroll-webcam-container');
const enrollVideo = document.getElementById('enroll-video'); // We'll hide this and use an img
const captureBtn = document.getElementById('capture-snapshot-btn');
const snapshotCanvas = document.getElementById('enroll-snapshot-canvas');

// Replace video with a preview of the main stream for simplicity
let previewImg = null;

toggleWebcamBtn.addEventListener('click', () => {
    if (webcamContainer.style.display === 'none') {
        webcamContainer.style.display = 'flex';
        toggleWebcamBtn.innerText = 'Stop Preview';
        
        // Use the same video feed as preview
        if (!previewImg) {
            previewImg = document.createElement('img');
            previewImg.src = "/video_feed";
            previewImg.style.width = "100%";
            previewImg.style.height = "100%";
            previewImg.style.objectFit = "cover";
            document.getElementById('enroll-webcam-container').querySelector('div').appendChild(previewImg);
            enrollVideo.style.display = 'none';
        }
    } else {
        webcamContainer.style.display = 'none';
        toggleWebcamBtn.innerText = 'Use Webcam';
    }
});

captureBtn.addEventListener('click', async () => {
    const name = document.getElementById('enroll-name').value;
    if (!name) {
        alert("Please enter a name first.");
        return;
    }

    captureBtn.innerText = "📸 Capturing...";
    captureBtn.disabled = true;

    try {
        // Fetch a fresh frame from the backend
        const response = await fetch('/capture');
        const blob = await response.blob();
        
        const formData = new FormData();
        formData.append('name', name);
        formData.append('file', blob, 'capture.jpg');
        
        await sendEnrollment(formData);
    } catch (err) {
        alert("Failed to capture from camera.");
    } finally {
        captureBtn.innerText = "Capture & Enroll";
        captureBtn.disabled = false;
        webcamContainer.style.display = 'none';
        toggleWebcamBtn.innerText = 'Use Webcam';
    }
});

// File input behavior
document.getElementById('enroll-file').addEventListener('change', (e) => {
    const submitBtn = document.getElementById('submit-enroll-file');
    if (e.target.files.length > 0) {
        submitBtn.style.display = 'block';
    } else {
        submitBtn.style.display = 'none';
    }
});

async function sendEnrollment(formData) {
    const status = document.getElementById('enroll-status');
    status.innerHTML = '<div style="color: var(--primary);">⏳ Training system...</div>';
    
    try {
        const response = await fetch('/enroll', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (response.ok) {
            status.innerHTML = `<div style="color: var(--success);">✅ ${data.message}</div>`;
            fetchEnrolledFaces();
            enrollForm.reset();
            document.getElementById('submit-enroll-file').style.display = 'none';
        } else {
            status.innerHTML = `<div style="color: var(--warning);">❌ ${data.message}</div>`;
        }
    } catch (err) {
        status.innerHTML = `<div style="color: var(--warning);">❌ Connection error.</div>`;
    }
}

// Enrollment Form (File submit)
enrollForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('name', document.getElementById('enroll-name').value);
    formData.append('file', document.getElementById('enroll-file').files[0]);
    await sendEnrollment(formData);
});

// History / Log
async function fetchHistory() {
    try {
        const response = await fetch('/history');
        const data = await response.json();
        state.history = data;
        renderLog();
    } catch (err) {
        console.error("Error fetching history:", err);
    }
}

function renderLog() {
    const container = state.activeTab === 'live' ? logList : document.getElementById('tab-log-list');
    if (!container && state.activeTab === 'live') return;
    
    logList.innerHTML = state.history.map(item => {
        const isUnknown = item.name === 'Unknown';
        const color = isUnknown ? 'var(--warning)' : 'var(--success)';
        return `
            <div class="log-item">
                <div>
                    <div style="font-weight: 600; font-size: 0.875rem; color: ${color};">
                        <span style="display:inline-block; width: 8px; height: 8px; border-radius: 50%; background: ${color}; margin-right: 8px;"></span>
                        ${item.name}
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${item.confidence}%; background: ${color};"></div>
                    </div>
                    <div style="font-size: 0.75rem; color: var(--text-muted); min-width: 40px; text-align: right;">
                        ${item.confidence}%
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// Stats Polling & Live Logic
setInterval(() => {
    if (state.activeTab === 'live') {
        fetchHistory();
        // Update mock stats
        document.getElementById('faces-detected-count').innerText = Math.floor(Math.random() * 2);
    }
}, 3000);

// Sensitivity Slider
const toleranceSlider = document.getElementById('tolerance-slider');
const toleranceValue = document.getElementById('tolerance-value');

toleranceSlider.addEventListener('input', async (e) => {
    const val = e.target.value;
    toleranceValue.innerText = val;
    
    const formData = new FormData();
    formData.append('value', val);
    
    try {
        await fetch('/set_tolerance', {
            method: 'POST',
            body: formData
        });
    } catch (err) {
        console.error("Error setting tolerance:", err);
    }
});

// Stats polling
setInterval(async () => {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        
        // Match IDs from index.html
        const enrolledEl = document.querySelectorAll('.stat-value')[1]; // '2 Enrolled'
        const recentEl = document.querySelectorAll('.stat-value')[2]; // '38 Recognitions'
        
        if (enrolledEl) enrolledEl.innerText = data.enrolled;
        if (recentEl) recentEl.innerText = data.recent_activity;
    } catch (err) {
        console.error("Stats error", err);
    }
}, 3000);

// Initialize
fetchEnrolledFaces();
fetchHistory();
