// --- Elite Laboratory Orchestration ---
let activeTab = 'tab-dashboard';
let activeLab = 'neuro';
let gazePoints = [];
const GAZE_RETENTION_MS = 10000;
let recentDetections = new Map();

/** [frameHeight, frameWidth] — matches server landmark / bbox space */
let lastFrameSize = [720, 1280];
let lastDetectionPayload = [];
const sparkBuf = { ear: [], mar: [], fusion: [] };
const SPARK_MAX = 90;

function resizeExpertCanvas() {
    const wrap = document.getElementById('expert-video-wrap');
    const canvas = document.getElementById('expert-overlay');
    if (!wrap || !canvas) return;
    const w = wrap.clientWidth;
    const h = wrap.clientHeight;
    if (w < 2 || h < 2) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function expertContainMap(pt) {
    const wrap = document.getElementById('expert-video-wrap');
    if (!wrap || !pt) return { x: 0, y: 0, scale: 1 };
    const fh = lastFrameSize[0];
    const fw = lastFrameSize[1];
    if (fh < 2 || fw < 2) return { x: 0, y: 0, scale: 1 };
    const dispW = wrap.clientWidth;
    const dispH = wrap.clientHeight;
    const scale = Math.min(dispW / fw, dispH / fh);
    const ox = (dispW - fw * scale) / 2;
    const oy = (dispH - fh * scale) / 2;
    return { x: pt[0] * scale + ox, y: pt[1] * scale + oy, scale };
}

function expertHueForTrack(tid) {
    const t = (tid == null ? 0 : Number(tid)) || 0;
    return `hsl(${((t * 47) % 360)}, 85%, 58%)`;
}

function drawPolyline(ctx, points, mapFn, closed) {
    if (!points || points.length < 2) return;
    ctx.beginPath();
    const p0 = mapFn(points[0]);
    ctx.moveTo(p0.x, p0.y);
    for (let i = 1; i < points.length; i++) {
        const p = mapFn(points[i]);
        ctx.lineTo(p.x, p.y);
    }
    if (closed) ctx.closePath();
    ctx.stroke();
}

function drawFaceMesh(ctx, landmarks, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.25;
    const partsOpen = ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge'];
    partsOpen.forEach((part) => {
        if (landmarks[part] && landmarks[part].length >= 2)
            drawPolyline(ctx, landmarks[part], (pt) => expertContainMap(pt), false);
    });
    if (landmarks.nose_tip && landmarks.nose_tip.length >= 2)
        drawPolyline(ctx, landmarks.nose_tip, (pt) => expertContainMap(pt), false);
    ['left_eye', 'right_eye'].forEach((part) => {
        if (landmarks[part] && landmarks[part].length >= 2)
            drawPolyline(ctx, landmarks[part], (pt) => expertContainMap(pt), true);
    });
    if (landmarks.top_lip && landmarks.top_lip.length >= 2)
        drawPolyline(ctx, landmarks.top_lip, (pt) => expertContainMap(pt), false);
    if (landmarks.bottom_lip && landmarks.bottom_lip.length >= 2)
        drawPolyline(ctx, landmarks.bottom_lip, (pt) => expertContainMap(pt), false);
}

function drawGazeRay(ctx, landmarks, gaze, color) {
    const nt = landmarks.nose_tip;
    if (!nt || !nt.length || !gaze) return;
    const mid = nt[Math.floor(nt.length / 2)];
    const p0 = expertContainMap(mid);
    const len = 55 + 40 * (Math.abs(gaze.gaze_x) + Math.abs(gaze.gaze_y));
    const p1 = { x: p0.x + gaze.gaze_x * len, y: p0.y + gaze.gaze_y * len };
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y);
    ctx.lineTo(p1.x, p1.y);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p1.x, p1.y, 4, 0, Math.PI * 2);
    ctx.fill();
}

function drawExpertScene(detections) {
    const canvas = document.getElementById('expert-overlay');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = canvas.width / Math.max(canvas.clientWidth || 1, 1);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const mesh = document.getElementById('expert-draw-mesh')?.checked !== false;
    const pts = document.getElementById('expert-draw-points')?.checked !== false;
    const gz = document.getElementById('expert-draw-gaze')?.checked !== false;
    const sel = document.getElementById('expert-face-select');
    let idx = sel ? parseInt(sel.value, 10) || 0 : 0;
    if (idx >= detections.length) idx = 0;

    detections.forEach((d, i) => {
        const lm = d.landmarks;
        if (!lm) return;
        const col = expertHueForTrack(d.track_id);
        if (mesh) drawFaceMesh(ctx, lm, col);
        if (pts) {
            ctx.fillStyle = col;
            for (const part in lm) {
                (lm[part] || []).forEach((pt) => {
                    const p = expertContainMap(pt);
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, i === idx ? 2.2 : 1.2, 0, Math.PI * 2);
                    ctx.fill();
                });
            }
        }
        if (gz && i === idx && d.gaze) drawGazeRay(ctx, lm, d.gaze, 'rgba(129, 140, 248, 0.95)');
    });

    if (detections[idx] && detections[idx].bbox) {
        const b = detections[idx].bbox;
        const top = b[0], right = b[1], bottom = b[2], left = b[3];
        const p1 = expertContainMap([left, top]);
        const p2 = expertContainMap([right, bottom]);
        ctx.strokeStyle = 'rgba(255,255,255,0.35)';
        ctx.lineWidth = 1;
        ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
    }
}

function pushSparkBuffers(d) {
    sparkBuf.ear.push(Number(d.ear) || 0);
    sparkBuf.mar.push(Number(d.mar) || 0);
    sparkBuf.fusion.push(Number(d.fusion_presence_score) || 0);
    ['ear', 'mar', 'fusion'].forEach((k) => {
        if (sparkBuf[k].length > SPARK_MAX) sparkBuf[k].shift();
    });
}

function applyExpertSidebar(dx) {
    if (!dx) return;
    const ef = document.getElementById('expert-fusion-val');
    if (ef) ef.innerText = dx.fusion_presence_score != null ? `${Math.round(dx.fusion_presence_score)}` : '—';
    const eff = document.getElementById('expert-fusion-fill');
    if (eff) eff.style.width = `${Math.min(100, dx.fusion_presence_score || 0)}%`;
    const gs = document.getElementById('expert-gaze-sym');
    if (gs) gs.innerText = dx.gaze_symmetry_quality != null ? `${dx.gaze_symmetry_quality}` : '—';
    const vg = document.getElementById('expert-vergence');
    if (vg && dx.gaze_detail) vg.innerText = String(dx.gaze_detail.vergence ?? '—');
    const elr = document.getElementById('expert-ear-lr');
    if (elr) elr.innerText = `${dx.ear_left ?? '—'} / ${dx.ear_right ?? '—'}`;
    const hpEl = document.getElementById('expert-head-pose');
    if (hpEl && dx.head_pose) {
        hpEl.innerText = `Y ${dx.head_pose.yaw_deg}  P ${dx.head_pose.pitch_deg}  R ${dx.head_pose.roll_deg}`;
    }
    const al = document.getElementById('expert-alert-val');
    if (al) {
        al.innerText = dx.alert_state || '—';
        al.style.color =
            dx.alert_state === 'HIGH_RISK' ? 'var(--danger)' :
            dx.alert_state === 'ELEVATED' ? 'var(--warning)' : 'var(--success)';
    }
    const raw = document.getElementById('raw-coords');
    if (raw) raw.innerText = JSON.stringify(dx.landmarks || {}, null, 2);
}

function drawSparkline(canvasId, arr, color, vmin, vmax) {
    const c = document.getElementById(canvasId);
    if (!c || !arr.length) return;
    const ctx = c.getContext('2d');
    const w = c.parentElement ? c.parentElement.clientWidth : 120;
    c.width = Math.max(80, Math.floor(w));
    const h = c.height;
    ctx.clearRect(0, 0, c.width, h);
    ctx.fillStyle = 'rgba(255,255,255,0.04)';
    ctx.fillRect(0, 0, c.width, h);
    let lo = vmin,
        hi = vmax;
    if (hi <= lo) {
        lo = Math.min(...arr) - 0.01;
        hi = Math.max(...arr) + 0.01;
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    arr.forEach((v, i) => {
        const x = (i / Math.max(arr.length - 1, 1)) * (c.width - 4) + 2;
        const t = (v - lo) / (hi - lo);
        const y = h - 4 - t * (h - 8);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
}

function refreshExpertFaceSelect(detections) {
    const sel = document.getElementById('expert-face-select');
    if (!sel) return;
    const n = detections.length;
    const cur = sel.value;
    sel.innerHTML = '';
    for (let i = 0; i < Math.max(n, 1); i++) {
        const opt = document.createElement('option');
        opt.value = String(i);
        const tid = n > 0 && detections[i] ? detections[i].track_id : null;
        opt.textContent = n > 0 ? `Face ${i + 1}  (track ${tid ?? '?'})` : '—';
        sel.appendChild(opt);
    }
    if (cur && parseInt(cur, 10) < n) sel.value = cur;
}

// Tab Navigation Engine
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
        document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
        item.classList.add('active');
        activeTab = item.dataset.tab;
        
        document.querySelectorAll('.tab-content').forEach(tc => {
            tc.classList.toggle('active', tc.id === activeTab);
        });
        if (activeTab === 'tab-history') fetchHistory();
        if (activeTab === 'tab-landmarks') {
            requestAnimationFrame(() => {
                resizeExpertCanvas();
                drawExpertScene(lastDetectionPayload);
            });
        }
    });
});

// Lab Switcher logic (within Hub)
function switchLab(target) {
    activeLab = target;
    document.querySelectorAll('.lab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.lab === target);
    });
    document.querySelectorAll('.lab-content').forEach(content => {
        content.style.display = content.id === `lab-${target}` ? 'block' : 'none';
    });
}

// Landmark Toggle Handler
document.getElementById('landmark-toggle').addEventListener('change', (e) => {
    const formData = new FormData();
    formData.append('state', e.target.checked);
    fetch('/toggle_landmarks', { method: 'POST', body: formData })
        .catch(err => console.error("Toggle Error:", err));
});

// WebSocket Connection
let socket;
function connectWS() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${proto}//${window.location.host}/ws`);
    socket.onopen = () => console.log("Elite Telemetry Connected.");
    socket.onclose = () => setTimeout(connectWS, 2000);
    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === 'biometric_update') updateDashboard(data);
        } catch (e) { console.error("WS Payload Error:", e); }
    };
}

function updateDashboard(data) {
    const detections = data.detections || [];
    if (data.frame_size && data.frame_size.length === 2 && data.frame_size[0] > 0)
        lastFrameSize = data.frame_size;

    const hud = document.getElementById('ai-latency-hud');
    if (hud) hud.innerText = `AI: ${data.ai_latency}ms`;

    lastDetectionPayload = detections;

    if (detections.length > 0) {
        const d = detections[0];
        const psych = d.psych || {};
        const gaze = d.gaze || { gaze_x: 0, gaze_y: 0 };
        const hp = d.head_pose || { yaw_deg: 0, pitch_deg: 0, roll_deg: 0 };

        const hubFus = document.getElementById('hub-fusion-val');
        if (hubFus) hubFus.innerText = d.fusion_presence_score != null ? `${Math.round(d.fusion_presence_score)}` : '—';
        const hubAl = document.getElementById('hub-alert-val');
        if (hubAl) {
            hubAl.innerText = d.alert_state || '—';
            hubAl.style.color =
                d.alert_state === 'HIGH_RISK' ? 'var(--danger)' :
                d.alert_state === 'ELEVATED' ? 'var(--warning)' : 'var(--success)';
        }
        document.body.classList.toggle('alert-mode', d.alert_state === 'HIGH_RISK');
        pushSparkBuffers(d);
        drawSparkline('spark-ear', sparkBuf.ear, '#34d399', 0.12, 0.55);
        drawSparkline('spark-mar', sparkBuf.mar, '#f472b6', 0, 0.65);
        drawSparkline('spark-fusion', sparkBuf.fusion, '#818cf8', 0, 100);

        if (activeTab === 'tab-dashboard') {
            if (activeLab === 'neuro') {
                document.getElementById('saccadic-val').innerText = psych.saccadic_velocity || "0.0";
                document.getElementById('gaze-val').innerText = `X: ${gaze.gaze_x}, Y: ${gaze.gaze_y}`;
                document.getElementById('gaze-stab-val').innerText = d.gaze_stability != null ? `${d.gaze_stability}` : '—';
                document.getElementById('blink-bpm-val').innerText = d.blink_bpm != null ? `${d.blink_bpm}` : '0';
                document.getElementById('head-yaw-val').innerText = `${hp.yaw_deg ?? 0}`;
                document.getElementById('attention-val').innerText = d.attention_score != null ? `${d.attention_score}%` : '—';
                drawHeatmap(gaze);
            } else if (activeLab === 'psych') {
                document.getElementById('stress-score-val').innerText = `${Math.round(psych.stress_score)}%`;
                document.getElementById('stress-score-fill').style.width = `${psych.stress_score}%`;
                document.getElementById('emotion-badge-large').innerText = (d.emotion || 'Neutral').toUpperCase();
                document.getElementById('emotion-conf-val').innerText = d.emotion_confidence != null ? `${Math.round(d.emotion_confidence)}%` : '—';
                const micro = d.micro_expression_activity != null ? Number(d.micro_expression_activity).toFixed(1) : '0';
                document.getElementById('micro-pulse-val').innerText = micro;
                drawVAPlane(psych.valence, psych.arousal);
            } else if (activeLab === 'cognitive') {
                document.getElementById('cognitive-load-val').innerText = `${Math.round(psych.cognitive_load)}%`;
                document.getElementById('cognitive-load-fill').style.width = `${psych.cognitive_load}%`;
                document.getElementById('perclos-val').innerText = `${Math.round(psych.perclos)}%`;
                document.getElementById('perclos-fill').style.width = `${psych.perclos}%`;
                const fat = d.fatigue_index != null ? Math.round(d.fatigue_index) : 0;
                document.getElementById('fatigue-val').innerText = `${fat}%`;
                document.getElementById('fatigue-fill').style.width = `${fat}%`;
            }
        } else if (activeTab === 'tab-landmarks') {
            const sel = document.getElementById('expert-face-select');
            const idx = sel ? parseInt(sel.value, 10) || 0 : 0;
            const dx = detections[idx] || d;
            refreshExpertFaceSelect(detections);
            applyExpertSidebar(dx);
            requestAnimationFrame(() => {
                resizeExpertCanvas();
                drawExpertScene(detections);
            });
        }

        updateDiscoveryLog(detections);
    } else {
        document.body.classList.remove('alert-mode');
        const hubFus = document.getElementById('hub-fusion-val');
        if (hubFus) hubFus.innerText = '—';
        const hubAl = document.getElementById('hub-alert-val');
        if (hubAl) {
            hubAl.innerText = '—';
            hubAl.style.color = 'var(--text-muted)';
        }
        if (activeTab === 'tab-landmarks') {
            const raw = document.getElementById('raw-coords');
            if (raw) raw.innerText = 'No face in frame.';
            refreshExpertFaceSelect([]);
            requestAnimationFrame(() => {
                resizeExpertCanvas();
                drawExpertScene([]);
            });
        }
    }
}

function updateDiscoveryLog(detections) {
    const log = document.getElementById('history-log');
    detections.forEach(d => {
        const now = Date.now();
        if (!recentDetections.has(d.name) || (now - recentDetections.get(d.name) > 3000)) {
            const item = document.createElement('div');
            item.className = 'log-item';
            item.style.borderLeft = `4px solid ${d.name === 'Unknown' ? 'var(--danger)' : 'var(--success)'}`;
            item.style.background = 'rgba(255,255,255,0.03)';
            item.style.padding = '1rem';
            item.style.borderRadius = '8px';
            const tid = d.track_id != null ? ` #${d.track_id}` : '';
            item.innerHTML = `<strong>${d.name}${tid}</strong><br><small>${d.emotion} | match ${Math.round(d.confidence || 0)}% | attn ${d.attention_score != null ? Math.round(d.attention_score) : '—'}%</small>`;
            log.prepend(item);
            recentDetections.set(d.name, now);
            if (log.children.length > 10) log.removeChild(log.lastChild);
        }
    });
}

function drawHeatmap(gaze) {
    const canvas = document.getElementById('gaze-heatmap');
    const ctx = canvas.getContext('2d');
    gazePoints.push({ x: gaze.gaze_x, y: gaze.gaze_y, t: Date.now() });
    gazePoints = gazePoints.filter(p => Date.now() - p.t < GAZE_RETENTION_MS);
    ctx.clearRect(0,0,canvas.width, canvas.height);
    gazePoints.forEach(p => {
        const x = (p.x + 1) * (canvas.width/2);
        const y = (p.y + 1) * (canvas.height/2);
        ctx.fillStyle = 'rgba(99, 102, 241, 0.05)';
        ctx.beginPath(); ctx.arc(x, y, 15, 0, Math.PI*2); ctx.fill();
    });
}

function drawVAPlane(v, a) {
    const canvas = document.getElementById('va-plane');
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0,0,w,h);
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.beginPath(); ctx.moveTo(w/2, 0); ctx.lineTo(w/2, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, h/2); ctx.lineTo(w, h/2); ctx.stroke();
    const px = (v + 1) * (w/2);
    const py = (1 - a) * (h/2);
    ctx.fillStyle = 'var(--primary)';
    ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI*2); ctx.fill();
}

// --- Comparison Lab Orchestration ---
const file1 = document.getElementById('file1');
const file2 = document.getElementById('file2');
file1.onchange = e => handlePreview(e.target.files[0], 'preview1', 'ph1');
file2.onchange = e => handlePreview(e.target.files[0], 'preview2', 'ph2');

function handlePreview(file, imgId, phId) {
    const reader = new FileReader();
    reader.onload = e => {
        document.getElementById(imgId).src = e.target.result;
        document.getElementById(imgId).style.display = 'block';
        document.getElementById(phId).style.display = 'none';
    };
    reader.readAsDataURL(file);
}

document.getElementById('compare-btn').onclick = async () => {
    if (!file1.files[0] || !file2.files[0]) return alert("Select 2 images first.");
    const formData = new FormData();
    formData.append('file1', file1.files[0]);
    formData.append('file2', file2.files[0]);
    
    document.getElementById('compare-result').innerText = "VERIFYING PATTERNS...";
    const res = await fetch('/compare', { method: 'POST', body: formData }).then(r => r.json());
    if (res.status === 'success') {
        const color = res.match ? 'var(--success)' : 'var(--danger)';
        document.getElementById('compare-result').innerHTML = `<span style="color:${color}">${res.match ? 'MATCH DETECTED' : 'NO MATCH'}</span><br><small>SIMILARITY: ${res.confidence}% &nbsp;|&nbsp; DIST: ${res.distance}</small>`;
    } else {
        alert(res.message);
    }
};

// --- Training Center Orchestration ---
document.getElementById('subject-file').onchange = e => handlePreview(e.target.files[0], 'enroll-preview-img', 'enroll-ph');
document.getElementById('enroll-form').onsubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append('name', document.getElementById('subject-name').value);
    formData.append('file', document.getElementById('subject-file').files[0]);
    
    document.getElementById('enroll-status').innerText = "ENROLLING...";
    const res = await fetch('/enroll', { method: 'POST', body: formData }).then(r => r.json());
    document.getElementById('enroll-status').innerText = res.message;
    if(res.status === 'success') fetchEnrolled();
};

// --- Audit & Exports ---
document.getElementById('download-csv-btn').onclick = () => window.location.href = '/export_csv';

async function fetchEnrolled() {
    try {
        const r = await fetch('/faces');
        const data = await r.json();
        document.getElementById('enrolled-count-stat').innerText = data.faces.length;
        const list = document.getElementById('enrolled-list');
        list.innerHTML = data.faces.map(f => `<div style="background:rgba(255,255,255,0.05); padding:1rem; border-radius:10px; text-align:center;"><strong>${f.name}</strong></div>`).join('');
    } catch (e) { console.error("Enrollment fetch failed."); }
}

async function fetchHistory() {
    try {
        const r = await fetch('/history');
        const data = await r.json();
        document.getElementById('full-history-log').innerHTML = data.history.map(h => {
            let extra = '';
            if (h.metrics_json) {
                try {
                    const m = typeof h.metrics_json === 'string' ? JSON.parse(h.metrics_json) : h.metrics_json;
                    const yaw = m.head_pose && m.head_pose.yaw_deg != null ? `yaw ${m.head_pose.yaw_deg}°` : '';
                    extra = [yaw, m.attention_score != null ? `attn ${Math.round(m.attention_score)}%` : '', m.fatigue_index != null ? `fatigue ${Math.round(m.fatigue_index)}` : '', m.fusion_presence_score != null ? `fusion ${Math.round(m.fusion_presence_score)}` : '', m.alert_state ? String(m.alert_state) : ''].filter(Boolean).join(' · ');
                } catch (_) {}
            }
            return `
            <div class="glass-card" style="padding:1rem;">
                <strong>${h.name}</strong><br><small>${h.timestamp} | ${h.emotion || ''} | conf ${h.confidence != null ? Math.round(h.confidence) : '—'}%</small>
                ${extra ? `<br><small style="color:var(--text-muted);">${extra}</small>` : ''}
            </div>`;
        }).join('');
    } catch (e) { console.error("History fetch failed."); }
}

function wireExpertControls() {
    const ev = document.getElementById('expert-video-feed');
    if (ev) {
        ev.addEventListener('load', () => {
            requestAnimationFrame(() => {
                resizeExpertCanvas();
                drawExpertScene(lastDetectionPayload);
            });
        });
    }
    window.addEventListener('resize', () => {
        if (activeTab === 'tab-landmarks') {
            requestAnimationFrame(() => {
                resizeExpertCanvas();
                drawExpertScene(lastDetectionPayload);
            });
        }
    });
    ['expert-draw-mesh', 'expert-draw-points', 'expert-draw-gaze'].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', () => drawExpertScene(lastDetectionPayload));
    });
    const sel = document.getElementById('expert-face-select');
    if (sel) {
        sel.addEventListener('change', () => {
            const i = parseInt(sel.value, 10) || 0;
            applyExpertSidebar(lastDetectionPayload[i]);
            drawExpertScene(lastDetectionPayload);
        });
    }
}

// Entry Point
wireExpertControls();
connectWS();
fetchEnrolled();
setInterval(fetchEnrolled, 10000);
