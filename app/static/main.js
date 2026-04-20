document.addEventListener('DOMContentLoaded', () => {
    // Nav logic
    const navBtns = document.querySelectorAll('.nav-btn');
    const views = document.querySelectorAll('.view');
    
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            navBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            views.forEach(v => v.classList.remove('active'));
            const target = btn.getAttribute('data-target');
            document.getElementById(target).classList.add('active');
            
            if(target === 'history-view') loadHistory();
            if(target === 'stats-view') loadStats();
        });
    });

    // Upload logic
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const dropContent = document.getElementById('drop-content');
    const clearBtn = document.getElementById('clear-btn');
    const predictBtn = document.getElementById('predict-btn');
    
    let currentFile = null;

    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    clearBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        currentFile = null;
        fileInput.value = '';
        imagePreview.src = '';
        imagePreview.classList.add('hidden');
        clearBtn.classList.add('hidden');
        dropContent.classList.remove('hidden');
        predictBtn.disabled = true;
        document.getElementById('result-card').classList.add('hidden');
    });

    function handleFile(file) {
        if (!file.type.match('image.*')) return;
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.remove('hidden');
            clearBtn.classList.remove('hidden');
            dropContent.classList.add('hidden');
            predictBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Prediction logic
    predictBtn.addEventListener('click', async () => {
        if (!currentFile) return;
        
        const loader = document.getElementById('loader');
        const resultCard = document.getElementById('result-card');
        
        loader.classList.remove('hidden');
        resultCard.classList.add('hidden');
        
        const formData = new FormData();
        formData.append('image', currentFile);
        
        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            
            if (data.error) throw new Error(data.error);
            
            displayResult(data);
            
        } catch (err) {
            alert("Error: " + err.message);
        } finally {
            loader.classList.add('hidden');
        }
    });

    function displayResult(data) {
        const resultCard = document.getElementById('result-card');

        // Meta pills
        document.getElementById('inference-time').textContent = `${data.inference_ms}ms`;
        document.getElementById('yolo-boxes').textContent = `${data.yolo_boxes} detection${data.yolo_boxes !== 1 ? 's' : ''}`;

        // Diagnosis header
        const severityBadge = document.getElementById('r-severity');
        const sev = (data.severity || 'unknown').toLowerCase();
        severityBadge.textContent = data.severity || 'Unknown';
        severityBadge.className = `severity-badge ${sev}`;

        document.getElementById('r-disease-name').textContent = data.disease_name;
        document.getElementById('r-crop-type').textContent = `Crop: ${data.crop_type}`;

        // Confidence bar
        document.getElementById('r-confidence-bar').style.width = '0%';
        setTimeout(() => {
            document.getElementById('r-confidence-bar').style.width = `${data.confidence}%`;
        }, 100);
        document.getElementById('r-confidence-val').textContent = `${data.confidence}%`;

        // Condition summary
        document.getElementById('r-description').textContent = data.description || '—';

        // ── AI Analysis Panel ──────────────────────────────────────────
        const ai = data.ai_analysis || {};

        // AI source pill
        const sourcePill = document.getElementById('ai-source-pill');
        const src = ai._source || 'fallback';
        if (src === 'gemini') {
            sourcePill.textContent = '✨ Gemini AI';
            sourcePill.style.background = 'rgba(139,92,246,0.18)';
            sourcePill.style.color = '#a78bfa';
        } else if (src === 'cache') {
            sourcePill.textContent = '✨ Gemini (cached)';
            sourcePill.style.background = 'rgba(139,92,246,0.12)';
            sourcePill.style.color = '#a78bfa';
        } else if (src === 'local-kb') {
            sourcePill.textContent = '📚 Local Knowledge Base';
            sourcePill.style.background = 'rgba(16,185,129,0.12)';
            sourcePill.style.color = '#34d399';
        } else {
            sourcePill.textContent = '';
            sourcePill.style.background = '';
        }

        // Urgency badge
        const urgencyEl = document.getElementById('r-urgency');
        const urgency = ai.urgency || '';
        urgencyEl.textContent = urgency ? `${urgency}` : '';
        urgencyEl.className = `urgency-badge ${getUrgencyClass(urgency)}`;

        renderList('r-symptoms',   ai.key_symptoms        || []);
        renderList('r-organic',    ai.organic_treatments  || [], true);
        renderList('r-chemical',   ai.chemical_treatments || [], true);
        renderList('r-prevention', ai.prevention_tips     || []);

        document.getElementById('r-economic').textContent = ai.economic_impact || '—';
        document.getElementById('r-recovery').textContent = ai.recovery_time   || '—';

        // AI panel response time (clean)
        const panelSrc = document.getElementById('ai-panel-source');
        const panelBadge = document.getElementById('ai-panel-badge');
        if (src === 'gemini' && ai._response_ms) {
            panelSrc.textContent = `${ai._response_ms}ms`;
            panelBadge.textContent = '✨ Gemini AI Analysis';
        } else if (src === 'cache') {
            panelSrc.textContent = 'cached';
            panelBadge.textContent = '✨ Gemini AI Analysis';
        } else if (src === 'local-kb') {
            panelSrc.textContent = 'local database';
            panelBadge.textContent = '📚 Local Knowledge Base';
            panelBadge.style.background = 'linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.1))';
            panelBadge.style.borderColor = 'rgba(16,185,129,0.3)';
            panelBadge.style.color = '#34d399';
        } else {
            panelSrc.textContent = '';
            panelBadge.textContent = '📋 Disease Info';
        }

        resultCard.classList.remove('hidden');
    }

    function renderList(elId, items, ordered = false) {
        const el = document.getElementById(elId);
        if (!items || items.length === 0) {
            el.innerHTML = '<li class="ai-empty">—</li>';
            return;
        }
        el.innerHTML = items.map(item => `<li>${item}</li>`).join('');
    }

    function getUrgencyClass(urgency = '') {
        const u = urgency.toLowerCase();
        if (u.includes('immediate')) return 'urgency-critical';
        if (u.includes('48'))       return 'urgency-high';
        if (u.includes('week'))     return 'urgency-medium';
        return 'urgency-low';
    }

    // History View Native Fetch
    async function loadHistory() {
        try {
            const res = await fetch('/api/history?limit=20');
            const data = await res.json();
            const tbody = document.getElementById('history-body');
            tbody.innerHTML = '';
            
            data.forEach(item => {
                const tr = document.createElement('tr');
                const conf = item.confidence.toFixed(1);
                const imgPath = item.result_image ? item.result_image.replace('app/', '/') : '';
                
                tr.innerHTML = `
                    <td><img src="${imgPath}" class="history-img" onerror="this.src=''"></td>
                    <td>${item.crop_type}</td>
                    <td><strong>${item.disease_name}</strong></td>
                    <td><div class="progress-bar" style="width: 60px; height: 6px; display:inline-block; vertical-align:middle; margin-right:8px;"><div class="progress" style="width:${conf}%"></div></div> ${conf}%</td>
                    <td><span class="severity-badge ${item.severity.toLowerCase()}" style="margin:0">${item.severity}</span></td>
                    <td style="color:var(--text-dim); font-size: 13px;">${item.timestamp}</td>
                `;
                tbody.appendChild(tr);
            });
        } catch(e) {
            console.error('Failed to load history', e);
        }
    }

    // Stats View Native Fetch
    async function loadStats() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();
            
            document.getElementById('stat-total').textContent = data.total_scans;
            
            const cropList = document.getElementById('stats-by-crop');
            cropList.innerHTML = '';
            data.by_crop.forEach(c => {
                cropList.innerHTML += `
                    <li>
                        <span>${c.crop_type}</span>
                        <span><strong>${c.total_scans}</strong> scans</span>
                    </li>
                `;
            });
            
            const disList = document.getElementById('stats-by-disease');
            disList.innerHTML = '';
            data.by_disease.forEach(d => {
                const conf = d.avg_conf ? d.avg_conf.toFixed(1) : 0;
                disList.innerHTML += `
                    <li>
                        <span>${d.disease_name} <span style="color:var(--text-dim); font-size:12px">(${conf}% avg)</span></span>
                        <span><strong>${d.count}</strong> cases</span>
                    </li>
                `;
            });
            
        } catch(e) {
            console.error('Failed to load stats', e);
        }
    }
});
