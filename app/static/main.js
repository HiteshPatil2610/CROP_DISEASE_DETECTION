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
        
        // Images and stats — fall back to the uploaded preview if no result image
        const annotatedImg = document.getElementById('annotated-image');
        if (data.result_image) {
            annotatedImg.src = data.result_image.replace(/\\/g, '/').replace('app/', '/');
        } else {
            annotatedImg.src = imagePreview.src; // fallback: original uploaded image
        }
        document.getElementById('yolo-boxes').textContent = `Boxes: ${data.yolo_boxes}`;
        document.getElementById('inference-time').textContent = `Time: ${data.inference_ms}ms`;
        
        // Diagnosis
        const severityBadge = document.getElementById('r-severity');
        severityBadge.textContent = data.severity;
        severityBadge.className = `severity-badge ${data.severity.toLowerCase()}`;
        
        document.getElementById('r-disease-name').textContent = data.disease_name;
        document.getElementById('r-crop-type').textContent = `Crop: ${data.crop_type}`;
        
        // Confidence
        document.getElementById('r-confidence-bar').style.width = '0%';
        setTimeout(() => {
            document.getElementById('r-confidence-bar').style.width = `${data.confidence}%`;
        }, 100);
        document.getElementById('r-confidence-val').textContent = `${data.confidence}%`;
        
        // Text
        document.getElementById('r-description').textContent = data.description;
        document.getElementById('r-treatment').textContent = data.treatment;
        
        resultCard.classList.remove('hidden');
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
