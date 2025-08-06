let currentTab = 'mcq';

function toggleTheme() {
    const body = document.body;
    const themeIcon = document.getElementById('theme-icon');

    if (body.classList.contains('light-mode')) {
        body.classList.remove('light-mode');
        body.classList.add('dark-mode');
        themeIcon.textContent = '🌙';
    } else {
        body.classList.remove('dark-mode');
        body.classList.add('light-mode');
        themeIcon.textContent = '☀️';
    }
}

function switchTab(tab) {
    currentTab = tab;
    const tabs = document.querySelectorAll('.tab-btn');
    const btnText = document.getElementById('btn-text');

    tabs.forEach(t => t.classList.remove('active'));
    document.querySelector(`[onclick="switchTab('${tab}')"]`).classList.add('active');

    if (tab === 'mcq') {
        btnText.innerHTML = '⚡ Generate MCQs';
    } else {
        btnText.innerHTML = '🎯 Generate Quiz';
    }
}

function handleFileSelect(input) {
    const fileLabel = document.getElementById('file-label');
    if (input.files && input.files[0]) {
        const fileName = input.files[0].name;
        fileLabel.innerHTML = `
                    <div style="font-size: 2rem; margin-bottom: 10px;">📄</div>
                    <div style="font-weight: 600;">${fileName}</div>
                    <div style="font-size: 0.9rem; opacity: 0.7; color: var(--primary-color);">Ready to process!</div>
                `;
        fileLabel.style.borderColor = 'var(--primary-color)';
        fileLabel.style.background = 'rgba(99, 102, 241, 0.05)';

        // Clear YouTube URL if PDF is selected
        document.getElementById('youtube-url').value = '';
    }
}

function handleUrlInput(input) {
    if (input.value.trim()) {
        // Clear PDF upload if URL is entered
        const pdfInput = document.getElementById('pdf-upload');
        pdfInput.value = '';
        const fileLabel = document.getElementById('file-label');
        fileLabel.innerHTML = `
                    <div style="font-size: 2rem; margin-bottom: 10px;">📎</div>
                    <div>Click to upload PDF</div>
                    <div style="font-size: 0.9rem; opacity: 0.7;">or drag and drop</div>
                `;
        fileLabel.style.borderColor = '';
        fileLabel.style.background = '';
    }
}

function generateContent() {
    const pdfFile = document.getElementById('pdf-upload').files[0];
    const youtubeUrl = document.getElementById('youtube-url').value.trim();
    const difficulty = document.getElementById('difficulty').value;
    const numQuestions = document.getElementById('num-questions').value;

    if (!pdfFile && !youtubeUrl) {
        alert('Please upload a PDF or enter a YouTube URL');
        return;
    }

    if (!numQuestions || numQuestions < 5 || numQuestions > 50) {
        alert('Please enter a valid number of questions (5-50)');
        return;
    }

    // Simulate processing
    const btn = document.querySelector('.generate-btn');
    const originalText = btn.innerHTML;
    const contentType = currentTab === 'mcq' ? 'MCQs' : 'Quiz';

    btn.innerHTML = `🔄 Generating ${contentType}...`;
    btn.disabled = true;

    setTimeout(() => {
        btn.innerHTML = originalText;
        btn.disabled = false;

        const source = pdfFile ? 'PDF document' : 'YouTube video';
        const message = `${contentType} generated successfully!\n\n` +
            `📊 Details:\n` +
            `• Source: ${source}\n` +
            `• Type: ${contentType}\n` +
            `• Difficulty: ${difficulty.charAt(0).toUpperCase() + difficulty.slice(1)}\n` +
            `• Questions: ${numQuestions}\n\n`;

        alert(message);
    }, 3000);
}

// Drag and drop functionality
const fileLabel = document.getElementById('file-label');

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    fileLabel.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    fileLabel.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    fileLabel.addEventListener(eventName, unhighlight, false);
});

function highlight() {
    fileLabel.style.borderColor = 'var(--primary-color)';
    fileLabel.style.background = 'rgba(99, 102, 241, 0.1)';
    fileLabel.style.transform = 'scale(1.02)';
}

function unhighlight() {
    fileLabel.style.borderColor = '';
    fileLabel.style.background = '';
    fileLabel.style.transform = '';
}

fileLabel.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;

    if (files.length > 0 && files[0].type === 'application/pdf') {
        document.getElementById('pdf-upload').files = files;
        handleFileSelect(document.getElementById('pdf-upload'));
    }
}

// Initialize number input validation
document.getElementById('num-questions').addEventListener('input', function (e) {
    const value = parseInt(e.target.value);
    if (value < 5) e.target.value = 5;
    if (value > 50) e.target.value = 50;
});