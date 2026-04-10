// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const changeImageBtn = document.getElementById('changeImageBtn');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsSection = document.getElementById('resultsSection');
const successResult = document.getElementById('successResult');
const rejectedResult = document.getElementById('rejectedResult');
const errorResult = document.getElementById('errorResult');
const tryAgainBtn = document.getElementById('tryAgainBtn');
const openCameraBtn = document.getElementById('openCameraBtn');
const uploadFileBtn = document.getElementById('uploadFileBtn');
const cameraContainer = document.getElementById('cameraContainer');
const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const capturePhotoBtn = document.getElementById('capturePhotoBtn');
const closeCameraBtn = document.getElementById('closeCameraBtn');

let cameraStream = null;

// File Handling
uploadFileBtn.addEventListener('click', () => imageInput.click());

imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleImageSelect(e.target.files[0]);
    }
});

function handleImageSelect(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file');
        return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.classList.remove('hidden');
        uploadArea.style.display = 'none';
    };
    reader.readAsDataURL(file);

    // Upload image
    uploadImage(file);
}

changeImageBtn.addEventListener('click', () => {
    imageInput.click();
});

async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        displayErrorResult('Camera not supported on this device/browser.');
        return;
    }

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: 'environment' }
            },
            audio: false
        });

        cameraVideo.srcObject = cameraStream;
        cameraContainer.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        uploadArea.style.display = 'none';
        resultsSection.classList.add('hidden');
    } catch (error) {
        displayErrorResult('Could not access camera. Please allow camera permission or upload from gallery.');
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    cameraVideo.srcObject = null;
    cameraContainer.classList.add('hidden');
}

function capturePhotoFromCamera() {
    if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
        displayErrorResult('Camera is not ready yet. Please try again.');
        return;
    }

    cameraCanvas.width = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;

    const ctx = cameraCanvas.getContext('2d');
    ctx.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);

    cameraCanvas.toBlob((blob) => {
        if (!blob) {
            displayErrorResult('Failed to capture image from camera.');
            return;
        }

        const capturedFile = new File([blob], `camera-capture-${Date.now()}.jpg`, { type: 'image/jpeg' });
        stopCamera();
        handleImageSelect(capturedFile);
    }, 'image/jpeg', 0.9);
}

openCameraBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    startCamera();
});

capturePhotoBtn.addEventListener('click', capturePhotoFromCamera);

closeCameraBtn.addEventListener('click', () => {
    stopCamera();
    uploadArea.style.display = 'block';
});

async function uploadImage(file) {
    // Show loading spinner
    loadingSpinner.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    successResult.classList.add('hidden');
    rejectedResult.classList.add('hidden');
    errorResult.classList.add('hidden');

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const contentType = response.headers.get('content-type') || '';
        let data = null;

        if (contentType.includes('application/json')) {
            data = await response.json();
        } else {
            const responseText = await response.text();
            data = {
                error: responseText || `Request failed with status ${response.status}`
            };
        }

        // Hide loading spinner
        loadingSpinner.classList.add('hidden');
        resultsSection.classList.remove('hidden');

        if (response.ok) {
            displaySuccessResult(data);
        } else {
            if (data.reason) {
                displayRejectedResult(data);
            } else {
                displayErrorResult(data.message || data.error || `Request failed with status ${response.status}`);
            }
        }
    } catch (error) {
        loadingSpinner.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        displayErrorResult(error.message);
    }
}

function displaySuccessResult(data) {
    successResult.classList.remove('hidden');
    rejectedResult.classList.add('hidden');
    errorResult.classList.add('hidden');

    // Update disease info
    const diseaseInfo = data.disease_info;
    document.getElementById('diseaseEmoji').textContent = diseaseInfo.emoji;
    document.getElementById('diseaseTitle').textContent = diseaseInfo.title;
    document.getElementById('diseaseDescription').textContent = diseaseInfo.description;
    
    const confidence = (data.confidence * 100).toFixed(1);
    document.getElementById('confidenceText').textContent = `Confidence: ${confidence}%`;

    // Display probabilities
    const probabilitiesChart = document.getElementById('probabilitiesChart');
    probabilitiesChart.innerHTML = '';

    for (const [className, probability] of Object.entries(data.all_probabilities)) {
        const percentage = (probability * 100).toFixed(1);
        
        const probabilityItem = document.createElement('div');
        probabilityItem.className = 'probability-item';

        const label = document.createElement('div');
        label.className = 'probability-label';
        label.textContent = className.charAt(0).toUpperCase() + className.slice(1);

        const barContainer = document.createElement('div');
        barContainer.className = 'probability-bar-container';

        const bar = document.createElement('div');
        bar.className = 'probability-bar';
        bar.textContent = `${percentage}%`;
        bar.style.width = `${percentage}%`;

        barContainer.appendChild(bar);

        const percentageText = document.createElement('div');
        percentageText.className = 'probability-percentage';
        percentageText.textContent = `${percentage}%`;

        probabilityItem.appendChild(label);
        probabilityItem.appendChild(barContainer);
        probabilityItem.appendChild(percentageText);

        probabilitiesChart.appendChild(probabilityItem);
    }

    // Display solutions
    const solutionsList = document.getElementById('solutionsList');
    solutionsList.innerHTML = '';

    diseaseInfo.recommended_solutions.forEach(solution => {
        const li = document.createElement('li');
        li.textContent = solution;
        solutionsList.appendChild(li);
    });
}

function displayRejectedResult(data) {
    rejectedResult.classList.remove('hidden');
    successResult.classList.add('hidden');
    errorResult.classList.add('hidden');

    document.getElementById('rejectionTitle').textContent = 
        data.reason === 'NOT_A_LEAF' ? 'Not a Banana Leaf' : 'Low Confidence';
    
    document.getElementById('rejectionMessage').textContent = data.message;
}

function displayErrorResult(message) {
    errorResult.classList.remove('hidden');
    successResult.classList.add('hidden');
    rejectedResult.classList.add('hidden');

    document.getElementById('errorMessage').textContent = 
        message || 'An error occurred while processing your image. Please try again.';
}

tryAgainBtn.addEventListener('click', () => {
    // Reset UI
    stopCamera();
    uploadArea.style.display = 'block';
    previewContainer.classList.add('hidden');
    resultsSection.classList.add('hidden');
    loadingSpinner.classList.add('hidden');
    imageInput.value = '';
});

// Initialize
console.log('🍌 Banana Leaf Disease Classifier Frontend Loaded');
