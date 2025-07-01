// Main JavaScript for Poverty Analysis Application

// Global variables
let currentFile = null;
let currentResults = null;
let loadingModal = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize application
    initializeApp();
    
    // Set up event listeners
    setupEventListeners();
    
    // Load initial data
    loadModelInfo();
    loadSystemStatus();
});

function initializeApp() {
    console.log('Initializing Poverty Analysis Application...');
    
    // Initialize Bootstrap modal
    loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
    
    // Set up drag and drop functionality
    setupDragAndDrop();
    
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Update model descriptions
    updateModelDescription();
    
    // Check system health
    checkSystemHealth();
}

function setupEventListeners() {
    // File input change
    document.getElementById('fileInput').addEventListener('change', handleFileSelect);
    
    // Model type change
    document.getElementById('modelType').addEventListener('change', updateModelDescription);
    
    // Form submission
    document.getElementById('uploadForm').addEventListener('submit', handleFormSubmission);
    
    // Upload area drag and drop
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => document.getElementById('fileInput').click());
}

function setupDragAndDrop() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight(e) {
        uploadArea.classList.add('drag-over');
    }
    
    function unhighlight(e) {
        uploadArea.classList.remove('drag-over');
    }
    
    uploadArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect();
        }
    }
}

function handleFileSelect() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (file) {
        displayFileInfo(file);
        document.getElementById('predictBtn').disabled = false;
    } else {
        clearFile();
    }
}

function displayFileInfo(file) {
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    
    fileName.textContent = `${file.name} (${formatFileSize(file.size)})`;
    fileInfo.classList.remove('d-none');
}

function clearFile() {
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    
    fileInput.value = '';
    fileInfo.classList.add('d-none');
    document.getElementById('predictBtn').disabled = true;
    
    // Reset upload area
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.innerHTML = `
        <div class="upload-content">
            <i class="fas fa-cloud-upload-alt fa-3x text-muted mb-3"></i>
            <p class="mb-2">Arrastra y suelta tu archivo Excel aquí</p>
            <p class="text-muted small">o haz clic para seleccionar</p>
            <button type="button" class="btn btn-outline-primary" onclick="document.getElementById('fileInput').click()">
                Seleccionar Archivo
            </button>
        </div>
    `;
    
    // Hide results and validation sections
    document.getElementById('resultsSection').classList.add('d-none');
    document.getElementById('validationSection').classList.add('d-none');
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function updateModelDescription() {
    const modelType = document.getElementById('modelType').value;
    const description = document.getElementById('modelDescription');
    
    const descriptions = {
        'neural': 'Red Neuronal: Análisis avanzado con alta precisión y capacidad de aprendizaje complejo',
        'logistic': 'Modelo Logístico: Análisis estadístico robusto para clasificación binaria',
        'linear': 'Modelo Lineal: Análisis simple y rápido para tendencias básicas'
    };
    
    description.textContent = descriptions[modelType] || descriptions['neural'];
}

function handleFormSubmission(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const file = formData.get('file');
    
    if (!file) {
        showAlert('Por favor selecciona un archivo', 'warning');
        return;
    }
    
    // Show loading modal
    showLoadingModal();
    
    // Send request
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoadingModal();
        
        if (data.success) {
            displayResults(data);
        } else {
            showAlert(data.error || 'Error en el análisis', 'danger');
        }
    })
    .catch(error => {
        hideLoadingModal();
        console.error('Error:', error);
        showAlert('Error de conexión', 'danger');
    });
}

function displayResults(data) {
    // Display analysis section
    displayAnalysisSection(data.analysis);
    
    // Display prediction results
    displayPredictionResults(data);
    
    // Show results section
    document.getElementById('resultsSection').classList.remove('d-none');
    document.getElementById('analysisSection').classList.remove('d-none');
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

function displayAnalysisSection(analysis) {
    if (!analysis || !analysis.success) return;
    
    // Display data quality cards
    displayDataQualityCards(analysis.statistics);
    
    // Display insights
    displayInsights(analysis.insights);
    
    // Display feature importance
    displayFeatureImportance(analysis.feature_importance);
}

function displayDataQualityCards(statistics) {
    const container = document.getElementById('dataQualityCards');
    
    const cards = [
        {
            title: 'Total de Registros',
            value: statistics.total_records,
            icon: 'fas fa-database',
            color: 'primary'
        },
        {
            title: 'Calidad de Datos',
            value: `${(statistics.data_quality_score * 100).toFixed(1)}%`,
            icon: 'fas fa-chart-line',
            color: statistics.data_quality_score > 0.8 ? 'success' : 'warning'
        },
        {
            title: 'Datos Faltantes',
            value: `${statistics.missing_data_percentage.toFixed(1)}%`,
            icon: 'fas fa-exclamation-triangle',
            color: statistics.missing_data_percentage < 5 ? 'success' : 'warning'
        },
        {
            title: 'Características',
            value: statistics.total_features,
            icon: 'fas fa-list',
            color: 'info'
        }
    ];
    
    container.innerHTML = cards.map(card => `
        <div class="col-md-3 col-sm-6 mb-3">
            <div class="card border-${card.color} h-100">
                <div class="card-body text-center">
                    <i class="${card.icon} fa-2x text-${card.color} mb-2"></i>
                    <h5 class="card-title">${card.value}</h5>
                    <p class="card-text small">${card.title}</p>
                </div>
            </div>
        </div>
    `).join('');
}

function displayInsights(insights) {
    const container = document.getElementById('insightsList');
    
    if (!insights || insights.length === 0) {
        container.innerHTML = '<p class="text-muted">No se encontraron insights específicos.</p>';
        return;
    }
    
    container.innerHTML = insights.map(insight => `
        <div class="alert alert-${getInsightColor(insight.type)} alert-dismissible fade show" role="alert">
            <i class="fas ${getInsightIcon(insight.type)} me-2"></i>
            <strong>${insight.title}:</strong> ${insight.message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `).join('');
}

function getInsightColor(type) {
    const colors = {
        'success': 'success',
        'warning': 'warning',
        'info': 'info',
        'error': 'danger'
    };
    return colors[type] || 'info';
}

function getInsightIcon(type) {
    const icons = {
        'success': 'fa-check-circle',
        'warning': 'fa-exclamation-triangle',
        'info': 'fa-info-circle',
        'error': 'fa-times-circle'
    };
    return icons[type] || 'fa-info-circle';
}

function displayFeatureImportance(features) {
    const container = document.getElementById('featureImportanceList');
    
    if (!features || features.length === 0) {
        container.innerHTML = '<p class="text-muted">No se pudo calcular la importancia de características.</p>';
        return;
    }
    
    const topFeatures = features.slice(0, 5);
    
    container.innerHTML = `
        <div class="row">
            ${topFeatures.map((feature, index) => `
                <div class="col-md-6 mb-2">
                    <div class="d-flex justify-content-between align-items-center">
                        <span class="badge bg-primary me-2">${index + 1}</span>
                        <span class="flex-grow-1">${feature[0]}</span>
                        <span class="badge bg-secondary">${(feature[1] * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

function displayPredictionResults(data) {
    // Display summary cards
    displaySummaryCards(data.summary);
    
    // Display prediction quality
    displayPredictionQuality(data.summary.prediction_quality);
    
    // Display recommendations
    displayRecommendations(data.summary.recommendations);
    
    // Display results table
    displayResultsTable(data.predictions);
}

function displaySummaryCards(summary) {
    const container = document.getElementById('summaryCards');
    
    const cards = [
        {
            title: 'Total Analizado',
            value: summary.total_records,
            icon: 'fas fa-users',
            color: 'primary'
        },
        {
            title: 'Predicción Pobreza',
            value: `${summary.poverty_percentage.toFixed(1)}%`,
            icon: 'fas fa-exclamation-triangle',
            color: summary.poverty_percentage > 20 ? 'danger' : 'warning'
        },
        {
            title: 'Confianza Promedio',
            value: `${(summary.average_confidence * 100).toFixed(1)}%`,
            icon: 'fas fa-chart-line',
            color: summary.average_confidence > 0.8 ? 'success' : 'info'
        },
        {
            title: 'Modelo Usado',
            value: summary.model_used,
            icon: 'fas fa-cogs',
            color: 'secondary'
        }
    ];
    
    container.innerHTML = cards.map(card => `
        <div class="col-md-3 col-sm-6 mb-3">
            <div class="card border-${card.color} h-100">
                <div class="card-body text-center">
                    <i class="${card.icon} fa-2x text-${card.color} mb-2"></i>
                    <h5 class="card-title">${card.value}</h5>
                    <p class="card-text small">${card.title}</p>
                </div>
            </div>
        </div>
    `).join('');
}

function displayPredictionQuality(quality) {
    const container = document.getElementById('predictionQualityInfo');
    
    if (!quality) return;
    
    const qualityColor = {
        'Excelente': 'success',
        'Buena': 'info',
        'Aceptable': 'warning',
        'Necesita Mejora': 'danger'
    }[quality.quality_level] || 'secondary';
    
    container.innerHTML = `
        <div class="alert alert-${qualityColor}">
            <div class="row">
                <div class="col-md-6">
                    <h6>Nivel de Calidad: <span class="badge bg-${qualityColor}">${quality.quality_level}</span></h6>
                    <p class="mb-1">Confianza Promedio: ${(quality.average_confidence * 100).toFixed(1)}%</p>
                    <p class="mb-1">Alta Confianza: ${(quality.high_confidence_rate * 100).toFixed(1)}%</p>
                </div>
                <div class="col-md-6">
                    <h6>Distribución de Confianza:</h6>
                    <div class="small">
                        ${Object.entries(quality.confidence_distribution).map(([level, rate]) => 
                            `<div>${level}: ${(rate * 100).toFixed(1)}%</div>`
                        ).join('')}
                    </div>
                </div>
            </div>
        </div>
    `;
}

function displayRecommendations(recommendations) {
    const container = document.getElementById('recommendationsList');
    
    if (!recommendations || recommendations.length === 0) {
        container.innerHTML = '<p class="text-muted">No hay recomendaciones específicas.</p>';
        return;
    }
    
    container.innerHTML = recommendations.map(rec => `
        <div class="alert alert-info">
            <i class="fas fa-lightbulb me-2"></i>
            ${rec}
        </div>
    `).join('');
}

function displayResultsTable(predictions) {
    const tbody = document.getElementById('resultsTableBody');
    
    tbody.innerHTML = predictions.map(pred => `
        <tr>
            <td>${pred.persona_key}</td>
            <td>
                <span class="badge ${pred.prediccion_pobreza == 1 ? 'bg-danger' : 'bg-success'}">
                    ${pred.estado_pobreza}
                </span>
            </td>
            <td>${pred.estado_pobreza}</td>
            <td>${(pred.probabilidad_pobreza * 100).toFixed(1)}%</td>
            <td>
                <div class="progress" style="height: 20px;">
                    <div class="progress-bar bg-${getConfidenceColor(pred.confianza)}" 
                         style="width: ${(pred.confianza * 100)}%">
                        ${(pred.confianza * 100).toFixed(0)}%
                    </div>
                </div>
            </td>
            <td>
                <span class="badge bg-${getRiskColor(pred.nivel_riesgo)}">
                    ${pred.nivel_riesgo}
                </span>
            </td>
            <td>${pred.modelo_usado}</td>
        </tr>
    `).join('');
}

function getConfidenceColor(confidence) {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'danger';
}

function getRiskColor(risk) {
    const colors = {
        'Bajo': 'success',
        'Moderado': 'warning',
        'Alto': 'danger',
        'Muy Alto': 'dark'
    };
    return colors[risk] || 'secondary';
}

function validateFile() {
    const formData = new FormData();
    const fileInput = document.getElementById('fileInput');
    
    if (!fileInput.files[0]) {
        showAlert('Por favor selecciona un archivo primero', 'warning');
        return;
    }
    
    formData.append('file', fileInput.files[0]);
    
    fetch('/validate', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            if (data.is_valid) {
                showAlert('Archivo válido. Puedes proceder con el análisis.', 'success');
                if (data.quick_analysis) {
                    displayQuickAnalysis(data.quick_analysis);
                }
            } else {
                showAlert('Archivo inválido: ' + data.errors.join(', '), 'danger');
            }
        } else {
            showAlert(data.error, 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error de validación', 'danger');
    });
}

function displayQuickAnalysis(analysis) {
    // Display quick analysis results
    console.log('Quick analysis:', analysis);
}

function downloadTemplate() {
    window.location.href = '/download_template';
}

function exportToExcel() {
    // Implementation for Excel export
    showAlert('Función de exportación a Excel en desarrollo', 'info');
}

function exportToCSV() {
    // Implementation for CSV export
    showAlert('Función de exportación a CSV en desarrollo', 'info');
}

function exportAnalysisReport() {
    // Implementation for PDF report export
    showAlert('Función de reporte PDF en desarrollo', 'info');
}

function loadModelInfo() {
    fetch('/models')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayModelInfo(data.models);
            }
        })
        .catch(error => console.error('Error loading model info:', error));
}

function displayModelInfo(models) {
    const container = document.getElementById('modelInfo');
    
    if (!models || Object.keys(models).length === 0) {
        container.innerHTML = '<p class="text-warning">No hay modelos disponibles</p>';
        return;
    }
    
    container.innerHTML = Object.entries(models).map(([name, info]) => `
        <div class="mb-2">
            <strong>${name}</strong>
            <span class="badge bg-${info.type === 'neural' ? 'primary' : 'secondary'} ms-2">
                ${info.type}
            </span>
        </div>
    `).join('');
}

function loadSystemStatus() {
    fetch('/health')
        .then(response => response.json())
        .then(data => {
            displaySystemStatus(data);
        })
        .catch(error => {
            console.error('Error loading system status:', error);
            displaySystemStatus({ status: 'error' });
        });
}

function displaySystemStatus(status) {
    const container = document.getElementById('systemStatus');
    const statusText = document.getElementById('status-text');
    
    if (status.status === 'healthy') {
        statusText.textContent = 'Sistema Activo';
        container.innerHTML = `
            <div class="text-success">
                <i class="fas fa-check-circle me-2"></i>
                Sistema funcionando correctamente
            </div>
            <div class="small text-muted mt-2">
                <div>Modelos disponibles: ${status.available_models}</div>
                <div>Última actualización: ${new Date(status.timestamp).toLocaleString()}</div>
            </div>
        `;
    } else {
        statusText.textContent = 'Sistema Inactivo';
        container.innerHTML = `
            <div class="text-danger">
                <i class="fas fa-times-circle me-2"></i>
                Error en el sistema
            </div>
        `;
    }
}

function showLoadingModal() {
    loadingModal.show();
}

function hideLoadingModal() {
    loadingModal.hide();
}

function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insert at the top of the container
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

function checkSystemHealth() {
    fetch('/health')
    .then(response => response.json())
    .then(data => {
        const statusText = document.getElementById('status-text');
        if (data.status === 'healthy') {
            statusText.textContent = `Sistema Activo (${data.available_models} modelos)`;
        } else {
            statusText.textContent = 'Sistema Inactivo';
        }
    })
    .catch(error => {
        console.error('Health check failed:', error);
        document.getElementById('status-text').textContent = 'Sistema Inactivo';
    });
} 