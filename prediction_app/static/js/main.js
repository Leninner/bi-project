// Main JavaScript for Labor and Economic Analysis App

// Global variables
let currentFile = null;
let currentResults = null;
let loadingModal = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Initialize Bootstrap modal
    loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
    
    // Set up event listeners
    setupEventListeners();
    
    // Update model descriptions
    updateModelDescription();
    
    // Check system health
    checkSystemHealth();
}

function setupEventListeners() {
    // File input change
    document.getElementById('fileInput').addEventListener('change', handleFileSelect);
    
    // Upload area drag and drop
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => document.getElementById('fileInput').click());
    
    // Form submission
    document.getElementById('uploadForm').addEventListener('submit', handleFormSubmit);
    
    // Model type change
    document.getElementById('modelType').addEventListener('change', updateModelDescription);
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processSelectedFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    event.currentTarget.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (isValidFileType(file)) {
            processSelectedFile(file);
        } else {
            showAlert('Por favor selecciona un archivo Excel (.xlsx o .xls)', 'danger');
        }
    }
}

function isValidFileType(file) {
    const validTypes = ['.xlsx', '.xls'];
    const fileName = file.name.toLowerCase();
    return validTypes.some(type => fileName.endsWith(type));
}

function processSelectedFile(file) {
    currentFile = file;
    
    // Update UI
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileInfo').classList.remove('d-none');
    document.getElementById('predictBtn').disabled = false;
    
    // Update upload area
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.innerHTML = `
        <div class="upload-content">
            <i class="fas fa-file-excel fa-3x text-success mb-3"></i>
            <p class="mb-2"><strong>${file.name}</strong></p>
            <p class="text-muted small">Archivo seleccionado</p>
            <p class="text-muted small">Tamaño: ${formatFileSize(file.size)}</p>
        </div>
    `;
}

function clearFile() {
    currentFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('fileInfo').classList.add('d-none');
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
        'neural': 'Red Neuronal: Análisis avanzado de patrones laborales',
        'linear': 'Modelo Lineal: Análisis rápido e interpretable',
        'logistic': 'Modelo Logístico: Clasificación de empleo, interpretable'
    };
    
    description.textContent = descriptions[modelType] || descriptions['neural'];
}

function handleFormSubmit(event) {
    event.preventDefault();
    
    if (!currentFile) {
        showAlert('Por favor selecciona un archivo primero', 'warning');
        return;
    }
    
    // Show loading modal
    showLoadingModal('Realizando análisis...', 'Procesando datos laborales y aplicando el modelo seleccionado');
    
    // Create FormData
    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('model_type', document.getElementById('modelType').value);
    
    // Send request
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoadingModal();
        
        if (data.success) {
            displayResults(data.results, data.summary);
            showAlert('Análisis completado exitosamente', 'success');
        } else {
            showAlert(`Error: ${data.error}`, 'danger');
            if (data.details) {
                console.error('Error details:', data.details);
            }
        }
    })
    .catch(error => {
        hideLoadingModal();
        showAlert(`Error de conexión: ${error.message}`, 'danger');
        console.error('Error:', error);
    });
}

function validateFile() {
    if (!currentFile) {
        showAlert('Por favor selecciona un archivo primero', 'warning');
        return;
    }
    
    showLoadingModal('Validando archivo...', 'Verificando estructura de datos laborales del Excel');
    
    const formData = new FormData();
    formData.append('file', currentFile);
    
    fetch('/validate', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoadingModal();
        
        if (data.success) {
            displayValidationResults(data.is_valid, data.errors);
        } else {
            showAlert(`Error de validación: ${data.error}`, 'danger');
        }
    })
    .catch(error => {
        hideLoadingModal();
        showAlert(`Error de conexión: ${error.message}`, 'danger');
        console.error('Error:', error);
    });
}

function displayValidationResults(isValid, errors) {
    const validationSection = document.getElementById('validationSection');
    const validationResults = document.getElementById('validationResults');
    
    let html = '';
    
    if (isValid) {
        html = `
            <div class="alert alert-success">
                <i class="fas fa-check-circle me-2"></i>
                <strong>Archivo válido</strong>
                <p class="mb-0 mt-2">El archivo cumple con todos los requisitos y está listo para la predicción.</p>
            </div>
        `;
    } else {
        html = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <strong>Archivo inválido</strong>
                <ul class="mb-0 mt-2">
                    ${errors.map(error => `<li>${error}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    validationResults.innerHTML = html;
    validationSection.classList.remove('d-none');
}

function displayResults(results, summary) {
    currentResults = results;
    
    // Display summary cards
    displaySummaryCards(summary);
    
    // Display results table
    displayResultsTable(results);
    
    // Show results section
    document.getElementById('resultsSection').classList.remove('d-none');
    
    // Scroll to results
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
}

function displaySummaryCards(summary) {
    const summaryCards = document.getElementById('summaryCards');
    
    const cards = [
        {
            title: 'Total Registros',
            value: summary.total_registros,
            icon: 'fas fa-users',
            class: 'info',
            subtitle: 'Personas analizadas'
        },
        {
            title: 'En Pobreza',
            value: summary.pobre_count,
            percentage: summary.pobre_percentage,
            icon: 'fas fa-exclamation-triangle',
            class: 'warning',
            subtitle: 'Personas en riesgo'
        },
        {
            title: 'Fuera de Pobreza',
            value: summary.no_pobre_count,
            percentage: summary.no_pobre_percentage,
            icon: 'fas fa-check-circle',
            class: 'success',
            subtitle: 'Personas estables'
        },
        {
            title: 'Subempleo',
            value: summary.subemployment_count,
            percentage: summary.subemployment_percentage,
            icon: 'fas fa-clock',
            class: 'danger',
            subtitle: 'Trabajo insuficiente'
        },
        {
            title: 'Alto Riesgo',
            value: summary.high_risk_count,
            percentage: summary.high_risk_percentage,
            icon: 'fas fa-exclamation-circle',
            class: 'danger',
            subtitle: 'Múltiples factores'
        },
        {
            title: 'Confianza Promedio',
            value: (summary.avg_confidence * 100).toFixed(1) + '%',
            icon: 'fas fa-chart-line',
            class: 'info',
            subtitle: 'Precisión del modelo'
        },
        {
            title: 'Ingreso Promedio Pobre',
            value: '$' + summary.avg_income_pobre,
            icon: 'fas fa-dollar-sign',
            class: 'warning',
            subtitle: 'Personas en pobreza'
        },
        {
            title: 'Ingreso Promedio No Pobre',
            value: '$' + summary.avg_income_no_pobre,
            icon: 'fas fa-dollar-sign',
            class: 'success',
            subtitle: 'Personas estables'
        }
    ];
    
    const html = cards.map(card => `
        <div class="col-lg-3 col-md-4 col-sm-6 mb-3">
            <div class="summary-card ${card.class}">
                <i class="${card.icon} fa-2x mb-3 text-${card.class}"></i>
                <div class="number">${card.value}</div>
                <div class="label">${card.title}</div>
                ${card.percentage ? `<div class="small text-muted">${card.percentage}%</div>` : ''}
                <div class="small text-muted">${card.subtitle}</div>
            </div>
        </div>
    `).join('');
    
    summaryCards.innerHTML = html;
    
    // Add detailed analysis section
    addDetailedAnalysis(summary);
}

function addDetailedAnalysis(summary) {
    const resultsSection = document.getElementById('resultsSection');
    const cardBody = resultsSection.querySelector('.card-body');
    
    // Check if detailed analysis already exists
    let detailedSection = cardBody.querySelector('#detailedAnalysis');
    if (detailedSection) {
        detailedSection.remove();
    }
    
    detailedSection = document.createElement('div');
    detailedSection.id = 'detailedAnalysis';
    detailedSection.className = 'mt-4';
    
    detailedSection.innerHTML = `
        <h6 class="mb-3"><i class="fas fa-chart-pie me-2"></i>Análisis Detallado</h6>
        
        <div class="row">
            <!-- Demographic Analysis -->
            <div class="col-md-6 mb-3">
                <div class="card border-info">
                    <div class="card-header bg-info text-white">
                        <h6 class="mb-0"><i class="fas fa-users me-2"></i>Análisis Demográfico</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-6">
                                <h6>Género</h6>
                                ${Object.entries(summary.gender_distribution || {}).map(([key, value]) => 
                                    `<div class="small">${key}: ${value}</div>`
                                ).join('')}
                            </div>
                            <div class="col-6">
                                <h6>Grupos de Edad</h6>
                                ${Object.entries(summary.age_groups || {}).map(([key, value]) => 
                                    `<div class="small">${key}: ${value}</div>`
                                ).join('')}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Employment Analysis -->
            <div class="col-md-6 mb-3">
                <div class="card border-success">
                    <div class="card-header bg-success text-white">
                        <h6 class="mb-0"><i class="fas fa-briefcase me-2"></i>Análisis Laboral</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-6">
                                <h6>Sectores Principales</h6>
                                ${Object.entries(summary.sector_distribution || {}).slice(0, 3).map(([key, value]) => 
                                    `<div class="small">${key}: ${value}</div>`
                                ).join('')}
                            </div>
                            <div class="col-6">
                                <h6>Condiciones de Actividad</h6>
                                ${Object.entries(summary.activity_distribution || {}).slice(0, 3).map(([key, value]) => 
                                    `<div class="small">${key}: ${value}</div>`
                                ).join('')}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Income Analysis -->
            <div class="col-md-6 mb-3">
                <div class="card border-warning">
                    <div class="card-header bg-warning text-dark">
                        <h6 class="mb-0"><i class="fas fa-dollar-sign me-2"></i>Análisis de Ingresos</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-6">
                                <h6>Distribución de Ingresos</h6>
                                ${Object.entries(summary.income_distribution || {}).map(([key, value]) => 
                                    `<div class="small">${key}: ${value}</div>`
                                ).join('')}
                            </div>
                            <div class="col-6">
                                <h6>Ingreso Per Cápita</h6>
                                ${Object.entries(summary.per_capita_distribution || {}).map(([key, value]) => 
                                    `<div class="small">${key}: ${value}</div>`
                                ).join('')}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Risk Analysis -->
            <div class="col-md-6 mb-3">
                <div class="card border-danger">
                    <div class="card-header bg-danger text-white">
                        <h6 class="mb-0"><i class="fas fa-exclamation-triangle me-2"></i>Análisis de Riesgo</h6>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-6">
                                <h6>Niveles de Riesgo</h6>
                                ${Object.entries(summary.risk_distribution || {}).map(([key, value]) => 
                                    `<div class="small">${key}: ${value}</div>`
                                ).join('')}
                            </div>
                            <div class="col-6">
                                <h6>Pobreza por Educación</h6>
                                ${Object.entries(summary.pobre_by_education || {}).slice(0, 3).map(([key, value]) => 
                                    `<div class="small">${key}: ${value}</div>`
                                ).join('')}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    cardBody.appendChild(detailedSection);
}

function displayResultsTable(results) {
    const tbody = document.getElementById('resultsTableBody');
    
    const html = results.map(row => `
        <tr>
            <td><strong>${row.persona_key}</strong></td>
            <td>
                <span class="badge ${row.prediccion_pobreza === 1 ? 'bg-warning' : 'bg-success'}">
                    ${row.estado_pobreza}
                </span>
            </td>
            <td>${row.genero || 'N/A'}</td>
            <td>${row.edad || 'N/A'}</td>
            <td>
                <span class="badge bg-info">${row.nivel_educativo || 'N/A'}</span>
            </td>
            <td>${row.sector_economico || 'N/A'}</td>
            <td>
                <span class="badge ${getActivityBadgeColor(row.condicion_actividad)}">
                    ${row.condicion_actividad || 'N/A'}
                </span>
            </td>
            <td>
                <span class="badge ${getIncomeBadgeColor(row.categoria_ingreso)}">
                    $${row.ingreso_laboral || 0} (${row.categoria_ingreso || 'N/A'})
                </span>
            </td>
            <td>
                ${row.indicador_subempleo === 1 ? 
                    '<span class="badge bg-danger">Sí</span>' : 
                    '<span class="badge bg-success">No</span>'
                }
            </td>
            <td>
                <span class="badge ${getRiskBadgeColor(row.nivel_riesgo)}">
                    ${row.nivel_riesgo || 'N/A'}
                </span>
            </td>
            <td>${(row.probabilidad_pobreza * 100).toFixed(1)}%</td>
            <td>
                <div class="progress" style="height: 20px;">
                    <div class="progress-bar ${getConfidenceColor(row.confianza)}" 
                         style="width: ${(row.confianza * 100)}%">
                        ${(row.confianza * 100).toFixed(0)}%
                    </div>
                </div>
            </td>
            <td>
                <span class="badge bg-secondary">${row.modelo_usado}</span>
            </td>
        </tr>
    `).join('');
    
    tbody.innerHTML = html;
}

function getActivityBadgeColor(activity) {
    const colors = {
        'Ocupado': 'bg-success',
        'Desempleado': 'bg-danger',
        'Inactivo': 'bg-secondary',
        'Jubilado': 'bg-info',
        'Estudiante': 'bg-primary',
        'Ama de casa': 'bg-warning',
        'Incapacitado': 'bg-dark',
        'Otro inactivo': 'bg-secondary',
        'Trabajador familiar': 'bg-info',
        'Sin información': 'bg-light text-dark'
    };
    return colors[activity] || 'bg-secondary';
}

function getIncomeBadgeColor(incomeCategory) {
    const colors = {
        'Sin ingreso': 'bg-danger',
        'Bajo': 'bg-warning',
        'Medio-bajo': 'bg-info',
        'Medio': 'bg-primary',
        'Alto': 'bg-success'
    };
    return colors[incomeCategory] || 'bg-secondary';
}

function getRiskBadgeColor(riskLevel) {
    const colors = {
        'Sin riesgo': 'bg-success',
        'Bajo': 'bg-info',
        'Medio': 'bg-warning',
        'Alto': 'bg-danger',
        'Muy alto': 'bg-dark'
    };
    return colors[riskLevel] || 'bg-secondary';
}

function getConfidenceColor(confidence) {
    if (confidence >= 0.8) return 'bg-success';
    if (confidence >= 0.6) return 'bg-warning';
    return 'bg-danger';
}

function downloadTemplate() {
    showLoadingModal('Generando plantilla...', 'Creando archivo Excel con estructura de ejemplo');
    
    fetch('/download_template')
    .then(response => {
        if (response.ok) {
            return response.blob();
        }
        throw new Error('Error al generar plantilla');
    })
    .then(blob => {
        hideLoadingModal();
        
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'labor_analysis_template.xlsx';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showAlert('Plantilla descargada exitosamente', 'success');
    })
    .catch(error => {
        hideLoadingModal();
        showAlert(`Error al descargar plantilla: ${error.message}`, 'danger');
    });
}

function exportToExcel() {
    if (!currentResults) {
        showAlert('No hay resultados para exportar', 'warning');
        return;
    }
    
    // Create workbook
    const wb = XLSX.utils.book_new();
    const ws = XLSX.utils.json_to_sheet(currentResults);
    
    // Add worksheet to workbook
    XLSX.utils.book_append_sheet(wb, ws, 'Análisis Laboral');
    
    // Generate filename
    const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
    const filename = `labor_analysis_${timestamp}.xlsx`;
    
    // Save file
    XLSX.writeFile(wb, filename);
    
    showAlert('Resultados exportados a Excel exitosamente', 'success');
}

function exportToCSV() {
    if (!currentResults) {
        showAlert('No hay resultados para exportar', 'warning');
        return;
    }
    
    // Convert to CSV
    const headers = Object.keys(currentResults[0]);
    const csvContent = [
        headers.join(','),
        ...currentResults.map(row => headers.map(header => `"${row[header]}"`).join(','))
    ].join('\n');
    
    // Create download link
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `labor_analysis_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    
    showAlert('Resultados exportados a CSV exitosamente', 'success');
}

function showLoadingModal(title, message) {
    document.getElementById('loadingTitle').textContent = title;
    document.getElementById('loadingMessage').textContent = message;
    loadingModal.show();
}

function hideLoadingModal() {
    loadingModal.hide();
}

function showAlert(message, type) {
    // Create alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Add to page
    document.body.appendChild(alertDiv);
    
    // Auto remove after 5 seconds
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