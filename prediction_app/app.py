"""
Flask Application for Poverty Analysis and Prediction
Responsibility: Main web application for comprehensive poverty analysis and prediction
"""
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import os
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import tempfile
import json
from datetime import datetime
import sys

# Add parent directory to path to import from ml_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_pipeline'))

# Import our modules
from models.predictor import PovertyPredictor
from data_analyzer import DataAnalyzer

app = Flask(__name__)
app.secret_key = 'poverty_analysis_secret_key_2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize predictor and analyzer
predictor = PovertyPredictor()
analyzer = None  # Will be initialized when data is loaded

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page with file upload and model selection"""
    # Get available models
    available_models = predictor.get_available_models()
    
    # Get sample Excel structure
    sample_structure = predictor.get_sample_excel_structure()
    
    return render_template('index.html', 
                         available_models=available_models,
                         sample_structure=sample_structure)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction with enhanced analysis"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            })
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            })
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload an Excel or CSV file (.xlsx, .xls, .csv)'
            })
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'success': False,
                'error': f'File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB'
            })
        
        # Get model type from form
        model_type = request.form.get('model_type', 'neural')
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, temp_filename)
        
        file.save(filepath)
        
        # Perform comprehensive analysis
        analysis_result = perform_comprehensive_analysis(filepath)
        
        # Make predictions
        prediction_result = predictor.predict_from_excel(filepath, model_type)
        
        # Clean up temporary file
        try:
            os.remove(filepath)
        except:
            pass
        
        if prediction_result['success']:
            # Combine analysis and prediction results
            combined_results = {
                'success': True,
                'analysis': analysis_result,
                'predictions': prediction_result['predictions'].to_dict('records'),
                'summary': prediction_result['summary'],
                'total_records': len(prediction_result['predictions']),
                'model_info': {
                    'model_used': prediction_result.get('model_used', model_type),
                    'confidence_level': prediction_result.get('confidence_level', 'high'),
                    'feature_importance': prediction_result.get('feature_importance', [])
                }
            }
            
            return jsonify(combined_results)
        else:
            return jsonify({
                'success': False,
                'error': 'Analysis failed',
                'details': prediction_result['errors'],
                'analysis': analysis_result
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        })

def perform_comprehensive_analysis(filepath: str) -> dict:
    """Perform comprehensive data analysis on uploaded file"""
    try:
        # Load data
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Initialize analyzer with the data
        global analyzer
        analyzer = DataAnalyzer(filepath)
        analyzer.df = df
        
        # Run complete analysis
        analysis_results = analyzer.run_complete_analysis()
        
        # Extract key insights
        insights = extract_key_insights(analysis_results, df)
        
        return {
            'success': True,
            'data_summary': analysis_results.get('summary', {}),
            'data_quality': analysis_results.get('data_quality', {}),
            'feature_importance': analysis_results.get('feature_importance', []),
            'insights': insights,
            'statistics': {
                'total_records': len(df),
                'total_features': len(df.columns),
                'missing_data_percentage': analysis_results.get('data_quality', {}).get('completeness', {}).get('missing_percentage', 0),
                'data_quality_score': analysis_results.get('data_quality', {}).get('overall_score', 0)
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Analysis error: {str(e)}'
        }

def extract_key_insights(analysis_results: dict, df: pd.DataFrame) -> list:
    """Extract key insights from analysis results"""
    insights = []
    
    try:
        # Data quality insights
        quality = analysis_results.get('data_quality', {})
        if quality.get('overall_score', 0) < 0.8:
            insights.append({
                'type': 'warning',
                'title': 'Data Quality Issues',
                'message': f"Data quality score is {quality.get('overall_score', 0):.1%}. Consider data cleaning."
            })
        
        # Missing data insights
        missing_pct = quality.get('completeness', {}).get('missing_percentage', 0)
        if missing_pct > 5:
            insights.append({
                'type': 'info',
                'title': 'Missing Data',
                'message': f"{missing_pct:.1f}% of data is missing. This may affect prediction accuracy."
            })
        
        # Feature importance insights
        top_features = analysis_results.get('feature_importance', [])[:3]
        if top_features:
            insights.append({
                'type': 'success',
                'title': 'Key Features',
                'message': f"Most important features: {', '.join([f[0] for f in top_features])}"
            })
        
        # Poverty indicators insights
        poverty_indicators = analysis_results.get('poverty_indicators', [])
        if poverty_indicators:
            insights.append({
                'type': 'info',
                'title': 'Poverty Indicators',
                'message': f"Found {len(poverty_indicators)} poverty-related indicators in the data."
            })
        
        # Distribution insights
        numerical_analysis = analysis_results.get('numerical_analysis', {})
        if 'ingreso_laboral' in df.columns:
            income_stats = numerical_analysis.get('descriptive_stats', {}).get('ingreso_laboral', {})
            if income_stats:
                median_income = income_stats.get('50%', 0)
                insights.append({
                    'type': 'info',
                    'title': 'Income Distribution',
                    'message': f"Median labor income: ${median_income:.2f}"
                })
        
    except Exception as e:
        insights.append({
            'type': 'error',
            'title': 'Analysis Error',
            'message': f"Error extracting insights: {str(e)}"
        })
    
    return insights

@app.route('/validate', methods=['POST'])
def validate_file():
    """Validate Excel file structure without making predictions"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            })
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            })
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload an Excel or CSV file (.xlsx, .xls, .csv)'
            })
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, temp_filename)
        
        file.save(filepath)
        
        # Validate file
        is_valid, errors = predictor.validate_excel_structure(filepath)
        
        # Perform quick data analysis for validation
        quick_analysis = perform_quick_analysis(filepath) if is_valid else None
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'is_valid': is_valid,
            'errors': errors,
            'quick_analysis': quick_analysis
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Validation error: {str(e)}'
        })

def perform_quick_analysis(filepath: str) -> dict:
    """Perform quick data analysis for validation"""
    try:
        # Load data
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        return {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'data_types': df.dtypes.value_counts().to_dict(),
            'sample_data': df.head(3).to_dict('records')
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/download_template')
def download_template():
    """Download Excel template with sample data"""
    try:
        # Get sample structure
        sample_data = predictor.get_sample_excel_structure()
        
        # Create DataFrame
        df = pd.DataFrame(sample_data)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            df.to_excel(tmp.name, index=False)
            tmp_path = tmp.name
        
        # Send file
        return send_file(tmp_path, 
                        as_attachment=True,
                        download_name='poverty_analysis_template.xlsx',
                        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error creating template: {str(e)}'
        })

@app.route('/models')
def get_models():
    """Get available models information"""
    try:
        models = predictor.get_available_models()
        return jsonify({
            'success': True,
            'models': models
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error getting models: {str(e)}'
        })

@app.route('/model_info/<model_name>')
def get_model_info(model_name):
    """Get detailed information about a specific model"""
    try:
        info = predictor.get_model_info(model_name)
        return jsonify({
            'success': True,
            'model_info': info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error getting model info: {str(e)}'
        })

@app.route('/analysis')
def get_analysis():
    """Get current analysis results"""
    global analyzer
    if analyzer and hasattr(analyzer, 'analysis_results'):
        return jsonify({
            'success': True,
            'analysis': analyzer.analysis_results
        })
    else:
        return jsonify({
            'success': False,
            'error': 'No analysis available. Please upload data first.'
        })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'available_models': len(predictor.get_available_models()),
        'upload_folder': os.path.exists(UPLOAD_FOLDER)
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 