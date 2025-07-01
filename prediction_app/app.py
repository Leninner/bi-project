"""
Flask Application for Labor and Economic Analysis
Responsibility: Main web application for employment and economic analysis
"""
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import os
import pandas as pd
from werkzeug.utils import secure_filename
import tempfile
import json
from datetime import datetime

# Import our modules
from models.predictor import PovertyPredictor

app = Flask(__name__)
app.secret_key = 'labor_analysis_secret_key_2024'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize predictor
predictor = PovertyPredictor()

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
    """Handle file upload and prediction"""
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
                'error': 'Invalid file type. Please upload an Excel file (.xlsx or .xls)'
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
        
        # Make predictions
        result = predictor.predict_from_excel(filepath, model_type)
        
        # Clean up temporary file
        try:
            os.remove(filepath)
        except:
            pass
        
        if result['success']:
            # Save results to session or temporary file for display
            results_data = result['predictions'].to_dict('records')
            
            return jsonify({
                'success': True,
                'results': results_data,
                'summary': result['summary'],
                'total_records': len(results_data)
            })
        else:
                    return jsonify({
            'success': False,
            'error': 'Analysis failed',
            'details': result['errors']
        })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        })

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
                'error': 'Invalid file type. Please upload an Excel file (.xlsx or .xls)'
            })
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, temp_filename)
        
        file.save(filepath)
        
        # Validate file
        is_valid, errors = predictor.validate_excel_structure(filepath)
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'is_valid': is_valid,
            'errors': errors
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Validation error: {str(e)}'
        })

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
                        download_name='labor_analysis_template.xlsx',
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

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'available_models': len(predictor.get_available_models())
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
    print("=" * 60)
    print("POVERTY PREDICTION WEB APPLICATION")
    print("=" * 60)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Available models: {len(predictor.get_available_models())}")
    print("=" * 60)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000) 