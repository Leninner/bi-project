"""
Data Analyzer Module
Responsibility: Analyze and understand the poverty dataset structure and characteristics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from feature_engineer import FeatureEngineer

# Suppress numpy warnings for division by zero in correlation calculations
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')


class DataAnalyzer:
    """Analyzes poverty dataset to understand structure and identify key features"""
    
    def __init__(self, data_path: str):
        """
        Initialize DataAnalyzer with dataset path
        
        Args:
            data_path: Path to the CSV dataset
        """
        self.data_path = data_path
        self.df = None
        self.analysis_results = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load the dataset from CSV file"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def get_basic_info(self) -> Dict:
        """Get basic information about the dataset"""
        if self.df is None:
            return {}
            
        info = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'data_types': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'unique_values_per_column': {col: self.df[col].nunique() for col in self.df.columns}
        }
        
        self.analysis_results['basic_info'] = info
        return info
    
    def identify_feature_types(self) -> Dict[str, List[str]]:
        """Identify numerical and categorical features"""
        if self.df is None:
            return {}
            
        numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()
        
        feature_types = {
            'numerical': numerical_features,
            'categorical': categorical_features
        }
        
        self.analysis_results['feature_types'] = feature_types
        return feature_types
    
    def analyze_numerical_features(self) -> Dict:
        """Analyze numerical features with statistical measures"""
        if self.df is None:
            return {}
            
        numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Handle correlation calculation with proper error handling
        try:
            # Use pandas corr with min_periods to handle missing data
            correlation_matrix = self.df[numerical_features].corr(min_periods=1)
            # Replace any remaining NaN values with 0
            correlation_matrix = correlation_matrix.fillna(0)
        except Exception as e:
            print(f"Warning: Could not calculate correlation matrix: {e}")
            correlation_matrix = pd.DataFrame(0, index=numerical_features, columns=numerical_features)
        
        analysis = {
            'descriptive_stats': self.df[numerical_features].describe().to_dict(),
            'correlation_matrix': correlation_matrix.to_dict(),
            'skewness': self.df[numerical_features].skew().fillna(0).to_dict(),
            'kurtosis': self.df[numerical_features].kurtosis().fillna(0).to_dict(),
            'outliers': self._detect_outliers(numerical_features),
            'distribution_analysis': self._analyze_distributions(numerical_features)
        }
        
        self.analysis_results['numerical_analysis'] = analysis
        return analysis
    
    def _detect_outliers(self, numerical_features: List[str]) -> Dict:
        """Detect outliers using IQR method"""
        outliers = {}
        for feature in numerical_features:
            Q1 = self.df[feature].quantile(0.25)
            Q3 = self.df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)).sum()
            outliers[feature] = {
                'count': int(outlier_count),
                'percentage': float(outlier_count / len(self.df) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
        return outliers
    
    def _analyze_distributions(self, numerical_features: List[str]) -> Dict:
        """Analyze distribution characteristics of numerical features"""
        distributions = {}
        for feature in numerical_features:
            data = self.df[feature].dropna()
            distributions[feature] = {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis()),
                'is_normal': abs(data.skew()) < 1 and abs(data.kurtosis()) < 1
            }
        return distributions
    
    def analyze_categorical_features(self) -> Dict:
        """Analyze categorical features"""
        if self.df is None:
            return {}
            
        categorical_features = self.df.select_dtypes(include=['object']).columns.tolist()
        
        analysis = {}
        for feature in categorical_features:
            analysis[feature] = {
                'unique_values': self.df[feature].nunique(),
                'value_counts': self.df[feature].value_counts().to_dict(),
                'missing_count': self.df[feature].isnull().sum(),
                'most_common': self.df[feature].mode().iloc[0] if not self.df[feature].mode().empty else None
            }
        
        self.analysis_results['categorical_analysis'] = analysis
        return analysis
    
    def identify_poverty_indicators(self) -> List[str]:
        """Identify potential poverty indicator columns with enhanced detection"""
        if self.df is None:
            return []
            
        # Enhanced poverty indicators with more comprehensive coverage
        income_keywords = ['ingreso', 'salario', 'renta', 'per_capita', 'income', 'salary', 'wage', 'remuneracion']
        
        # Employment/work indicators that correlate with poverty
        employment_keywords = ['horas', 'trabajo', 'work', 'empleo', 'employment', 'desea_trabajar', 
                             'disponible_trabajar', 'ocupacion', 'laboral']
        
        # Education and social indicators
        social_keywords = ['nivel_instruccion', 'education', 'instruccion', 'escolaridad', 'estudios']
        
        # Economic sector and activity indicators
        sector_keywords = ['sector', 'actividad', 'condact', 'economic']
        
        # Combined keywords for comprehensive poverty detection
        poverty_keywords = income_keywords + employment_keywords + social_keywords + sector_keywords
        
        potential_indicators = []
        
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in poverty_keywords):
                potential_indicators.append(col)
        
        # Add specific known indicators from your dataset
        specific_indicators = ['ingreso_laboral', 'ingreso_per_capita', 'horas_trabajo_semana', 
                             'desea_trabajar_mas', 'disponible_trabajar_mas', 'nivel_instruccion',
                             'sector_id', 'condact_id']
        
        # Ensure all specific indicators are included if they exist in the dataset
        for indicator in specific_indicators:
            if indicator in self.df.columns and indicator not in potential_indicators:
                potential_indicators.append(indicator)
        
        self.analysis_results['poverty_indicators'] = potential_indicators
        return potential_indicators
    
    def generate_feature_importance_ranking(self) -> List[Tuple[str, float]]:
        """Generate feature importance ranking based on correlation with all poverty indicators"""
        if self.df is None:
            return []
            
        poverty_indicators = self.identify_poverty_indicators()
        if not poverty_indicators:
            return []
        
        numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove poverty indicators from features to avoid self-correlation
        features = [f for f in numerical_features if f not in poverty_indicators]
        
        # Calculate average correlation with all poverty indicators
        feature_importance = {}
        
        for feature in features:
            total_correlation = 0.0
            valid_correlations = 0
            
            for indicator in poverty_indicators:
                try:
                    # Use pandas corr with min_periods to handle missing data
                    corr = abs(self.df[feature].corr(self.df[indicator], min_periods=1))
                    # Handle NaN values
                    if not pd.isna(corr):
                        total_correlation += corr
                        valid_correlations += 1
                except Exception as e:
                    print(f"Warning: Could not calculate correlation for {feature} with {indicator}: {e}")
            
            # Calculate average correlation across all indicators
            if valid_correlations > 0:
                avg_correlation = total_correlation / valid_correlations
            else:
                avg_correlation = 0.0
                
            feature_importance[feature] = avg_correlation
        
        # Convert to list of tuples and sort by importance
        correlations = [(feature, importance) for feature, importance in feature_importance.items()]
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Store detailed results for analysis
        self.analysis_results['feature_importance'] = {
            'rankings': correlations,
            'poverty_indicators_used': poverty_indicators,
            'total_features_analyzed': len(features)
        }
        
        return correlations
    
    def generate_feature_importance_for_indicator(self, target_indicator: str) -> List[Tuple[str, float]]:
        """Generate feature importance for a specific poverty indicator"""
        if self.df is None or target_indicator not in self.df.columns:
            return []
        
        numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        features = [f for f in numerical_features if f != target_indicator]
        
        correlations = []
        for feature in features:
            try:
                corr = abs(self.df[feature].corr(self.df[target_indicator], min_periods=1))
                if not pd.isna(corr):
                    correlations.append((feature, corr))
            except Exception as e:
                print(f"Warning: Could not calculate correlation for {feature} with {target_indicator}: {e}")
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        return correlations
    
    def compare_feature_importance_across_indicators(self) -> Dict[str, List[Tuple[str, float]]]:
        """Compare feature importance across different poverty indicators"""
        if self.df is None:
            return {}
        
        poverty_indicators = self.identify_poverty_indicators()
        comparisons = {}
        
        for indicator in poverty_indicators:
            if indicator in self.df.columns:
                comparisons[indicator] = self.generate_feature_importance_for_indicator(indicator)
        
        self.analysis_results['cross_indicator_importance'] = comparisons
        return comparisons
    
    def analyze_data_quality(self) -> Dict:
        """Analyze overall data quality"""
        if self.df is None:
            return {}
        
        quality_metrics = {
            'completeness': {},
            'consistency': {},
            'accuracy': {},
            'overall_score': 0.0
        }
        
        # Completeness analysis
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isnull().sum().sum()
        completeness_score = (total_cells - missing_cells) / total_cells
        quality_metrics['completeness'] = {
            'score': completeness_score,
            'missing_cells': missing_cells,
            'total_cells': total_cells,
            'missing_percentage': missing_cells / total_cells * 100
        }
        
        # Consistency analysis (check for logical inconsistencies)
        consistency_issues = []
        
        # Check for logical inconsistencies
        if 'edad' in self.df.columns and 'nivel_instruccion' in self.df.columns:
            # Children with high education levels
            young_high_edu = ((self.df['edad'] < 18) & (self.df['nivel_instruccion'] > 3)).sum()
            if young_high_edu > 0:
                consistency_issues.append(f"{young_high_edu} registros con edad < 18 y educación alta")
        
        if 'horas_trabajo_semana' in self.df.columns and 'ingreso_laboral' in self.df.columns:
            # People working many hours with zero income
            work_no_income = ((self.df['horas_trabajo_semana'] > 20) & (self.df['ingreso_laboral'] == 0)).sum()
            if work_no_income > 0:
                consistency_issues.append(f"{work_no_income} registros trabajando >20h con ingreso 0")
        
        consistency_score = 1.0 - (len(consistency_issues) / len(self.df) * 0.1)  # Penalty for inconsistencies
        quality_metrics['consistency'] = {
            'score': consistency_score,
            'issues': consistency_issues,
            'issue_count': len(consistency_issues)
        }
        
        # Accuracy analysis (check for extreme outliers)
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_percentage = 0
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < Q1 - 3 * IQR) | (self.df[col] > Q3 + 3 * IQR)).sum()
            outlier_percentage += outliers / len(self.df)
        
        outlier_percentage /= len(numerical_cols)
        accuracy_score = 1.0 - outlier_percentage
        quality_metrics['accuracy'] = {
            'score': accuracy_score,
            'outlier_percentage': outlier_percentage * 100
        }
        
        # Overall quality score
        quality_metrics['overall_score'] = (
            completeness_score * 0.4 + 
            consistency_score * 0.3 + 
            accuracy_score * 0.3
        )
        
        self.analysis_results['data_quality'] = quality_metrics
        return quality_metrics
    
    def run_complete_analysis(self) -> Dict:
        """Run complete data analysis pipeline"""
        print("Starting comprehensive data analysis...")
        
        # Load data if not already loaded
        if self.df is None:
            self.load_data()
        
        if self.df is None:
            return {}
        
        # Run all analysis components
        analysis_results = {
            'basic_info': self.get_basic_info(),
            'feature_types': self.identify_feature_types(),
            'numerical_analysis': self.analyze_numerical_features(),
            'categorical_analysis': self.analyze_categorical_features(),
            'poverty_indicators': self.identify_poverty_indicators(),
            'feature_importance': self.generate_feature_importance_ranking(),
            'cross_indicator_importance': self.compare_feature_importance_across_indicators(),
            'data_quality': self.analyze_data_quality()
        }
        
        # Generate summary statistics
        summary = self._generate_analysis_summary(analysis_results)
        analysis_results['summary'] = summary
        
        self.analysis_results = analysis_results
        
        print("Data analysis completed successfully!")
        return analysis_results
    
    def _generate_analysis_summary(self, results: Dict) -> Dict:
        """Generate summary of analysis results"""
        summary = {
            'dataset_size': f"{results['basic_info']['shape'][0]:,} rows, {results['basic_info']['shape'][1]} columns",
            'data_quality_score': f"{results['data_quality']['overall_score']:.2%}",
            'poverty_indicators_found': len(results['poverty_indicators']),
            'top_features': [feature for feature, _ in results['feature_importance'][:5]],
            'missing_data_percentage': f"{results['data_quality']['completeness']['missing_percentage']:.1f}%",
            'data_issues': len(results['data_quality']['consistency']['issues'])
        }
        return summary
    
    def analyze_informal_sector_by_age(self, age_min: int = 15, age_max: int = 24, 
                                     sector_id: int = 2, ciudad_id: int = 10150) -> Dict:
        """
        Analyze informal sector employment by age group and city
        
        Args:
            age_min: Minimum age (default: 15)
            age_max: Maximum age (default: 24)
            sector_id: Sector ID for informal (default: 2)
            ciudad_id: City ID for Ambato (default: 10150)
            
        Returns:
            Dictionary with analysis results
        """
        if self.df is None:
            return {}
        
        # Filter data for specific age range, sector, and city
        mask = (
            (self.df['edad'] >= age_min) & 
            (self.df['edad'] <= age_max) & 
            (self.df['sector_id'] == sector_id) & 
            (self.df['ciudad_id'] == ciudad_id)
        )
        
        filtered_data = self.df[mask]
        
        analysis = {
            'total_persons': len(filtered_data),
            'age_range': f"{age_min}-{age_max}",
            'sector_id': sector_id,
            'ciudad_id': ciudad_id,
            'percentage_of_total': (len(filtered_data) / len(self.df)) * 100 if len(self.df) > 0 else 0,
            'demographics': {
                'age_distribution': filtered_data['edad'].value_counts().to_dict(),
                'gender_distribution': filtered_data['sexo'].value_counts().to_dict(),
                'education_distribution': filtered_data['nivel_instruccion'].value_counts().to_dict()
            },
            'employment_characteristics': {
                'avg_income': filtered_data['ingreso_laboral'].mean() if 'ingreso_laboral' in filtered_data.columns else 0,
                'avg_income_per_capita': filtered_data['ingreso_per_capita'].mean() if 'ingreso_per_capita' in filtered_data.columns else 0,
                'avg_work_hours': filtered_data['horas_trabajo_semana'].mean() if 'horas_trabajo_semana' in filtered_data.columns else 0,
                'want_more_work': filtered_data['desea_trabajar_mas'].value_counts().to_dict() if 'desea_trabajar_mas' in filtered_data.columns else {},
                'available_more_work': filtered_data['disponible_trabajar_mas'].value_counts().to_dict() if 'disponible_trabajar_mas' in filtered_data.columns else {}
            }
        }
        
        self.analysis_results['informal_sector_analysis'] = analysis
        return analysis
    
    def analyze_underemployment_by_income(self, ciudad_id: int = 10150) -> Dict:
        """
        Analyze underemployment due to insufficient income in a specific city
        
        Args:
            ciudad_id: City ID for Ambato (default: 10150)
            
        Returns:
            Dictionary with analysis results
        """
        if self.df is None:
            return {}
        
        # Filter data for specific city
        city_data = self.df[self.df['ciudad_id'] == ciudad_id]
        
        # Define underemployment conditions (condact_id: 2, 3, 1)
        underemployment_conditions = [2, 3, 1]  # Subempleado, Ocupado parcial o desempleado
        
        # Filter for underemployment
        underemployed = city_data[city_data['condact_id'].isin(underemployment_conditions)]
        
        # Calculate income thresholds for underemployment
        if 'ingreso_per_capita' in city_data.columns:
            income_threshold = city_data['ingreso_per_capita'].median()
            low_income_threshold = city_data['ingreso_per_capita'].quantile(0.25)
        else:
            income_threshold = city_data['ingreso_laboral'].median()
            low_income_threshold = city_data['ingreso_laboral'].quantile(0.25)
        
        # Identify underemployment due to insufficient income
        income_underemployed = underemployed[
            (underemployed['ingreso_per_capita'] < income_threshold) if 'ingreso_per_capita' in underemployed.columns 
            else (underemployed['ingreso_laboral'] < income_threshold)
        ]
        
        analysis = {
            'total_city_population': len(city_data),
            'total_underemployed': len(underemployed),
            'underemployed_due_to_income': len(income_underemployed),
            'percentage_underemployed': (len(underemployed) / len(city_data)) * 100 if len(city_data) > 0 else 0,
            'percentage_income_underemployed': (len(income_underemployed) / len(city_data)) * 100 if len(city_data) > 0 else 0,
            'income_thresholds': {
                'median_income': income_threshold,
                'low_income_threshold': low_income_threshold
            },
            'underemployment_by_condition': {
                'subempleado': len(underemployed[underemployed['condact_id'] == 2]),
                'ocupado_parcial': len(underemployed[underemployed['condact_id'] == 3])
            },
            'demographics': {
                'age_distribution': income_underemployed['edad'].value_counts().to_dict(),
                'gender_distribution': income_underemployed['sexo'].value_counts().to_dict(),
                'education_distribution': income_underemployed['nivel_instruccion'].value_counts().to_dict(),
                'sector_distribution': income_underemployed['sector_id'].value_counts().to_dict()
            },
            'income_characteristics': {
                'avg_income_underemployed': income_underemployed['ingreso_laboral'].mean() if 'ingreso_laboral' in income_underemployed.columns else 0,
                'avg_income_per_capita_underemployed': income_underemployed['ingreso_per_capita'].mean() if 'ingreso_per_capita' in income_underemployed.columns else 0,
                'avg_work_hours_underemployed': income_underemployed['horas_trabajo_semana'].mean() if 'horas_trabajo_semana' in income_underemployed.columns else 0
            }
        }
        
        self.analysis_results['underemployment_analysis'] = analysis
        return analysis
    
    def predict_informal_sector_growth(self, age_min: int = 15, age_max: int = 24, 
                                     sector_id: int = 2, ciudad_id: int = 10150) -> Dict:
        """
        Predict informal sector growth for specific demographic
        
        Args:
            age_min: Minimum age
            age_max: Maximum age
            sector_id: Sector ID for informal
            ciudad_id: City ID
            
        Returns:
            Dictionary with prediction results
        """
        if self.df is None:
            return {}
        
        # Get current analysis
        current_analysis = self.analyze_informal_sector_by_age(age_min, age_max, sector_id, ciudad_id)
        
        # Simple trend analysis based on historical data
        if 'anio' in self.df.columns and 'mes' in self.df.columns:
            # Group by year and month to see trends
            time_series = self.df.groupby(['anio', 'mes']).size().reset_index(name='count')
            
            # Calculate growth rate (simple linear trend)
            if len(time_series) > 1:
                growth_rate = (time_series['count'].iloc[-1] - time_series['count'].iloc[0]) / len(time_series)
            else:
                growth_rate = 0
        else:
            growth_rate = 0
        
        prediction = {
            'current_count': current_analysis.get('total_persons', 0),
            'growth_rate': growth_rate,
            'predicted_next_period': current_analysis.get('total_persons', 0) + growth_rate,
            'confidence_level': 'medium',  # Could be enhanced with statistical models
            'factors_affecting_growth': [
                'Economic conditions',
                'Employment policies',
                'Educational attainment',
                'Migration patterns'
            ]
        }
        
        self.analysis_results['informal_sector_prediction'] = prediction
        return prediction
    
    def predict_underemployment_trends(self, ciudad_id: int = 10150) -> Dict:
        """
        Predict underemployment trends for a specific city
        
        Args:
            ciudad_id: City ID
            
        Returns:
            Dictionary with prediction results
        """
        if self.df is None:
            return {}
        
        # Get current analysis
        current_analysis = self.analyze_underemployment_by_income(ciudad_id)
        
        # Analyze trends over time if available
        if 'anio' in self.df.columns and 'mes' in self.df.columns:
            city_data = self.df[self.df['ciudad_id'] == ciudad_id]
            underemployment_conditions = [2, 3]
            
            # Time series analysis
            time_series = city_data[city_data['condact_id'].isin(underemployment_conditions)].groupby(['anio', 'mes']).size().reset_index(name='underemployed_count')
            
            if len(time_series) > 1:
                trend = (time_series['underemployed_count'].iloc[-1] - time_series['underemployed_count'].iloc[0]) / len(time_series)
            else:
                trend = 0
        else:
            trend = 0
        
        prediction = {
            'current_underemployed': current_analysis.get('underemployed_due_to_income', 0),
            'trend_direction': 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable',
            'trend_magnitude': abs(trend),
            'predicted_next_period': current_analysis.get('underemployed_due_to_income', 0) + trend,
            'risk_factors': [
                'Low educational attainment',
                'Limited job opportunities',
                'Economic instability',
                'Seasonal employment patterns'
            ],
            'recommendations': [
                'Improve educational access',
                'Create formal employment opportunities',
                'Implement income support programs',
                'Develop skills training programs'
            ]
        }
        
        self.analysis_results['underemployment_prediction'] = prediction
        return prediction
    
    def analyze_poverty_targets(self, poverty_target_income: pd.Series, poverty_target_multi: pd.Series) -> Dict:
        """
        Analyze poverty target distributions and characteristics
        
        Args:
            poverty_target_income: Poverty target created using income threshold method
            poverty_target_multi: Poverty target created using multi-factor method
            
        Returns:
            Dictionary with poverty target analysis results
        """
        if self.df is None:
            return {}
        
        analysis = {
            'income_threshold': {
                'total_in_poverty': int(poverty_target_income.sum()),
                'total_population': len(poverty_target_income),
                'poverty_rate': (poverty_target_income.sum() / len(poverty_target_income)) * 100 if len(poverty_target_income) > 0 else 0,
                'distribution': poverty_target_income.value_counts().to_dict()
            },
            'multi_factor': {
                'total_in_poverty': int(poverty_target_multi.sum()),
                'total_population': len(poverty_target_multi),
                'poverty_rate': (poverty_target_multi.sum() / len(poverty_target_multi)) * 100 if len(poverty_target_multi) > 0 else 0,
                'distribution': poverty_target_multi.value_counts().to_dict()
            }
        }
        
        # Analyze characteristics of people in poverty for both methods
        for method_name, target_series in [('income_threshold', poverty_target_income), ('multi_factor', poverty_target_multi)]:
            in_poverty_mask = target_series == 1
            in_poverty_data = self.df[in_poverty_mask]
            
            if len(in_poverty_data) > 0:
                analysis[method_name]['characteristics'] = {
                    'demographics': {
                        'age_distribution': in_poverty_data['edad'].value_counts().to_dict(),
                        'gender_distribution': in_poverty_data['sexo'].value_counts().to_dict(),
                        'education_distribution': in_poverty_data['nivel_instruccion'].value_counts().to_dict(),
                        'sector_distribution': in_poverty_data['sector_id'].value_counts().to_dict()
                    },
                    'economic_indicators': {
                        'avg_income': in_poverty_data['ingreso_laboral'].mean() if 'ingreso_laboral' in in_poverty_data.columns else 0,
                        'avg_income_per_capita': in_poverty_data['ingreso_per_capita'].mean() if 'ingreso_per_capita' in in_poverty_data.columns else 0,
                        'avg_work_hours': in_poverty_data['horas_trabajo_semana'].mean() if 'horas_trabajo_semana' in in_poverty_data.columns else 0,
                        'median_income': in_poverty_data['ingreso_laboral'].median() if 'ingreso_laboral' in in_poverty_data.columns else 0,
                        'median_income_per_capita': in_poverty_data['ingreso_per_capita'].median() if 'ingreso_per_capita' in in_poverty_data.columns else 0
                    },
                    'employment_status': {
                        'want_more_work': in_poverty_data['desea_trabajar_mas'].value_counts().to_dict() if 'desea_trabajar_mas' in in_poverty_data.columns else {},
                        'available_more_work': in_poverty_data['disponible_trabajar_mas'].value_counts().to_dict() if 'disponible_trabajar_mas' in in_poverty_data.columns else {},
                        'employment_condition': in_poverty_data['condact_id'].value_counts().to_dict() if 'condact_id' in in_poverty_data.columns else {}
                    }
                }
            else:
                analysis[method_name]['characteristics'] = {}
        
        # Compare the two methods
        analysis['comparison'] = {
            'difference_in_poverty_rate': abs(analysis['income_threshold']['poverty_rate'] - analysis['multi_factor']['poverty_rate']),
            'correlation_between_methods': poverty_target_income.corr(poverty_target_multi) if len(poverty_target_income) > 0 else 0,
            'agreement_rate': ((poverty_target_income == poverty_target_multi).sum() / len(poverty_target_income)) * 100 if len(poverty_target_income) > 0 else 0
        }
        
        self.analysis_results['poverty_target_analysis'] = analysis
        return analysis
    
    def get_poverty_targets(self, method: str = 'both') -> Dict[str, pd.Series]:
        """
        Get poverty target variables for machine learning
        
        Args:
            method: 'income_threshold', 'multi_factor', or 'both'
            
        Returns:
            Dictionary with poverty target series
        """
        if self.df is None:
            return {}
        
        feature_engineer = FeatureEngineer()
        targets = {}
        
        if method in ['income_threshold', 'both']:
            targets['income_threshold'] = feature_engineer.create_poverty_target(self.df, method='income_threshold')
        
        if method in ['multi_factor', 'both']:
            targets['multi_factor'] = feature_engineer.create_poverty_target(self.df, method='multi_factor')
        
        return targets
    
    def save_analysis_report(self, output_path: str = "analysis_report.txt"):
        """Save analysis results to a text file"""
        with open(output_path, 'w') as f:
            f.write("=== POVERTY DATASET ANALYSIS REPORT ===\n\n")
            
            for key, value in self.analysis_results.items():
                f.write(f"{key.upper()}:\n")
                f.write(f"{value}\n\n")
        
        print(f"Analysis report saved to {output_path}")


if __name__ == "__main__":
    # Test the analyzer with poverty target integration
    analyzer = DataAnalyzer("../data/poverty_dataset.csv")
    results = analyzer.run_complete_analysis()
    
    # Test getting poverty targets separately
    print("\n=== TESTING POVERTY TARGETS ===")
    poverty_targets = analyzer.get_poverty_targets(method='both')
    print(f"Available poverty targets: {list(poverty_targets.keys())}")
    
    if 'income_threshold' in poverty_targets:
        print(f"Income threshold target shape: {poverty_targets['income_threshold'].shape}")
        print(f"Income threshold target distribution: {poverty_targets['income_threshold'].value_counts().to_dict()}")
    
    if 'multi_factor' in poverty_targets:
        print(f"Multi-factor target shape: {poverty_targets['multi_factor'].shape}")
        print(f"Multi-factor target distribution: {poverty_targets['multi_factor'].value_counts().to_dict()}")
    
    analyzer.save_analysis_report() 