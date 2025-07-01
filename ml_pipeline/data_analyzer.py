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
            'memory_usage': self.df.memory_usage(deep=True).sum()
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
            'kurtosis': self.df[numerical_features].kurtosis().fillna(0).to_dict()
        }
        
        self.analysis_results['numerical_analysis'] = analysis
        return analysis
    
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
                'missing_count': self.df[feature].isnull().sum()
            }
        
        self.analysis_results['categorical_analysis'] = analysis
        return analysis
    
    def identify_poverty_indicators(self) -> List[str]:
        """Identify potential poverty indicator columns"""
        if self.df is None:
            return []
            
        # Direct poverty indicators (income-related)
        income_keywords = ['ingreso', 'salario', 'renta', 'per_capita', 'income', 'salary', 'wage']
        
        # Employment/work indicators that correlate with poverty
        employment_keywords = ['horas', 'trabajo', 'work', 'empleo', 'employment', 'desea_trabajar', 'disponible_trabajar']
        
        # Education and social indicators
        social_keywords = ['nivel_instruccion', 'education', 'instruccion', 'escolaridad']
        
        # Combined keywords for comprehensive poverty detection
        poverty_keywords = income_keywords + employment_keywords + social_keywords
        
        potential_indicators = []
        
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in poverty_keywords):
                potential_indicators.append(col)
        
        # Add specific known indicators from your dataset
        specific_indicators = ['ingreso_laboral', 'ingreso_per_capita', 'horas_trabajo_semana', 
                             'desea_trabajar_mas', 'disponible_trabajar_mas', 'nivel_instruccion']
        
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
            'ranking': correlations,
            'poverty_indicators_used': poverty_indicators,
            'total_features_analyzed': len(features),
            'correlation_details': feature_importance
        }
        
        return correlations
    
    def generate_feature_importance_for_indicator(self, target_indicator: str) -> List[Tuple[str, float]]:
        """
        Generate feature importance ranking for a specific poverty indicator
        
        Args:
            target_indicator: Specific poverty indicator to analyze
            
        Returns:
            List of tuples (feature, importance) sorted by importance
        """
        if self.df is None or target_indicator not in self.df.columns:
            return []
        
        numerical_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target from features
        features = [f for f in numerical_features if f != target_indicator]
        
        # Calculate correlations with proper error handling
        correlations = []
        for feature in features:
            try:
                # Use pandas corr with min_periods to handle missing data
                corr = abs(self.df[feature].corr(self.df[target_indicator], min_periods=1))
                # Handle NaN values
                if pd.isna(corr):
                    corr = 0.0
                correlations.append((feature, corr))
            except Exception as e:
                print(f"Warning: Could not calculate correlation for {feature} with {target_indicator}: {e}")
                correlations.append((feature, 0.0))
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        return correlations
    
    def compare_feature_importance_across_indicators(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        Compare feature importance across all poverty indicators
        
        Returns:
            Dictionary with indicator name as key and feature importance ranking as value
        """
        if self.df is None:
            return {}
        
        poverty_indicators = self.identify_poverty_indicators()
        if not poverty_indicators:
            return {}
        
        comparison_results = {}
        
        for indicator in poverty_indicators:
            importance_ranking = self.generate_feature_importance_for_indicator(indicator)
            comparison_results[indicator] = importance_ranking
        
        # Store comparison results
        self.analysis_results['feature_importance_comparison'] = comparison_results
        
        return comparison_results
    
    def run_complete_analysis(self) -> Dict:
        """Run complete analysis of the dataset"""
        print("=== COMPLETE DATASET ANALYSIS ===")
        
        # Load data
        self.load_data()
        
        # Run all analyses
        basic_info = self.get_basic_info()
        feature_types = self.identify_feature_types()
        numerical_analysis = self.analyze_numerical_features()
        categorical_analysis = self.analyze_categorical_features()
        poverty_indicators = self.identify_poverty_indicators()
        feature_importance = self.generate_feature_importance_ranking()
        
        # Compare feature importance across all indicators
        feature_importance_comparison = self.compare_feature_importance_across_indicators()
        
        # Create poverty targets using FeatureEngineer
        print("\n=== POVERTY TARGET CREATION ===")
        feature_engineer = FeatureEngineer()
        
        # Create poverty targets using both methods
        poverty_target_income = feature_engineer.create_poverty_target(self.df, method='income_threshold')
        poverty_target_multi = feature_engineer.create_poverty_target(self.df, method='multi_factor')
        
        # Analyze poverty target distributions
        poverty_target_analysis = self.analyze_poverty_targets(poverty_target_income, poverty_target_multi)
        
        # Run new specific analyses
        print("\n=== SPECIFIC ANALYSES ===")
        
        # Analysis 1: Informal sector by age (15-24) in Ambato
        print("Analyzing informal sector employment (15-24 years) in Ambato...")
        informal_analysis = self.analyze_informal_sector_by_age(15, 24, 2, 10150)
        
        # Analysis 2: Underemployment due to insufficient income in Ambato
        print("Analyzing underemployment due to insufficient income in Ambato...")
        underemployment_analysis = self.analyze_underemployment_by_income(10150)
        
        # Predictions
        print("Generating predictions...")
        informal_prediction = self.predict_informal_sector_growth(15, 24, 2, 10150)
        underemployment_prediction = self.predict_underemployment_trends(10150)
        
        # Print summary
        print(f"\nDataset Shape: {basic_info.get('shape', 'N/A')}")
        print(f"Numerical Features: {len(feature_types.get('numerical', []))}")
        print(f"Categorical Features: {len(feature_types.get('categorical', []))}")
        print(f"Poverty Indicators: {poverty_indicators}")
        print(f"\nTop 5 Most Important Features:")
        if isinstance(feature_importance, list):
            # Handle old format (backward compatibility)
            for feature, importance in feature_importance[:5]:
                print(f"  {feature}: {importance:.4f}")
        else:
            # Handle new format
            ranking = feature_importance.get('ranking', [])
            for feature, importance in ranking[:5]:
                print(f"  {feature}: {importance:.4f}")
            print(f"  (Based on {len(feature_importance.get('poverty_indicators_used', []))} poverty indicators)")
        
        # Print feature importance comparison summary
        print(f"\n=== FEATURE IMPORTANCE COMPARISON ===")
        print(f"Analyzed {len(feature_importance_comparison)} poverty indicators:")
        for indicator, ranking in feature_importance_comparison.items():
            if ranking:
                top_feature, top_importance = ranking[0]
                print(f"  {indicator}: Top feature '{top_feature}' (importance: {top_importance:.4f})")
        
        # Print poverty target analysis
        print(f"\n=== POVERTY TARGET ANALYSIS ===")
        print(f"Income Threshold Method:")
        print(f"  Poverty rate: {poverty_target_analysis['income_threshold']['poverty_rate']:.2f}%")
        print(f"  Total in poverty: {poverty_target_analysis['income_threshold']['total_in_poverty']}")
        print(f"Multi-Factor Method:")
        print(f"  Poverty rate: {poverty_target_analysis['multi_factor']['poverty_rate']:.2f}%")
        print(f"  Total in poverty: {poverty_target_analysis['multi_factor']['total_in_poverty']}")
        
        # Print specific analysis results
        print(f"\n=== INFORMAL SECTOR ANALYSIS (15-24 years, Ambato) ===")
        print(f"Total persons in informal sector: {informal_analysis.get('total_persons', 0)}")
        print(f"Percentage of total population: {informal_analysis.get('percentage_of_total', 0):.2f}%")
        print(f"Average income: ${informal_analysis.get('employment_characteristics', {}).get('avg_income', 0):.2f}")
        print(f"Average work hours: {informal_analysis.get('employment_characteristics', {}).get('avg_work_hours', 0):.1f} hours/week")
        
        print(f"\n=== UNDEREMPLOYMENT ANALYSIS (Ambato) ===")
        print(f"Total underemployed due to insufficient income: {underemployment_analysis.get('underemployed_due_to_income', 0)}")
        print(f"Percentage of city population: {underemployment_analysis.get('percentage_income_underemployed', 0):.2f}%")
        print(f"Subempleado: {underemployment_analysis.get('underemployment_by_condition', {}).get('subempleado', 0)}")
        print(f"Ocupado parcial: {underemployment_analysis.get('underemployment_by_condition', {}).get('ocupado_parcial', 0)}")
        
        print(f"\n=== PREDICTIONS ===")
        print(f"Informal sector growth rate: {informal_prediction.get('growth_rate', 0):.2f}")
        print(f"Predicted informal sector next period: {informal_prediction.get('predicted_next_period', 0):.0f}")
        print(f"Underemployment trend: {underemployment_prediction.get('trend_direction', 'stable')}")
        print(f"Predicted underemployed next period: {underemployment_prediction.get('predicted_next_period', 0):.0f}")
        
        return self.analysis_results
    
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