"""
ArcheryAI Pro - Report Generator
Comprehensive analysis report generation with visualizations and recommendations
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import jinja2
import webbrowser

from ..utils.logger import get_logger
from .visualizer import Visualizer

class ReportGenerator:
    """
    Comprehensive report generator for ArcheryAI Pro.
    
    This class generates detailed analysis reports including visualizations,
    performance metrics, and personalized recommendations.
    """
    
    def __init__(self):
        """Initialize the ReportGenerator."""
        self.logger = get_logger(__name__)
        self.visualizer = Visualizer()
        
        # Report templates
        self.templates_dir = Path("templates")
        self.templates_dir.mkdir(exist_ok=True)
        
        # Create default HTML template
        self._create_default_template()
        
        self.logger.info("ReportGenerator initialized successfully")
    
    def generate_report(self, analysis_results: Dict, output_dir: Path) -> Path:
        """
        Generate comprehensive analysis report.
        
        Args:
            analysis_results: Complete analysis results
            output_dir: Output directory path
            
        Returns:
            Path to generated report file
        """
        try:
            self.logger.info("Generating comprehensive analysis report...")
            
            # Create report directory
            report_dir = output_dir / "report"
            report_dir.mkdir(exist_ok=True)
            
            # Generate different report formats
            report_paths = {}
            
            # HTML report
            html_path = self._generate_html_report(analysis_results, report_dir)
            if html_path:
                report_paths['html'] = html_path
            
            # JSON report
            json_path = self._generate_json_report(analysis_results, report_dir)
            if json_path:
                report_paths['json'] = json_path
            
            # CSV report
            csv_path = self._generate_csv_report(analysis_results, report_dir)
            if csv_path:
                report_paths['csv'] = csv_path
            
            # PDF report (if possible)
            pdf_path = self._generate_pdf_report(analysis_results, report_dir)
            if pdf_path:
                report_paths['pdf'] = pdf_path
            
            # Generate visualizations
            self._generate_report_visualizations(analysis_results, report_dir)
            
            # Create report index
            index_path = self._create_report_index(report_paths, report_dir)
            
            self.logger.info(f"Report generated successfully: {index_path}")
            return index_path
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return output_dir / "report_generation_failed.txt"
    
    def _generate_html_report(self, analysis_results: Dict, report_dir: Path) -> Optional[Path]:
        """Generate HTML report."""
        try:
            # Prepare data for template
            template_data = self._prepare_template_data(analysis_results)
            
            # Load template
            template_loader = jinja2.FileSystemLoader(searchpath=str(self.templates_dir))
            template_env = jinja2.Environment(loader=template_loader)
            template = template_env.get_template('analysis_report.html')
            
            # Render template
            html_content = template.render(**template_data)
            
            # Save HTML file
            html_path = report_dir / "analysis_report.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report generated: {html_path}")
            return html_path
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {str(e)}")
            return None
    
    def _generate_json_report(self, analysis_results: Dict, report_dir: Path) -> Optional[Path]:
        """Generate JSON report."""
        try:
            # Create a clean version of the results for JSON export
            json_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'analysis_type': 'archery_form_evaluation'
                },
                'summary': {
                    'overall_score': analysis_results.get('performance_metrics', {}).get('overall_performance', 0),
                    'form_score': analysis_results.get('analysis_phases', {}).get('form_evaluation', {}).get('overall_score', 0),
                    'feedback_count': len(analysis_results.get('corrective_feedback', [])),
                    'visualization_count': len(analysis_results.get('3d_visualizations', []))
                },
                'performance_metrics': analysis_results.get('performance_metrics', {}),
                'form_evaluation': analysis_results.get('analysis_phases', {}).get('form_evaluation', {}),
                'corrective_feedback': analysis_results.get('corrective_feedback', []),
                'recommendations': self._generate_recommendations(analysis_results)
            }
            
            # Save JSON file
            json_path = report_dir / "analysis_results.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            self.logger.info(f"JSON report generated: {json_path}")
            return json_path
            
        except Exception as e:
            self.logger.error(f"Error generating JSON report: {str(e)}")
            return None
    
    def _generate_csv_report(self, analysis_results: Dict, report_dir: Path) -> Optional[Path]:
        """Generate CSV report."""
        try:
            # Extract key metrics for CSV
            csv_data = []
            
            # Performance metrics
            performance_metrics = analysis_results.get('performance_metrics', {})
            csv_data.append({
                'metric': 'overall_performance',
                'value': performance_metrics.get('overall_performance', 0),
                'category': 'performance'
            })
            csv_data.append({
                'metric': 'accuracy_score',
                'value': performance_metrics.get('accuracy_score', 0),
                'category': 'performance'
            })
            csv_data.append({
                'metric': 'consistency_score',
                'value': performance_metrics.get('consistency_score', 0),
                'category': 'performance'
            })
            csv_data.append({
                'metric': 'efficiency_score',
                'value': performance_metrics.get('efficiency_score', 0),
                'category': 'performance'
            })
            csv_data.append({
                'metric': 'stability_score',
                'value': performance_metrics.get('stability_score', 0),
                'category': 'performance'
            })
            
            # Form evaluation scores
            form_evaluation = analysis_results.get('analysis_phases', {}).get('form_evaluation', {})
            phase_scores = form_evaluation.get('phase_scores', {})
            
            for phase, score in phase_scores.items():
                csv_data.append({
                    'metric': f'{phase}_score',
                    'value': score,
                    'category': 'form_evaluation'
                })
            
            # Create DataFrame and save
            df = pd.DataFrame(csv_data)
            csv_path = report_dir / "analysis_metrics.csv"
            df.to_csv(csv_path, index=False)
            
            self.logger.info(f"CSV report generated: {csv_path}")
            return csv_path
            
        except Exception as e:
            self.logger.error(f"Error generating CSV report: {str(e)}")
            return None
    
    def _generate_pdf_report(self, analysis_results: Dict, report_dir: Path) -> Optional[Path]:
        """Generate PDF report."""
        try:
            # This would require additional PDF generation libraries
            # For now, return None as placeholder
            self.logger.warning("PDF report generation not implemented")
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating PDF report: {str(e)}")
            return None
    
    def _generate_report_visualizations(self, analysis_results: Dict, report_dir: Path):
        """Generate visualizations for the report."""
        try:
            # Generate performance chart
            performance_data = analysis_results.get('performance_metrics', {})
            if performance_data:
                self.visualizer.create_performance_chart(performance_data, report_dir)
            
            # Generate biomechanics analysis plot
            biomechanics_data = analysis_results.get('analysis_phases', {}).get('biomechanics', {})
            if biomechanics_data:
                self.visualizer.create_biomechanics_analysis_plot(biomechanics_data, report_dir)
            
            self.logger.info("Report visualizations generated")
            
        except Exception as e:
            self.logger.error(f"Error generating report visualizations: {str(e)}")
    
    def _create_report_index(self, report_paths: Dict[str, Path], report_dir: Path) -> Path:
        """Create report index file."""
        try:
            index_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ArcheryAI Pro - Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .report-links {{ margin: 20px 0; }}
        .report-link {{ display: block; margin: 10px 0; padding: 10px; background-color: #e8f4f8; border-radius: 3px; text-decoration: none; color: #333; }}
        .report-link:hover {{ background-color: #d0e8f0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏹 ArcheryAI Pro - Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="report-links">
        <h2>Available Reports:</h2>
"""
            
            for report_type, report_path in report_paths.items():
                relative_path = report_path.relative_to(report_dir)
                index_content += f"""
        <a href="{relative_path}" class="report-link">
            📄 {report_type.upper()} Report - {relative_path.name}
        </a>
"""
            
            index_content += """
    </div>
    
    <div class="footer">
        <p><em>Generated by ArcheryAI Pro - Advanced Biomechanical Analysis System</em></p>
    </div>
</body>
</html>
"""
            
            index_path = report_dir / "index.html"
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write(index_content)
            
            return index_path
            
        except Exception as e:
            self.logger.error(f"Error creating report index: {str(e)}")
            return report_dir / "index.html"
    
    def _prepare_template_data(self, analysis_results: Dict) -> Dict[str, Any]:
        """Prepare data for HTML template."""
        try:
            # Extract key data
            performance_metrics = analysis_results.get('performance_metrics', {})
            form_evaluation = analysis_results.get('analysis_phases', {}).get('form_evaluation', {})
            corrective_feedback = analysis_results.get('corrective_feedback', [])
            
            # Calculate summary statistics
            overall_score = performance_metrics.get('overall_performance', 0)
            form_score = form_evaluation.get('overall_score', 0)
            
            # Determine performance level
            if overall_score >= 0.9:
                performance_level = "Excellent"
                performance_color = "green"
            elif overall_score >= 0.8:
                performance_level = "Good"
                performance_color = "blue"
            elif overall_score >= 0.7:
                performance_level = "Average"
                performance_color = "orange"
            elif overall_score >= 0.6:
                performance_level = "Below Average"
                performance_color = "red"
            else:
                performance_level = "Poor"
                performance_color = "darkred"
            
            # Prepare phase scores
            phase_scores = form_evaluation.get('phase_scores', {})
            phase_data = []
            for phase, score in phase_scores.items():
                phase_data.append({
                    'name': phase.replace('_', ' ').title(),
                    'score': score,
                    'percentage': score * 100
                })
            
            # Prepare feedback data
            feedback_data = []
            for feedback in corrective_feedback[:10]:  # Limit to top 10
                feedback_data.append({
                    'issue': feedback.get('issue', {}).get('description', 'Unknown issue'),
                    'correction': feedback.get('correction', {}).get('description', 'No correction available'),
                    'priority': feedback.get('priority', 1)
                })
            
            return {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'overall_score': overall_score,
                'form_score': form_score,
                'performance_level': performance_level,
                'performance_color': performance_color,
                'phase_scores': phase_data,
                'corrective_feedback': feedback_data,
                'feedback_count': len(corrective_feedback),
                'visualization_count': len(analysis_results.get('3d_visualizations', [])),
                'recommendations': self._generate_recommendations(analysis_results)
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing template data: {str(e)}")
            return {}
    
    def _generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """Generate personalized recommendations."""
        try:
            recommendations = []
            
            # Performance-based recommendations
            performance_metrics = analysis_results.get('performance_metrics', {})
            
            if performance_metrics.get('accuracy_score', 0) < 0.8:
                recommendations.append("Focus on improving form consistency and reducing movement during aiming")
            
            if performance_metrics.get('consistency_score', 0) < 0.8:
                recommendations.append("Practice maintaining consistent technique across all shots")
            
            if performance_metrics.get('efficiency_score', 0) < 0.8:
                recommendations.append("Optimize movement patterns and reduce unnecessary motion")
            
            if performance_metrics.get('stability_score', 0) < 0.8:
                recommendations.append("Improve body stability and reduce sway during shot execution")
            
            # Form-based recommendations
            form_evaluation = analysis_results.get('analysis_phases', {}).get('form_evaluation', {})
            phase_scores = form_evaluation.get('phase_scores', {})
            
            if phase_scores.get('draw', 0) < 0.7:
                recommendations.append("Work on improving draw phase consistency and technique")
            
            if phase_scores.get('anchor', 0) < 0.7:
                recommendations.append("Focus on maintaining consistent anchor point and head stability")
            
            if phase_scores.get('release', 0) < 0.7:
                recommendations.append("Practice smooth release technique and proper follow-through")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Continue practicing with focus on form fundamentals")
                recommendations.append("Consider seeking professional coaching for advanced techniques")
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Continue practicing with focus on form fundamentals"]
    
    def _create_default_template(self):
        """Create default HTML template."""
        try:
            template_path = self.templates_dir / "analysis_report.html"
            
            if not template_path.exists():
                template_content = """
<!DOCTYPE html>
<html>
<head>
    <title>ArcheryAI Pro - Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #2c3e50; margin-bottom: 10px; }
        .score-section { display: flex; justify-content: space-around; margin: 30px 0; }
        .score-card { text-align: center; padding: 20px; border-radius: 8px; background-color: #ecf0f1; }
        .score-value { font-size: 2em; font-weight: bold; color: {{ performance_color }}; }
        .phase-scores { margin: 30px 0; }
        .phase-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .phase-card { padding: 15px; border-radius: 5px; background-color: #f8f9fa; border-left: 4px solid #3498db; }
        .feedback-section { margin: 30px 0; }
        .feedback-item { margin: 10px 0; padding: 15px; border-radius: 5px; background-color: #fff3cd; border-left: 4px solid #ffc107; }
        .recommendations { margin: 30px 0; }
        .recommendation-item { margin: 10px 0; padding: 10px; background-color: #d1ecf1; border-radius: 5px; }
        .footer { text-align: center; margin-top: 30px; color: #7f8c8d; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏹 ArcheryAI Pro - Analysis Report</h1>
            <p>Generated on: {{ timestamp }}</p>
        </div>
        
        <div class="score-section">
            <div class="score-card">
                <h3>Overall Performance</h3>
                <div class="score-value">{{ "%.1f"|format(overall_score * 100) }}%</div>
                <p>{{ performance_level }}</p>
            </div>
            <div class="score-card">
                <h3>Form Score</h3>
                <div class="score-value">{{ "%.1f"|format(form_score * 100) }}%</div>
            </div>
        </div>
        
        <div class="phase-scores">
            <h2>Phase-by-Phase Analysis</h2>
            <div class="phase-grid">
                {% for phase in phase_scores %}
                <div class="phase-card">
                    <h4>{{ phase.name }}</h4>
                    <div class="score-value">{{ "%.1f"|format(phase.percentage) }}%</div>
                </div>
                {% endfor %}
            </div>
        </div>
        
        <div class="feedback-section">
            <h2>Corrective Feedback ({{ feedback_count }} items)</h2>
            {% for feedback in corrective_feedback %}
            <div class="feedback-item">
                <strong>Issue:</strong> {{ feedback.issue }}<br>
                <strong>Correction:</strong> {{ feedback.correction }}
            </div>
            {% endfor %}
        </div>
        
        <div class="recommendations">
            <h2>Personalized Recommendations</h2>
            {% for recommendation in recommendations %}
            <div class="recommendation-item">
                • {{ recommendation }}
            </div>
            {% endfor %}
        </div>
        
        <div class="footer">
            <p><em>Generated by ArcheryAI Pro - Advanced Biomechanical Analysis System</em></p>
            <p>Visualizations: {{ visualization_count }} generated</p>
        </div>
    </div>
</body>
</html>
"""
                
                with open(template_path, 'w', encoding='utf-8') as f:
                    f.write(template_content)
                
                self.logger.info("Default HTML template created")
                
        except Exception as e:
            self.logger.error(f"Error creating default template: {str(e)}")
    
    def open_report(self, report_path: Path):
        """Open the generated report in a web browser."""
        try:
            if report_path.exists():
                webbrowser.open(f'file://{report_path.absolute()}')
                self.logger.info(f"Report opened in browser: {report_path}")
            else:
                self.logger.error(f"Report file not found: {report_path}")
                
        except Exception as e:
            self.logger.error(f"Error opening report: {str(e)}") 