# ArcheryAI Pro - GitHub Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.7+ (tested with 3.8-3.11)
- Git installed
- 2GB+ free disk space

### Installation
```bash
# Clone repository
git clone https://github.com/Shwet15-ibx/ArcheryAI-Pro.git
cd ArcheryAI-Pro

# Install dependencies
pip install -r requirements_production.txt

# Create necessary directories
mkdir -p logs uploads results static/uploads

# Start application
python run_production.py
```

### Access Application
- **Web Interface**: http://localhost:5000
- **API Endpoints**: http://localhost:5000/api/

### Demo Scripts Available
- `python simple_demo.py` - Basic analysis demo
- `python fast_demo.py` - Ultra-fast performance demo
- `python demo.py` - Comprehensive analysis demo

## 📁 Repository Structure

```
ArcheryAI-Pro/
├── app.py                 # Main Flask application
├── run_production.py      # Production server launcher
├── config.py             # Configuration management
├── main.py               # Core analysis engine
├── simple_demo.py        # Lightweight demo
├── fast_demo.py          # Ultra-fast demo
├── demo.py              # Comprehensive demo
├── requirements_production.txt  # Dependencies
├── README.md            # Project documentation
├── DEPLOYMENT.md        # This deployment guide
├── .gitignore          # Git ignore rules
├── templates/          # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── analyze.html
│   └── dashboard.html
├── src/                # Source code modules
│   ├── core/
│   ├── analysis/
│   └── visualization/
├── static/            # Static assets
├── logs/             # Application logs
├── uploads/          # File uploads
└── results/          # Analysis results
```

## 🔧 Configuration

### Environment Variables (Optional)
```bash
export SECRET_KEY="your-secret-key"
export DATABASE_URL="sqlite:///archeryai.db"
export UPLOAD_FOLDER="uploads"
export RESULTS_FOLDER="results"
export HOST="0.0.0.0"
export PORT="5000"
export DEBUG="False"
```

## 🎯 Usage Examples

### Web Interface
1. Navigate to http://localhost:5000
2. Upload video or use live camera
3. View real-time analysis results
4. Access dashboard for history

### Command Line
```bash
# Analyze single video
python main.py --video sample_video.mp4

# Batch analysis
python main.py --input_folder videos/ --output results/

# Performance testing
python fast_demo.py --benchmark
```

## 🚀 Production Deployment

### Using Gunicorn (Recommended)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 run_production:create_app()
```

### Docker Deployment
```bash
docker build -t archeryai-pro .
docker run -p 5000:5000 archeryai-pro
```

## 📊 Performance Metrics
- **Processing Speed**: ~0.3s per analysis
- **Memory Usage**: <256MB typical
- **Supported Formats**: MP4, AVI, MOV, WebM
- **Max File Size**: 100MB (configurable)

## 🔍 Troubleshooting

### Common Issues
1. **Port 5000 in use**: Change PORT in config.py
2. **Missing dependencies**: Install from requirements_production.txt
3. **Permission errors**: Ensure write access to logs/, uploads/, results/
4. **Memory issues**: Reduce video resolution or use fast_demo.py

### Debug Mode
```bash
export DEBUG=True
python run_production.py
```

## 📞 Support
- Check logs in `logs/archeryai_app.log`
- Review demo outputs in `demo_output/`
- Test with provided sample videos

## 📈 Version
- **Current**: v2.0.0
- **License**: MIT
- **Author**: Sports Biomechanics Research Team
