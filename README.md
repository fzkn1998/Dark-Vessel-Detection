# 🌊 Dark Vessel Detection

A user-friendly web interface for detecting dark vessels using satellite imagery. This application helps identify vessels that may be intentionally hiding their location by turning off their AIS transponders.

> **Inspired by**: [xView3 Challenge](https://www.xview3.org/) winning solution by [BloodAxe](https://github.com/BloodAxe)

## ✨ Features

- **Interactive Web Interface**: Easy-to-use dashboard for analyzing satellite imagery
- **Real-time Detection**: Quick identification of potential dark vessels
- **Visualization Tools**: View and analyze detection results with various filters
- **Export Results**: Save detection results in multiple formats
- **Responsive Design**: Works on both desktop and tablet devices

## 🖥️ Project Structure

```
dark_vessel_detection/
├── app.py              # Main Flask application
├── static/             # Static files (JS, CSS, images)
│   ├── css/           # Stylesheets
│   │   └── style.css  # Main stylesheet
│   ├── js/            # Frontend JavaScript
│   │   └── app.js     # Main JavaScript file
│   ├── Vessel.jpg     # Sample vessel image
│   └── logo.png       # Application logo
├── templates/          # HTML templates
│   ├── base.html      # Base template
│   ├── index.html     # Main dashboard
│   └── scene.html     # Scene visualization page
├── 0d30f9dfc2891b6bp.csv  # Dataset file
├── public.csv         # Public dataset
├── new.mp4            # Demo video
├── requirements.txt   # Python dependencies
└── README.md
```

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/fzkn1998/dark-vessel-detection.git
   cd dark-vessel-detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Install in development mode:
   ```bash
   pip install -e .
   ```
   
### Web Application
Start the Flask web application:
```bash
python app.py
```
Then open `http://localhost:5000` in your browser.

## 🧠 Model Architecture

The solution uses a custom CircleNet architecture with the following key components:

1. **Encoder-Decoder Network**: Based on U-Net architecture with pre-trained encoders
2. **Multi-task Learning**:
   - Object detection (vessel/non-vessel)
   - Length estimation
   - Fishing activity classification
3. **Ensemble Learning**: Combines predictions from multiple models for improved accuracy

## 📊 Results

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| B4_UNet | 0.89 | 0.85 | 0.87 |
| B5_UNet | 0.90 | 0.86 | 0.88 |
| V2S_UNet | 0.91 | 0.85 | 0.88 |
| Ensemble | 0.92 | 0.87 | 0.89 |

## 📚 References

- [xView3 Challenge](https://www.xview3.org/)
- [xView3 First Place Solution](https://github.com/BloodAxe/xView3-The-First-Place-Solution)
- [CenterNet: Objects as Points](https://arxiv.org/abs/1904.07850)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The xView3 Challenge organizers
- BloodAxe for the winning solution
- Open-source community for various tools and libraries
See the notebooks/ directory for data exploration and processing examples.

## References
- [xView3 Challenge](https://www.xview3.org/)
- [xView3 First Place Solution](https://github.com/BloodAxe/xView3-The-First-Place-Solution)





