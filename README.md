# Google Play Store Keyword Analyzer

A powerful tool for analyzing Google Play Store app listings to extract keywords and metadata for App Store Optimization (ASO) purposes. This tool is particularly useful for marketing teams looking to uncover actionable insights from health and wellness app listings.

## Features

- Web-based interface for easy access
- Support for both file upload and direct URL input
- Advanced keyword extraction using NLP and TF-IDF techniques
- Tone classification (clinical, wellness, lifestyle)
- Keyword trend visualization
- Competitive analysis through similarity matrices
- Downloadable reports in CSV format

## Installation

1. Clone the repository:
```bash
git clone https://github.com/realmattsev/ASOanalyzer.git
cd ASOanalyzer
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
import nltk
nltk.download('stopwords')
```

## Usage

1. Start the Flask application:
```bash
python flask_google_play_analyzer.py
```

2. Open your web browser and navigate to `http://localhost:5001`

3. Upload a text file containing Google Play Store URLs or paste URLs directly

4. View the analysis results in the dashboard

## Project Structure

- `google_play_analyzer.py`: Core analysis functionality
- `flask_google_play_analyzer.py`: Web interface
- `requirements.txt`: Python dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 