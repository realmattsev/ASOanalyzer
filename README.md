# App Store & Google Play Keyword Analyzer

A comprehensive tool for analyzing app listings from both the Apple App Store and Google Play Store. This tool helps you understand keyword usage, tone distribution, and provides detailed metrics for app optimization.

## Features

### Core Analysis
- **Dual Platform Support**: Analyze apps from both Apple App Store and Google Play Store
- **Keyword Analysis**: Extract and analyze keywords from app titles, subtitles, and descriptions
- **Tone Classification**: Automatically classify app content into different tones (Professional, Friendly, etc.)
- **TF-IDF Scoring**: Calculate keyword importance using TF-IDF algorithm

### Enhanced Reporting
- **Excel Reports**: Generate comprehensive Excel reports for each platform with multiple sheets:
  - App Statistics
  - App Details
  - Keyword Analysis
  - Top Keywords
  - Tone Distribution
- **Comparison Analysis**: Compare metrics between App Store and Google Play Store apps
- **Visual Analytics**: Generate charts and visualizations for:
  - Keyword distribution
  - Tone analysis
  - Metrics comparison
  - Score differences

### Export Options
- **Multiple Formats**: Export data in CSV, Excel, and PNG formats
- **Zipped Reports**: Download complete analysis packages
- **Platform-Specific Reports**: Separate reports for App Store and Google Play Store
- **Comparison Reports**: Side-by-side analysis of both platforms

## Installation

1. Clone the repository:
```bash
git clone https://github.com/realmattsev/ASOanalyzer.git
cd ASOanalyzer
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"
```

## Usage

1. Start the Flask application:
```bash
python app_store_analyzer.py
```

2. Access the web interface at `http://localhost:5000`

3. Choose your analysis platform:
   - App Store
   - Google Play Store
   - Comparison Analysis

4. Input app URLs:
   - Paste URLs directly
   - Upload a text file with URLs
   - Use the comparison feature for both platforms

5. View and download results:
   - Excel reports with multiple sheets
   - CSV files for raw data
   - PNG charts for visualizations
   - Comparison analysis package

## File Structure

- `app_store_analyzer.py`: Main Flask application
- `keywordanalyzer2.py`: App Store analysis script
- `google_play_analyzer.py`: Google Play Store analysis script
- `requirements.txt`: Python dependencies
- `templates/`: HTML templates for the web interface
- `static/`: CSS and JavaScript files

## Dependencies

- Flask
- Pandas
- NLTK
- BeautifulSoup4
- Requests
- Matplotlib
- Seaborn
- Scikit-learn
- SpaCy
- Openpyxl

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 