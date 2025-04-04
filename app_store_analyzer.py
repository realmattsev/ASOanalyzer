# App Store Analyzer
# A unified tool for analyzing both Apple App Store and Google Play Store listings
# This script provides functionality to scrape and analyze app metadata from both platforms

import os
import sys
from flask import Flask, request, render_template_string
import pandas as pd
import nltk
from nltk.corpus import stopwords
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import shutil

# Download required NLTK data for text processing
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stopwords for text analysis
stopwords = set(stopwords.words('english'))

# Initialize Flask application
app = Flask(__name__, static_url_path='', static_folder='.')

# Add this function after the imports
def detect_platform(urls):
    """
    Detect the platform (App Store or Google Play) based on URL format.
    Returns 'appstore' or 'googleplay'.
    """
    # Count URLs for each platform
    app_store_count = sum(1 for url in urls if 'apps.apple.com' in url)
    google_play_count = sum(1 for url in urls if 'play.google.com' in url)
    
    # Return the platform with more URLs
    return 'appstore' if app_store_count >= google_play_count else 'googleplay'

# Add these functions after the imports
def validate_url(url):
    """
    Validate if a URL is a valid App Store or Google Play Store URL.
    Returns (is_valid, platform) tuple.
    """
    url = url.strip().lower()
    if 'apps.apple.com' in url:
        return True, 'appstore'
    elif 'play.google.com' in url:
        return True, 'googleplay'
    return False, None

def clean_url(url):
    """
    Clean and standardize a URL.
    """
    url = url.strip()
    # Remove any trailing slashes
    url = url.rstrip('/')
    # Ensure URL starts with https://
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url

def process_platform_urls(urls, platform):
    """
    Process URLs for a specific platform and return the results.
    """
    print(f"\n=== Processing {platform} URLs ===")
    print(f"Number of URLs to process: {len(urls)}")
    
    # Save URLs to appropriate file
    filename = "app.txt" if platform == "appstore" else "gapp.txt"
    print(f"Saving URLs to file: {filename}")
    
    with open(filename, "w") as f:
        f.write('\n'.join(urls))
    
    try:
        # Run the appropriate analyzer script
        script = "keywordanalyzer2.py" if platform == "appstore" else "google_play_analyzer.py"
        print(f"Running script: {script}")
        
        # Check if the script exists
        if not os.path.exists(script):
            print(f"ERROR: Script {script} not found!")
            return False, f"❌ {platform} analysis failed: Script {script} not found"
        
        # Run the script with more detailed output
        result = subprocess.run(
            ["python", script], 
            check=True,
            capture_output=True,
            text=True
        )
        
        print(f"Script output: {result.stdout}")
        if result.stderr:
            print(f"Script errors: {result.stderr}")
            
        # Check if the expected output directories were created
        if platform == "appstore":
            title_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and "title_subtitle_analysis" in d]
            desc_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and "description_analysis" in d]
            
            if not title_dirs and not desc_dirs:
                print("ERROR: App Store analysis did not create expected output directories")
                return False, f"❌ {platform} analysis failed: No output directories created"
            
            print(f"App Store analysis created directories: {title_dirs + desc_dirs}")
        else:
            google_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and "google_play_title_subtitle_analysis" in d]
            
            if not google_dirs:
                print("ERROR: Google Play analysis did not create expected output directories")
                return False, f"❌ {platform} analysis failed: No output directories created"
            
            print(f"Google Play analysis created directories: {google_dirs}")
            
        return True, f"✅ {platform} analysis completed successfully."
    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {e}")
        print(f"Error output: {e.stderr}")
        return False, f"❌ {platform} analysis failed: {e}"
    except Exception as ex:
        print(f"Unexpected error: {ex}")
        import traceback
        traceback.print_exc()
        return False, f"⚠️ {platform} analysis error: {ex}"

def create_comparison_data(appstore_data, googleplay_data):
    """
    Create comparison data between App Store and Google Play Store results.
    """
    print("\n=== Creating comparison data ===")
    
    # Extract metrics from both platforms
    appstore_metrics = {m['label']: m['value'] for m in appstore_data.get('metrics', [])}
    googleplay_metrics = {m['label']: m['value'] for m in googleplay_data.get('metrics', [])}
    
    print(f"App Store metrics: {appstore_metrics}")
    print(f"Google Play metrics: {googleplay_metrics}")
    
    # Create metrics comparison
    metrics_comparison = []
    
    # Apps Analyzed
    appstore_apps = int(appstore_metrics.get('Apps Analyzed', 0))
    googleplay_apps = int(googleplay_metrics.get('Apps Analyzed', 0))
    metrics_comparison.append({
        'Metric': 'Apps Analyzed',
        'App Store': appstore_apps,
        'Google Play': googleplay_apps,
        'Difference': googleplay_apps - appstore_apps
    })
    
    # Average Keyword Count
    appstore_keywords = float(appstore_metrics.get('Avg Keyword Count', 0))
    googleplay_keywords = float(googleplay_metrics.get('Avg Keyword Count', 0))
    metrics_comparison.append({
        'Metric': 'Avg Keyword Count',
        'App Store': appstore_keywords,
        'Google Play': googleplay_keywords,
        'Difference': round(googleplay_keywords - appstore_keywords, 2)
    })
    
    # Average Keyword Density
    appstore_density = float(appstore_metrics.get('Avg Keyword Density', '0%').rstrip('%'))
    googleplay_density = float(googleplay_metrics.get('Avg Keyword Density', '0%').rstrip('%'))
    metrics_comparison.append({
        'Metric': 'Avg Keyword Density',
        'App Store': appstore_density,  # Store as float for plotting
        'Google Play': googleplay_density,  # Store as float for plotting
        'Difference': round(googleplay_density - appstore_density, 2)
    })
    
    # Create metrics chart
    plt.figure(figsize=(12, 6))
    metrics_df = pd.DataFrame(metrics_comparison)
    
    # Create bar chart
    x = range(len(metrics_df))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], metrics_df['App Store'], width, label='App Store')
    plt.bar([i + width/2 for i in x], metrics_df['Google Play'], width, label='Google Play')
    
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('Metrics Comparison')
    plt.xticks(x, metrics_df['Metric'], rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_chart_path = f"metrics_comparison_{timestamp}.png"
    plt.savefig(metrics_chart_path)
    plt.close()
    
    # Extract keywords from both platforms
    appstore_keywords = []
    googleplay_keywords = []
    
    # Parse the HTML tables to extract keyword data
    if appstore_data.get('top_keywords_table'):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(appstore_data['top_keywords_table'], 'html.parser')
        rows = soup.find_all('tr')[1:]  # Skip header row
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 4:
                appstore_keywords.append({
                    'Keyword': cols[0].text,
                    'TF-IDF Score': float(cols[1].text),
                    'Raw Count': int(cols[2].text),
                    '# Apps Used In': int(cols[3].text)
                })
    
    if googleplay_data.get('top_keywords_table'):
        soup = BeautifulSoup(googleplay_data['top_keywords_table'], 'html.parser')
        rows = soup.find_all('tr')[1:]  # Skip header row
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 4:
                googleplay_keywords.append({
                    'Keyword': cols[0].text,
                    'TF-IDF Score': float(cols[1].text),
                    'Raw Count': int(cols[2].text),
                    '# Apps Used In': int(cols[3].text)
                })
    
    # Create keyword comparison
    keyword_comparison = []
    all_keywords = set()
    
    for kw in appstore_keywords:
        all_keywords.add(kw['Keyword'])
    
    for kw in googleplay_keywords:
        all_keywords.add(kw['Keyword'])
    
    for keyword in all_keywords:
        appstore_score = next((kw['TF-IDF Score'] for kw in appstore_keywords if kw['Keyword'] == keyword), 0)
        googleplay_score = next((kw['TF-IDF Score'] for kw in googleplay_keywords if kw['Keyword'] == keyword), 0)
        
        keyword_comparison.append({
            'Keyword': keyword,
            'App Store Score': appstore_score,
            'Google Play Score': googleplay_score,
            'Score Difference': round(googleplay_score - appstore_score, 4)
        })
    
    # Sort by absolute difference in scores
    keyword_comparison.sort(key=lambda x: abs(x['Score Difference']), reverse=True)
    keyword_comparison = keyword_comparison[:20]  # Keep top 20
    
    # Extract tone data from both platforms
    appstore_tones = {}
    googleplay_tones = {}
    
    for metric in appstore_data.get('metrics', []):
        if 'Apps' in metric['label'] and metric['label'] != 'Apps Analyzed':
            tone = metric['label'].replace(' Apps', '')
            appstore_tones[tone] = int(metric['value'])
    
    for metric in googleplay_data.get('metrics', []):
        if 'Apps' in metric['label'] and metric['label'] != 'Apps Analyzed':
            tone = metric['label'].replace(' Apps', '')
            googleplay_tones[tone] = int(metric['value'])
    
    # Create tone comparison
    tone_comparison = []
    all_tones = set(list(appstore_tones.keys()) + list(googleplay_tones.keys()))
    
    for tone in all_tones:
        appstore_count = appstore_tones.get(tone, 0)
        googleplay_count = googleplay_tones.get(tone, 0)
        
        tone_comparison.append({
            'Tone': tone,
            'App Store Count': appstore_count,
            'Google Play Count': googleplay_count,
            'Difference': googleplay_count - appstore_count
        })
    
    # Create tone chart
    plt.figure(figsize=(12, 6))
    tone_df = pd.DataFrame(tone_comparison)
    
    # Create bar chart
    x = range(len(tone_df))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], tone_df['App Store Count'], width, label='App Store')
    plt.bar([i + width/2 for i in x], tone_df['Google Play Count'], width, label='Google Play')
    
    plt.xlabel('Tone')
    plt.ylabel('Count')
    plt.title('Tone Distribution Comparison')
    plt.xticks(x, tone_df['Tone'], rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save chart
    tones_chart_path = f"tones_comparison_{timestamp}.png"
    plt.savefig(tones_chart_path)
    plt.close()
    
    # Create export directory
    export_dir = f"comparison_analysis_{timestamp}"
    os.makedirs(export_dir, exist_ok=True)
    
    # Format metrics for display (add % back to density values)
    display_metrics = []
    for metric in metrics_comparison:
        display_metric = metric.copy()
        if metric['Metric'] == 'Avg Keyword Density':
            display_metric['App Store'] = f"{metric['App Store']}%"
            display_metric['Google Play'] = f"{metric['Google Play']}%"
        display_metrics.append(display_metric)
    
    # Save comparison data to CSV files
    pd.DataFrame(display_metrics).to_csv(os.path.join(export_dir, "metrics_comparison.csv"), index=False)
    pd.DataFrame(keyword_comparison).to_csv(os.path.join(export_dir, "keywords_comparison.csv"), index=False)
    pd.DataFrame(tone_comparison).to_csv(os.path.join(export_dir, "tone_comparison.csv"), index=False)
    
    # Move charts to export directory
    shutil.move(metrics_chart_path, os.path.join(export_dir, "metrics_comparison.png"))
    shutil.move(tones_chart_path, os.path.join(export_dir, "tones_comparison.png"))
    
    # Create zip file
    zip_path = shutil.make_archive(export_dir, 'zip', export_dir)
    print(f"Comparison data exported to: {zip_path}")
    
    return {
        'metrics': display_metrics,  # Return formatted metrics for display
        'keywords': keyword_comparison,
        'tones': tone_comparison,
        'metrics_chart': os.path.join(export_dir, "metrics_comparison.png"),
        'tones_chart': os.path.join(export_dir, "tones_comparison.png")
    }

def export_comparison_data(comparison_data, timestamp):
    """
    Export comparison data to Excel and create visualizations.
    """
    print("\n=== Exporting comparison data ===")
    
    # Create export directory
    export_dir = f"comparison_analysis_{timestamp}"
    os.makedirs(export_dir, exist_ok=True)
    
    # Convert metrics data to numeric values for plotting
    metrics_df = pd.DataFrame(comparison_data['metrics'])
    
    # Create a copy for display with percentage signs
    display_metrics = metrics_df.copy()
    
    # Convert percentage strings to float values for plotting
    for idx, row in metrics_df.iterrows():
        if 'Density' in row['Metric']:
            metrics_df.at[idx, 'App Store'] = float(str(row['App Store']).rstrip('%'))
            metrics_df.at[idx, 'Google Play'] = float(str(row['Google Play']).rstrip('%'))
        else:
            metrics_df.at[idx, 'App Store'] = float(str(row['App Store']).replace(',', ''))
            metrics_df.at[idx, 'Google Play'] = float(str(row['Google Play']).replace(',', ''))
    
    # Create metrics comparison chart
    plt.figure(figsize=(12, 6))
    metrics_df.plot(x='Metric', y=['App Store', 'Google Play'], kind='bar')
    plt.title('Metrics Comparison')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save metrics chart
    metrics_chart_path = os.path.join(export_dir, "metrics_comparison.png")
    plt.savefig(metrics_chart_path)
    plt.close()
    
    # Create keyword comparison chart
    if comparison_data['keywords']:
        keywords_df = pd.DataFrame(comparison_data['keywords'])
        plt.figure(figsize=(12, 6))
        keywords_df.plot(x='Keyword', y=['App Store Score', 'Google Play Score'], kind='bar')
        plt.title('Keyword Score Comparison')
        plt.xlabel('Keyword')
        plt.ylabel('TF-IDF Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save keywords chart
        keywords_chart_path = os.path.join(export_dir, "keywords_comparison.png")
        plt.savefig(keywords_chart_path)
        plt.close()
    
    # Create tone comparison chart
    if comparison_data['tones']:
        tones_df = pd.DataFrame(comparison_data['tones'])
        plt.figure(figsize=(12, 6))
        tones_df.plot(x='Tone', y=['App Store Count', 'Google Play Count'], kind='bar')
        plt.title('Tone Distribution Comparison')
        plt.xlabel('Tone')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save tones chart
        tones_chart_path = os.path.join(export_dir, "tones_comparison.png")
        plt.savefig(tones_chart_path)
        plt.close()
    
    # Save data to Excel
    excel_path = os.path.join(export_dir, "comparison_report.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Save metrics with percentage signs for display
        display_metrics.to_excel(writer, sheet_name='Metrics Comparison', index=False)
        
        # Save keywords comparison
        if comparison_data['keywords']:
            keywords_df.to_excel(writer, sheet_name='Keywords Comparison', index=False)
        
        # Save tones comparison
        if comparison_data['tones']:
            tones_df.to_excel(writer, sheet_name='Tone Comparison', index=False)
    
    # Create zip file
    zip_path = shutil.make_archive(export_dir, 'zip', export_dir)
    print(f"Comparison data exported to: {zip_path}")
    
    return export_dir

# Update the HTML template to separate platforms
HTML_FORM = '''
<!DOCTYPE html>
<html>
<head>
    <title>App Store Analyzer</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        /* CSS styles for the web interface */
        body {font-family: 'Poppins', sans-serif;margin: 0;padding: 0;background: linear-gradient(135deg, #006E90, #1A936F);color: #fff;}
        header {background: linear-gradient(135deg, #005F73, #0A9396);padding: 20px;text-align: center;display: flex;flex-direction: column;align-items: center;}
        .logo {max-width: 150px;margin-bottom: 10px;}
        main {padding: 30px;background-color: rgba(255, 255, 255, 0.95);color: #333;border-radius: 16px;margin: 40px auto;max-width: 1000px;box-shadow: 0 8px 20px rgba(0,0,0,0.3);text-align: center;}
        h1, h2 {color: #005F73;font-weight: 700;}
        pre {background-color: #E6F1F5;padding: 15px;border-radius: 12px;white-space: pre-wrap;color: #333;text-align: left;box-shadow: inset 0 4px 8px rgba(0,0,0,0.1);}
        input[type="file"] {margin-bottom: 15px;}
        input[type="submit"] {background-color: #0A9396;color: white;padding: 12px 24px;border: none;border-radius: 8px;cursor: pointer;font-weight: 600;transition: background-color 0.3s ease;}
        input[type="submit"]:hover {background-color: #007F86;}
        .btn {display: inline-block;padding: 10px 18px;margin: 8px 4px;background-color: #0A9396;color: white;text-decoration: none;border-radius: 8px;box-shadow: 0 3px 6px rgba(0,0,0,0.15);transition: background-color 0.3s ease;}
        .btn:hover {background-color: #007F86;}
        .platform-selector {margin: 20px 0;padding: 15px;background: #E6F1F5;border-radius: 8px;}
        .platform-btn {margin: 10px;padding: 10px 20px;border: 2px solid #0A9396;border-radius: 5px;background: white;color: #0A9396;cursor: pointer;transition: all 0.3s ease;}
        .platform-btn.active {background: #0A9396;color: white;}
        .platform-btn:hover {background: #0A9396;color: white;}
        .url-input {margin: 20px 0;}
        textarea {width: 100%;height: 150px;padding: 10px;border-radius: 8px;border: 1px solid #ddd;margin-bottom: 15px;font-family: inherit;}
        .input-label {display: block;text-align: left;margin-bottom: 5px;font-weight: 600;color: #005F73;}
        .platform-indicator {margin: 10px 0;padding: 10px;background: #E6F1F5;border-radius: 8px;display: none;}
        .loading {display: none;margin: 20px 0;}
        .loading-spinner {border: 4px solid #f3f3f3;border-top: 4px solid #0A9396;border-radius: 50%;width: 40px;height: 40px;animation: spin 1s linear infinite;margin: 0 auto;}
        @keyframes spin {0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); }}
        .error-message {color: #dc3545;background: #f8d7da;padding: 10px;border-radius: 8px;margin: 10px 0;display: none;}
        .success-message {color: #28a745;background: #d4edda;padding: 10px;border-radius: 8px;margin: 10px 0;display: none;}
        .url-count {font-size: 14px;color: #666;margin-top: 5px;}
        .platform-badge {display: inline-block;padding: 5px 10px;border-radius: 15px;font-size: 14px;margin-left: 10px;}
        .platform-badge.appstore {background: #007AFF;color: white;}
        .platform-badge.googleplay {background: #00C853;color: white;}
        .platform-results {margin-top: 30px;padding: 20px;background: #fff;border-radius: 8px;box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
        .platform-results h3 {color: #005F73;margin-top: 0;}
        .platform-results.appstore {border-left: 4px solid #007AFF;}
        .platform-results.googleplay {border-left: 4px solid #00C853;}
        .analysis-status {display: flex;align-items: center;gap: 10px;margin: 10px 0;}
        .status-icon {font-size: 20px;}
        .status-text {flex-grow: 1;}
        .comparison-section {
            margin-top: 40px;
            padding: 20px;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .comparison-section h3 {
            color: #005F73;
            margin-top: 0;
            border-bottom: 2px solid #E6F1F5;
            padding-bottom: 10px;
        }
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .comparison-table th,
        .comparison-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #E6F1F5;
        }
        .comparison-table th {
            background: #F8F9FA;
            font-weight: 600;
        }
        .difference-positive {
            color: #28a745;
        }
        .difference-negative {
            color: #dc3545;
        }
        .comparison-chart {
            margin: 20px 0;
            text-align: center;
        }
        .platform-section {
            margin: 30px 0;
            padding: 20px;
            border-radius: 12px;
            background: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .appstore-section {
            border-left: 5px solid #007AFF;
        }
        
        .googleplay-section {
            border-left: 5px solid #00C853;
        }
        
        .platform-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .platform-icon {
            font-size: 24px;
        }
        
        .platform-title {
            margin: 0;
            color: #005F73;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .metric-card {
            background: #F8F9FA;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: 600;
            color: #005F73;
        }
        
        .metric-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        
        .chart-container {
            margin: 30px 0;
            text-align: center;
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .download-section {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #E6F1F5;
        }
        
        .download-title {
            font-size: 16px;
            color: #005F73;
            margin-bottom: 10px;
        }
        
        .download-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
    </style>
</head>
<body>
    <header>
        <h1>📊 App Store Analyzer</h1>
    </header>
    <main>
        <p style="text-align: left; max-width: 800px; margin: 0 auto 30px auto; font-size: 16px; line-height: 1.6;">
            <strong>The App Store Analyzer</strong> is a powerful tool for analyzing app listings from both the Apple App Store and Google Play Store. 
            Paste your app URLs below, and the analyzer will automatically detect and process URLs from both platforms.
        </p>

        <form method="post" class="url-input" id="analyzerForm">
            <input type="hidden" name="platform" id="platform" value="both">
            <label class="input-label" for="urls">Paste App Store URLs (one per line):</label>
            <textarea name="urls" id="urls" placeholder="https://apps.apple.com/app/example1&#10;https://play.google.com/store/apps/details?id=com.example2" required></textarea>
            <div class="url-count" id="urlCount"></div>
            <div id="platformIndicator" class="platform-indicator"></div>
            <div id="errorMessage" class="error-message"></div>
            <div id="successMessage" class="success-message"></div>
            <div id="loading" class="loading">
                <div class="loading-spinner"></div>
                <p>Analyzing your apps...</p>
            </div>
            <input type="submit" value="Run Analyzer" id="submitButton">
        </form>

        {% if status %}
            <h2>Status:</h2>
            <pre>{{ status }}</pre>
        {% endif %}

        {% if appstore_results %}
            <div class="platform-section appstore-section">
                <div class="platform-header">
                    <span class="platform-icon">🍎</span>
                    <h2 class="platform-title">Apple App Store Analysis</h2>
                </div>

                <div class="metrics-grid">
                    {% for metric in appstore_results.metrics %}
                    <div class="metric-card">
                        <div class="metric-value">{{ metric.value }}</div>
                        <div class="metric-label">{{ metric.label }}</div>
                    </div>
                    {% endfor %}
                </div>

                {% if appstore_results.top_keywords_table %}
                    <h3>💡 Top Keyword Opportunities</h3>
                    <table class="comparison-table">
                        {{ appstore_results.top_keywords_table|safe }}
                    </table>
                {% endif %}

                {% if appstore_results.tone_chart %}
                    <div class="chart-container">
                        <h3>🎨 Tone Distribution</h3>
                        <img src="{{ appstore_results.tone_chart }}" alt="Tone Distribution">
                    </div>
                {% endif %}

                {% if appstore_results.keyword_chart %}
                    <div class="chart-container">
                        <h3>🔑 Top Keywords</h3>
                        <img src="{{ appstore_results.keyword_chart }}" alt="Top Keywords">
                    </div>
                {% endif %}

                {% if appstore_results.download_buttons %}
                    <div class="download-section">
                        <h4 class="download-title">📥 Download Reports</h4>
                        <div class="download-buttons">
                            {% for button in appstore_results.download_buttons %}
                                {{ button|safe }}
                            {% endfor %}
                        </div>
                    </div>
                {% endif %}
            </div>
        {% endif %}

        {% if googleplay_results %}
            <div class="platform-section googleplay-section">
                <div class="platform-header">
                    <span class="platform-icon">🤖</span>
                    <h2 class="platform-title">Google Play Store Analysis</h2>
                </div>

                <div class="metrics-grid">
                    {% for metric in googleplay_results.metrics %}
                    <div class="metric-card">
                        <div class="metric-value">{{ metric.value }}</div>
                        <div class="metric-label">{{ metric.label }}</div>
                    </div>
                    {% endfor %}
                </div>

                {% if googleplay_results.top_keywords_table %}
                    <h3>💡 Top Keyword Opportunities</h3>
                    <table class="comparison-table">
                        {{ googleplay_results.top_keywords_table|safe }}
                    </table>
                {% endif %}

                {% if googleplay_results.tone_chart %}
                    <div class="chart-container">
                        <h3>🎨 Tone Distribution</h3>
                        <img src="{{ googleplay_results.tone_chart }}" alt="Tone Distribution">
                    </div>
                {% endif %}

                {% if googleplay_results.keyword_chart %}
                    <div class="chart-container">
                        <h3>🔑 Top Keywords</h3>
                        <img src="{{ googleplay_results.keyword_chart }}" alt="Top Keywords">
                    </div>
                {% endif %}

                {% if googleplay_results.download_buttons %}
                    <div class="download-section">
                        <h4 class="download-title">📥 Download Reports</h4>
                        <div class="download-buttons">
                            {% for button in googleplay_results.download_buttons %}
                                {{ button|safe }}
                            {% endfor %}
                        </div>
                    </div>
                {% endif %}
            </div>
        {% endif %}

        {% if comparison_results %}
            <div class="platform-section">
                <div class="platform-header">
                    <span class="platform-icon">📊</span>
                    <h2 class="platform-title">Platform Comparison</h2>
                </div>
                
                <h3>📈 Metrics Comparison</h3>
                <table class="comparison-table">
                    <tr>
                        <th>Metric</th>
                        <th>App Store</th>
                        <th>Google Play</th>
                        <th>Difference</th>
                    </tr>
                    {% for metric in comparison_results.metrics %}
                    <tr>
                        <td>{{ metric.Metric }}</td>
                        <td>{{ metric['App Store'] }}</td>
                        <td>{{ metric['Google Play'] }}</td>
                        <td class="{{ 'difference-positive' if metric.Difference > 0 else 'difference-negative' }}">
                            {{ metric.Difference }}
                        </td>
                    </tr>
                    {% endfor %}
                </table>
                
                <div class="chart-container">
                    <img src="{{ comparison_results.metrics_chart }}" alt="Metrics Comparison">
                </div>

                <h3>🔑 Top Keyword Comparison</h3>
                <table class="comparison-table">
                    <tr>
                        <th>Keyword</th>
                        <th>App Store Score</th>
                        <th>Google Play Score</th>
                        <th>Score Difference</th>
                    </tr>
                    {% for kw in comparison_results.keywords %}
                    <tr>
                        <td>{{ kw.Keyword }}</td>
                        <td>{{ kw['App Store Score'] }}</td>
                        <td>{{ kw['Google Play Score'] }}</td>
                        <td class="{{ 'difference-positive' if kw['Score Difference'] > 0 else 'difference-negative' }}">
                            {{ kw['Score Difference'] }}
                        </td>
                    </tr>
                    {% endfor %}
                </table>

                <h3>🎨 Tone Distribution Comparison</h3>
                <table class="comparison-table">
                    <tr>
                        <th>Tone</th>
                        <th>App Store Count</th>
                        <th>Google Play Count</th>
                        <th>Difference</th>
                    </tr>
                    {% for tone in comparison_results.tones %}
                    <tr>
                        <td>{{ tone.Tone }}</td>
                        <td>{{ tone['App Store Count'] }}</td>
                        <td>{{ tone['Google Play Count'] }}</td>
                        <td class="{{ 'difference-positive' if tone.Difference > 0 else 'difference-negative' }}">
                            {{ tone.Difference }}
                        </td>
                    </tr>
                    {% endfor %}
                </table>
                
                <div class="chart-container">
                    <img src="{{ comparison_results.tones_chart }}" alt="Tone Distribution Comparison">
                </div>

                {% if comparison_results.download_buttons %}
                    <div class="download-section">
                        <h4 class="download-title">📥 Download Comparison Reports</h4>
                        <div class="download-buttons">
                            {% for button in comparison_results.download_buttons %}
                                {{ button|safe }}
                            {% endfor %}
                        </div>
                    </div>
                {% endif %}
            </div>
        {% endif %}
    </main>

    <script>
        const textarea = document.getElementById('urls');
        const platformInput = document.getElementById('platform');
        const platformIndicator = document.getElementById('platformIndicator');
        const urlCount = document.getElementById('urlCount');
        const errorMessage = document.getElementById('errorMessage');
        const successMessage = document.getElementById('successMessage');
        const loading = document.getElementById('loading');
        const submitButton = document.getElementById('submitButton');
        const form = document.getElementById('analyzerForm');

        function updateUrlCount() {
            const urls = textarea.value.split('\\n').filter(url => url.trim());
            urlCount.textContent = `${urls.length} URL${urls.length !== 1 ? 's' : ''} detected`;
        }

        function validateUrls() {
            const urls = textarea.value.split('\\n').filter(url => url.trim());
            const validUrls = [];
            const invalidUrls = [];
            
            urls.forEach(url => {
                const [isValid, platform] = validateUrl(url);
                if (isValid) {
                    validUrls.push({ url, platform });
                } else {
                    invalidUrls.push(url);
                }
            });

            return { validUrls, invalidUrls };
        }

        function validateUrl(url) {
            url = url.trim().toLowerCase();
            if (url.includes('apps.apple.com')) {
                return [true, 'appstore'];
            } else if (url.includes('play.google.com')) {
                return [true, 'googleplay'];
            }
            return [false, null];
        }

        textarea.addEventListener('input', function() {
            updateUrlCount();
            const { validUrls, invalidUrls } = validateUrls();
            
            if (validUrls.length > 0) {
                const appStoreUrls = validUrls.filter(u => u.platform === 'appstore');
                const googlePlayUrls = validUrls.filter(u => u.platform === 'googleplay');
                
                platformIndicator.style.display = 'block';
                let platformText = '';
                
                if (appStoreUrls.length > 0) {
                    platformText += `<span class="platform-badge appstore">${appStoreUrls.length} App Store URL${appStoreUrls.length !== 1 ? 's' : ''}</span>`;
                }
                if (googlePlayUrls.length > 0) {
                    platformText += `<span class="platform-badge googleplay">${googlePlayUrls.length} Google Play URL${googlePlayUrls.length !== 1 ? 's' : ''}</span>`;
                }
                
                platformIndicator.innerHTML = platformText;
                
                errorMessage.style.display = 'none';
                successMessage.style.display = 'block';
                successMessage.textContent = `✅ ${validUrls.length} valid URL${validUrls.length !== 1 ? 's' : ''} detected`;
            } else {
                platformIndicator.style.display = 'none';
                successMessage.style.display = 'none';
            }

            if (invalidUrls.length > 0) {
                errorMessage.style.display = 'block';
                errorMessage.textContent = `⚠️ ${invalidUrls.length} invalid URL${invalidUrls.length !== 1 ? 's' : ''} detected. Please ensure URLs are from App Store or Google Play Store.`;
            } else {
                errorMessage.style.display = 'none';
            }
        });

        form.addEventListener('submit', function(e) {
            const { validUrls, invalidUrls } = validateUrls();
            
            if (validUrls.length === 0) {
                e.preventDefault();
                errorMessage.style.display = 'block';
                errorMessage.textContent = '❌ Please provide at least one valid URL.';
                return;
            }

            if (invalidUrls.length > 0) {
                if (!confirm('Some URLs are invalid. Do you want to proceed with only the valid URLs?')) {
                    e.preventDefault();
                    return;
                }
            }

            loading.style.display = 'block';
            submitButton.disabled = true;
        });
    </script>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main route handler for the Flask application.
    Handles both App Store and Google Play Store analysis.
    """
    status = "Waiting for URLs..."
    appstore_results = None
    googleplay_results = None
    comparison_results = None

    if request.method == 'POST':
        urls = request.form.get('urls', '').strip()
        
        if urls:
            # Split and clean URLs
            url_list = [clean_url(url) for url in urls.split('\n') if url.strip()]
            
            # Validate URLs
            valid_urls = []
            invalid_urls = []
            for url in url_list:
                is_valid, platform = validate_url(url)
                if is_valid:
                    valid_urls.append(url)
                else:
                    invalid_urls.append(url)
            
            if not valid_urls:
                status = "❌ No valid URLs provided. Please ensure URLs are from App Store or Google Play Store."
                return render_template_string(
                    HTML_FORM,
                    status=status,
                    appstore_results=None,
                    googleplay_results=None,
                    comparison_results=None
                )
            
            # Separate URLs by platform
            appstore_urls = [url for url in valid_urls if 'apps.apple.com' in url]
            googleplay_urls = [url for url in valid_urls if 'play.google.com' in url]
            
            status = f"📥 Processing {len(valid_urls)} valid URL{'s' if len(valid_urls) > 1 else ''}..."
            if invalid_urls:
                status += f"\n⚠️ {len(invalid_urls)} invalid URL{'s' if len(invalid_urls) > 1 else ''} were ignored."
            
            # Process App Store URLs if any
            if appstore_urls:
                success, msg = process_platform_urls(appstore_urls, 'appstore')
                status += f"\n{msg}"
                if success:
                    appstore_results = process_results('appstore')
                    if appstore_results:
                        status += "\n✅ App Store analysis completed successfully."
                    else:
                        status += "\n❌ App Store analysis failed to process results."
            
            # Process Google Play URLs if any
            if googleplay_urls:
                success, msg = process_platform_urls(googleplay_urls, 'googleplay')
                status += f"\n{msg}"
                if success:
                    googleplay_results = process_results('googleplay')
                    if googleplay_results:
                        status += "\n✅ Google Play analysis completed successfully."
                    else:
                        status += "\n❌ Google Play analysis failed to process results."
            
            # After processing both platforms, create comparison data
            if appstore_results and googleplay_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                comparison_data = create_comparison_data(appstore_results, googleplay_results)
                export_dir = export_comparison_data(comparison_data, timestamp)
                
                # Add download buttons for comparison files
                comparison_data['download_buttons'] = []
                for f in os.listdir(export_dir):
                    path = os.path.join(export_dir, f)
                    comparison_data['download_buttons'].append(
                        f'<a href="/{path}" class="btn" download>{f}</a>'
                    )
                
                comparison_results = comparison_data
                status += "\n✅ Comparison analysis completed successfully."
            
            status += "\n✅ Analysis complete. See results below."
        else:
            status = "❌ No URLs provided. Please paste at least one URL."

    return render_template_string(
        HTML_FORM,
        status=status,
        appstore_results=appstore_results,
        googleplay_results=googleplay_results,
        comparison_results=comparison_results
    )

def process_results(platform):
    """
    Process and return the analysis results for a specific platform.
    """
    try:
        print(f"\n=== Processing {platform} results ===")
        
        # Find the latest analysis directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if platform == "appstore":
            # App Store creates two directories: title_subtitle_analysis and description_analysis
            # We'll use the title_subtitle_analysis for consistency with Google Play
            dir_pattern = "title_subtitle_analysis"
            file_pattern = "apps"
        else:
            dir_pattern = "google_play_title_subtitle_analysis"
            file_pattern = "google_play"
        
        print(f"Looking for directories matching pattern: {dir_pattern}")
        report_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and dir_pattern in d]
        print(f"Found directories: {report_dirs}")
        
        if not report_dirs:
            print(f"No {platform} analysis directories found")
            return None

        latest_dir = max(report_dirs, key=os.path.getctime)
        print(f"Processing {platform} results from directory: {latest_dir}")
        files = os.listdir(latest_dir)
        print(f"Files in directory: {files}")

        # Load and process results
        app_files = [f for f in files if f"{file_pattern}" in f and f.endswith(".csv")]
        kw_files = [f for f in files if "keywords" in f and f.endswith(".csv")]
        
        print(f"App files found: {app_files}")
        print(f"Keyword files found: {kw_files}")

        if not app_files or not kw_files:
            print(f"Missing required files for {platform} analysis")
            return None

        app_df = pd.read_csv(os.path.join(latest_dir, app_files[0]))
        print(f"App DataFrame columns: {app_df.columns.tolist()}")
        print(f"App DataFrame shape: {app_df.shape}")
        
        kw_df = pd.read_csv(os.path.join(latest_dir, kw_files[0]))
        print(f"Keyword DataFrame columns: {kw_df.columns.tolist()}")
        print(f"Keyword DataFrame shape: {kw_df.shape}")

        # Prepare metrics in a more structured format
        metrics = [
            {'label': 'Apps Analyzed', 'value': len(app_df)}
        ]

        if 'Keyword Count' in app_df.columns:
            valid_keywords = app_df['Keyword Count'][app_df['Keyword Count'] > 0]
            metrics.append({
                'label': 'Avg Keyword Count',
                'value': f"{round(valid_keywords.mean(), 2) if not valid_keywords.empty else 0}"
            })

        if 'Keyword Ratio' in app_df.columns:
            valid_ratios = app_df['Keyword Ratio'][app_df['Keyword Ratio'] > 0]
            metrics.append({
                'label': 'Avg Keyword Density',
                'value': f"{round(valid_ratios.mean() * 100, 2)}%" if not valid_ratios.empty else "0%"
            })

        if 'Tone' in app_df.columns:
            tone_counts = app_df['Tone'].value_counts()
            for tone, count in tone_counts.items():
                metrics.append({
                    'label': f'{tone} Apps',
                    'value': count
                })

        # Generate tone distribution chart
        tone_chart = ""
        if 'Tone' in app_df.columns:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=tone_counts.index, y=tone_counts.values)
            plt.title(f'Tone Distribution - {platform.title()}')
            plt.xlabel('Tone')
            plt.ylabel('Number of Apps')
            tone_chart_path = os.path.join(latest_dir, f"{platform}_tone_chart.png")
            plt.savefig(tone_chart_path)
            plt.close()
            tone_chart = tone_chart_path
            print(f"Generated tone chart: {tone_chart_path}")

        # Generate keyword analysis
        top_keywords_table = ""
        keyword_chart = ""
        if not kw_df.empty:
            top_kw = kw_df.sort_values(by="TF-IDF Score", ascending=False).head(20)
            top_keywords_table = top_kw[['Keyword', 'TF-IDF Score', 'Raw Count', '# Apps Used In']].to_html(index=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(data=top_kw, x='TF-IDF Score', y='Keyword')
            plt.title(f'Top Keywords by TF-IDF Score - {platform.title()}')
            keyword_chart_path = os.path.join(latest_dir, f"{platform}_keywords_chart.png")
            plt.savefig(keyword_chart_path)
            plt.close()
            keyword_chart = keyword_chart_path
            print(f"Generated keyword chart: {keyword_chart_path}")

        # Create Excel report with multiple sheets
        excel_path = os.path.join(latest_dir, f"{platform}_analysis_report.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # App Statistics Sheet
            app_stats = pd.DataFrame({
                'Metric': [m['label'] for m in metrics],
                'Value': [m['value'] for m in metrics]
            })
            app_stats.to_excel(writer, sheet_name='App Statistics', index=False)
            
            # App Details Sheet
            app_df.to_excel(writer, sheet_name='App Details', index=False)
            
            # Keyword Analysis Sheet
            if not kw_df.empty:
                kw_df.to_excel(writer, sheet_name='Keyword Analysis', index=False)
            
            # Top Keywords Sheet
            if not kw_df.empty:
                top_kw.to_excel(writer, sheet_name='Top Keywords', index=False)
            
            # Tone Distribution Sheet
            if 'Tone' in app_df.columns:
                tone_dist = pd.DataFrame({
                    'Tone': tone_counts.index,
                    'Count': tone_counts.values
                })
                tone_dist.to_excel(writer, sheet_name='Tone Distribution', index=False)

        print(f"Generated Excel report: {excel_path}")

        # Create download buttons for all files
        download_buttons = []
        for f in files:
            path = os.path.join(latest_dir, f)
            download_buttons.append(f'<a href="/{path}" class="btn" download>{f}</a>')
        
        # Add Excel report download button
        download_buttons.append(f'<a href="/{excel_path}" class="btn" download>📊 Download Excel Report</a>')

        # Add download button for failed URLs if any
        failed_urls_file = "failed_urls.txt" if platform == "appstore" else "google_play_failed_urls.txt"
        if os.path.exists(failed_urls_file):
            download_buttons.append(f'<a href="/{failed_urls_file}" class="btn" download>{failed_urls_file}</a>')

        print(f"Successfully processed {platform} results")
        return {
            'metrics': metrics,
            'top_keywords_table': top_keywords_table,
            'tone_chart': tone_chart,
            'keyword_chart': keyword_chart,
            'download_buttons': download_buttons
        }

    except Exception as e:
        print(f"Error processing {platform} results: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    app.run(debug=True, port=5000) 