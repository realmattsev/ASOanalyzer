# Flask Google Play Keyword Analyzer
# This Flask application provides a web interface for the Google Play Store Keyword Analyzer.
# It allows users to upload a file with Google Play Store URLs or paste URLs directly,
# and then displays the analysis results in a user-friendly dashboard.

from flask import Flask, request, render_template_string
import os
import subprocess
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download required NLTK data for text processing
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stopwords for text analysis
stopwords = set(stopwords.words('english'))

# Initialize Flask application
app = Flask(__name__, static_url_path='', static_folder='.')

# HTML template for the web interface
# This template includes the form for file upload/URL input and sections for displaying results
HTML_FORM = '''
<!DOCTYPE html>
<html>
<head>
    <title>MindNation Google Play Keyword Analyzer</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        /* CSS styles for the web interface */
        body {font-family: 'Poppins', sans-serif;margin: 0;padding: 0;background: linear-gradient(135deg, #006E90, #1A936F);color: #fff;}
        header {background: linear-gradient(135deg, #005F73, #0A9396);padding: 20px;text-align: center;display: flex;flex-direction: column;align-items: center;}
        .logo {max-width: 150px;margin-bottom: 10px;}
        .google-play-logo {max-width: 120px;margin-top: 10px;filter: brightness(0) invert(1);}
        main {padding: 30px;background-color: rgba(255, 255, 255, 0.95);color: #333;border-radius: 16px;margin: 40px auto;max-width: 1000px;box-shadow: 0 8px 20px rgba(0,0,0,0.3);text-align: center;}
        h1, h2 {color: #005F73;font-weight: 700;}
        pre {background-color: #E6F1F5;padding: 15px;border-radius: 12px;white-space: pre-wrap;color: #333;text-align: left;box-shadow: inset 0 4px 8px rgba(0,0,0,0.1);}
        input[type="file"] {margin-bottom: 15px;}
        input[type="submit"] {background-color: #0A9396;color: white;padding: 12px 24px;border: none;border-radius: 8px;cursor: pointer;font-weight: 600;transition: background-color 0.3s ease;}
        input[type="submit"]:hover {background-color: #007F86;}
        .btn {display: inline-block;padding: 10px 18px;margin: 8px 4px;background-color: #0A9396;color: white;text-decoration: none;border-radius: 8px;box-shadow: 0 3px 6px rgba(0,0,0,0.15);transition: background-color 0.3s ease;}
        .btn:hover {background-color: #007F86;}
        footer {text-align: center;padding: 15px;background: linear-gradient(135deg, #005F73, #0A9396);color: white;position: fixed;bottom: 0;width: 100%;}
        .file-upload {margin: 20px 0;}
        table {margin: 20px auto;width: 90%;border-collapse: collapse;}
        th, td {padding: 10px;border: 1px solid #ddd;text-align: left;}
        th {background-color: #f0f0f0;}
        .chart-caption {font-size: 0.95rem; color: #444; margin: 10px auto 30px; max-width: 700px;}
        .tabs {display: flex;justify-content: center;margin-bottom: 20px;}
        .tab {padding: 10px 20px;background-color: #E6F1F5;color: #005F73;cursor: pointer;border-radius: 8px 8px 0 0;margin: 0 5px;}
        .tab.active {background-color: #0A9396;color: white;}
        .tab-content {display: none;}
        .tab-content.active {display: block;}
        textarea {width: 100%;height: 150px;padding: 10px;border-radius: 8px;border: 1px solid #ddd;margin-bottom: 15px;font-family: inherit;}
        .input-label {display: block;text-align: left;margin-bottom: 5px;font-weight: 600;color: #005F73;}
    </style>
</head>
<body>
    <!-- Header with logos and title -->
    <header>
        <img src="https://www.mindnation.com/images/logo/mindnation-logo.png" alt="MindNation Logo" class="logo">
        <h1>📊 MindNation Google Play Keyword Analyzer</h1>
        <img src="https://play.google.com/about/images/play-logo.svg" alt="Google Play Logo" class="google-play-logo">
    </header>
    <main>
        <!-- Description of the tool -->
        <p style="text-align: left; max-width: 800px; margin: 0 auto 30px auto; font-size: 16px; line-height: 1.6;">
            <strong>The MindNation Google Play Keyword Analyzer</strong> is a web-based tool designed for marketing teams to uncover actionable insights from health and wellness app listings on the Google Play Store. By uploading a <code>.txt</code> file of Google Play Store URLs or pasting URLs directly, users can analyze each app's title, subtitle, and description to extract high-impact keywords using advanced NLP and TF-IDF techniques. The tool identifies dominant tones (e.g., clinical, wellness, lifestyle), visualizes keyword trends, and highlights opportunities for improved App Store Optimization (ASO). Results are presented through a clear, branded dashboard with downloadable reports—perfect for marketers focused on competitive positioning and organic discoverability.
        </p>
        
        <!-- Tab navigation for input methods -->
        <div class="tabs">
            <div class="tab active" onclick="openTab('file-tab')">Upload File</div>
            <div class="tab" onclick="openTab('url-tab')">Paste URLs</div>
        </div>
        
        <!-- File upload tab -->
        <div id="file-tab" class="tab-content active">
            <form method="post" enctype="multipart/form-data" class="file-upload">
                <input type="hidden" name="input_type" value="file">
                <input type="file" name="file" accept=".txt" required>
                <br><br>
                <input type="submit" value="Run Google Play Analyzer">
            </form>
        </div>
        
        <!-- URL input tab -->
        <div id="url-tab" class="tab-content">
            <form method="post" class="url-input">
                <input type="hidden" name="input_type" value="urls">
                <label class="input-label" for="urls">Paste Google Play Store URLs (one per line):</label>
                <textarea name="urls" id="urls" placeholder="https://play.google.com/store/apps/details?id=com.example.app&#10;https://play.google.com/store/apps/details?id=com.another.app" required></textarea>
                <input type="submit" value="Run Google Play Analyzer">
            </form>
        </div>

        <!-- Status messages -->
        {% if status %}
            <h2>Status:</h2>
            <pre>{{ status }}</pre>
        {% endif %}

        <!-- Summary metrics section -->
        {% if summary %}
            <h2>📈 Summary Metrics</h2>
            <table>{{ summary|safe }}</table>
        {% endif %}

        <!-- Top keywords section -->
        {% if top_keywords_table %}
            <h2>💡 Top Keyword Opportunities</h2>
            <table>{{ top_keywords_table|safe }}</table>
            <p class="chart-caption">
                This table lists the top 20 keywords based on their TF-IDF score, indicating which words are most distinctively used across analyzed apps.
                Keywords with high TF-IDF scores and broad usage (many apps) are strong candidates for organic discovery and search optimization.
            </p>
        {% endif %}

        <!-- Tone distribution chart -->
        {% if tone_chart %}
            <h2>🎨 Tone Distribution</h2>
            <img src="{{ tone_chart }}" width="500">
            <p class="chart-caption">
                This bar chart shows the tone breakdown of app messaging based on common wellness, clinical, or lifestyle language.
                It helps identify whether apps are positioning themselves as more empathetic, medical, or motivational.
            </p>
        {% endif %}

        <!-- Keyword chart -->
        {% if keyword_chart %}
            <h2>🔑 Top Keywords</h2>
            <img src="{{ keyword_chart }}" width="700">
            <p class="chart-caption">
                A visual ranking of the top 20 keywords weighted by TF-IDF. This helps highlight strategic themes that apps in this category are emphasizing.
            </p>
        {% endif %}

        <!-- Download buttons section -->
        {% if download_buttons %}
            <h2>📥 Download Reports</h2>
            {% for button in download_buttons %}
                {{ button|safe }}
            {% endfor %}
        {% endif %}
    </main>
    <footer>
        © MindNation - All Rights Reserved
    </footer>
    
    <!-- JavaScript for tab switching -->
    <script>
        function openTab(tabId) {
            // Hide all tab contents
            var tabContents = document.getElementsByClassName('tab-content');
            for (var i = 0; i < tabContents.length; i++) {
                tabContents[i].classList.remove('active');
            }
            
            // Deactivate all tabs
            var tabs = document.getElementsByClassName('tab');
            for (var i = 0; i < tabs.length; i++) {
                tabs[i].classList.remove('active');
            }
            
            // Show the selected tab content and activate the tab
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main route handler for the Flask application.
    
    This function handles both GET and POST requests:
    - GET: Displays the initial form
    - POST: Processes the uploaded file or pasted URLs and runs the analyzer
    
    Returns:
        str: Rendered HTML template with analysis results
    """
    # Initialize variables for results
    status = "Waiting for file upload..."
    download_buttons = []
    summary = ""
    tone_chart = ""
    keyword_chart = ""
    top_keywords_table = ""

    if request.method == 'POST':
        # Determine input type (file upload or direct URL pasting)
        input_type = request.form.get('input_type', 'file')
        
        if input_type == 'file':
            # Handle file upload
            uploaded_file = request.files['file']
            if uploaded_file.filename.endswith('.txt'):
                uploaded_file.save("gapp.txt")
                status = "📥 File uploaded. Running Google Play analyzer..."
        else:  # input_type == 'urls'
            # Handle direct URL pasting
            urls = request.form.get('urls', '').strip()
            if urls:
                # Save URLs to gapp.txt
                with open("gapp.txt", "w") as f:
                    f.write(urls)
                status = "📥 URLs pasted. Running Google Play analyzer..."
            else:
                status = "❌ No URLs provided. Please paste at least one URL."

        if status.startswith("📥"):
            try:
                # Run the Google Play analyzer script
                subprocess.run(["python", "google_play_analyzer.py"], check=True)

                # Debug information
                print("Analyzer completed. Checking for output directories...")
                all_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
                print(f"All directories: {all_dirs}")
                
                # Locate the latest analysis directory
                report_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and ("google_play_title_subtitle_analysis" in d or "google_play_description_analysis" in d)]
                print(f"Analysis directories found: {report_dirs}")
                
                if not report_dirs:
                    status = "❌ No analysis results found. Please check if the analyzer completed successfully."
                    return render_template_string(
                        HTML_FORM,
                        status=status,
                        download_buttons=[],
                        summary="",
                        tone_chart="",
                        keyword_chart="",
                        top_keywords_table=""
                    )
                
                # Use the title/subtitle analysis directory if available, otherwise use the description analysis
                title_dirs = [d for d in report_dirs if "google_play_title_subtitle_analysis" in d]
                if title_dirs:
                    latest_dir = max(title_dirs, key=os.path.getctime)
                else:
                    latest_dir = max(report_dirs, key=os.path.getctime)

                # Get files in the analysis directory
                files = os.listdir(latest_dir)

                # Check if required files exist
                required_files = {
                    "apps": [f for f in files if "google_play_apps" in f],
                    "keywords": [f for f in files if "google_play_keywords" in f]
                }
                
                if not required_files["apps"] or not required_files["keywords"]:
                    status = "❌ Analysis files incomplete. Please check if the analyzer completed successfully."
                    return render_template_string(
                        HTML_FORM,
                        status=status,
                        download_buttons=[],
                        summary="",
                        tone_chart="",
                        keyword_chart="",
                        top_keywords_table=""
                    )

                # Load the app data
                app_df = pd.read_csv(os.path.join(latest_dir, required_files["apps"][0]))

                # Prepare summary metrics
                summary_data = {
                    'Metric': [],
                    'Value': []
                }

                # Add total apps analyzed
                summary_data['Metric'].insert(0, 'Apps Analyzed')
                summary_data['Value'].insert(0, len(app_df))

                # Calculate keyword effectiveness metrics
                if 'Keyword Count' in app_df.columns:
                    valid_keywords = app_df['Keyword Count'][app_df['Keyword Count'] > 0]
                    summary_data['Metric'].append('Avg Keyword Count')
                    summary_data['Value'].append(round(valid_keywords.mean(), 2) if not valid_keywords.empty else 0)

                if 'Keyword Ratio' in app_df.columns:
                    valid_ratios = app_df['Keyword Ratio'][app_df['Keyword Ratio'] > 0]
                    summary_data['Metric'].append('Avg Keyword Density')
                    summary_data['Value'].append(f"{round(valid_ratios.mean() * 100, 2)}%" if not valid_ratios.empty else "0%")

                # Calculate tone distribution
                if 'Tone' in app_df.columns:
                    tone_counts = app_df['Tone'].value_counts()
                    for tone, count in tone_counts.items():
                        summary_data['Metric'].append(f'{tone} Apps')
                        summary_data['Value'].append(count)

                # Convert summary data to HTML table
                summary_df = pd.DataFrame(summary_data)
                summary = summary_df.to_html(index=False)

                # Generate tone distribution chart
                if 'Tone' in app_df.columns:
                    tone_counts = app_df['Tone'].value_counts()
                    tone_plot = tone_counts.plot(kind="bar", color="#0A9396", title="Tone Distribution").get_figure()
                    tone_chart_path = os.path.join(latest_dir, "google_play_tone_chart.png")
                    tone_plot.savefig(tone_chart_path)
                    tone_plot.clf()
                    tone_chart = tone_chart_path

                # Generate keyword analysis
                kw_df = pd.read_csv(os.path.join(latest_dir, required_files["keywords"][0]))
                if not kw_df.empty:
                    top_kw = kw_df.sort_values(by="TF-IDF Score", ascending=False).head(20)
                    top_keywords_table = top_kw[['Keyword', 'TF-IDF Score', 'Raw Count', '# Apps Used In']].to_html(index=False)

                    keyword_plot = top_kw.plot.barh(x="Keyword", y="TF-IDF Score", color="#005F73", title="Top Keywords").get_figure()
                    keyword_chart_path = os.path.join(latest_dir, "google_play_keywords_chart.png")
                    keyword_plot.savefig(keyword_chart_path)
                    keyword_plot.clf()
                    keyword_chart = keyword_chart_path

                # Create download buttons for all files
                for f in files:
                    path = os.path.join(latest_dir, f)
                    download_buttons.append(f'<a href="/{path}" class="btn" download>{f}</a>')

                # Add download button for failed URLs if any
                if os.path.exists("google_play_failed_urls.txt"):
                    download_buttons.append(f'<a href="/google_play_failed_urls.txt" class="btn" download>google_play_failed_urls.txt</a>')

                status = "✅ Analysis complete. See results below."

            except subprocess.CalledProcessError as e:
                # Handle script execution errors
                status = f"❌ Script failed: {e}"
            except Exception as ex:
                # Handle unexpected errors
                status = f"⚠️ Unexpected error: {ex}"

    # Render the template with all variables
    return render_template_string(
        HTML_FORM,
        status=status,
        download_buttons=download_buttons,
        summary=summary,
        tone_chart=tone_chart,
        keyword_chart=keyword_chart,
        top_keywords_table=top_keywords_table
    )

if __name__ == '__main__':
    # Run the Flask application in debug mode on port 5001
    app.run(debug=True, port=5001) 