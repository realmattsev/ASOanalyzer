from flask import Flask, request, render_template_string
import os
import subprocess
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stopwords
stopwords = set(stopwords.words('english'))

app = Flask(__name__, static_url_path='', static_folder='.')

HTML_FORM = '''
<!DOCTYPE html>
<html>
<head>
    <title>MindNation Keyword Analyzer</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {font-family: 'Poppins', sans-serif;margin: 0;padding: 0;background: linear-gradient(135deg, #006E90, #1A936F);color: #fff;}
        header {background: linear-gradient(135deg, #005F73, #0A9396);padding: 20px;text-align: center;}
        main {padding: 30px;background-color: rgba(255, 255, 255, 0.95);color: #333;border-radius: 16px;margin: 40px auto;max-width: 1000px;box-shadow: 0 8px 20px rgba(0,0,0,0.3);text-align: center;}
        h1, h2 {color: #005F73;font-weight: 700;}
        pre {background-color: #E6F1F5;padding: 15px;border-radius: 12px;white-space: pre-wrap;color: #333;text-align: left;box-shadow: inset 0 4px 8px rgba(0,0,0,0.1);}
        input[type="file"] {margin-bottom: 15px;}
        input[type="submit"] {background-color: #0A9396;color: white;padding: 12px 24px;border: none;border-radius: 8px;cursor: pointer;font-weight: 600;transition: background-color 0.3s ease;}
        input[type="submit"]:hover {background-color: #007F86;}
        .btn {display: inline-block;padding: 10px 18px;margin: 8px 4px;background-color: #0A9396;color: white;text-decoration: none;border-radius: 8px;box-shadow: 0 3px 6px rgba(0,0,0,0.15);transition: background-color 0.3s ease;}
        .btn:hover {background-color: #007F86;}
        footer {text-align: center;padding: 15px;background: linear-gradient(135deg, #005F73, #0A9396);color: white;position: fixed;bottom: 0;width: 100%;}
        .logo {max-width: 150px;margin-bottom: 20px;}
        .file-upload {margin: 20px 0;}
        table {margin: 20px auto;width: 90%;border-collapse: collapse;}
        th, td {padding: 10px;border: 1px solid #ddd;text-align: left;}
        th {background-color: #f0f0f0;}
        .chart-caption {font-size: 0.95rem; color: #444; margin: 10px auto 30px; max-width: 700px;}
    </style>
</head>
<body>
    <header>
        <img src="https://www.mindnation.com/images/logo/mindnation-logo.png" alt="MindNation Logo" class="logo">
        <h1>📊 MindNation Keyword Analyzer</h1>
    </header>
    <main>
        <p style="text-align: left; max-width: 800px; margin: 0 auto 30px auto; font-size: 16px; line-height: 1.6;">
            <strong>The MindNation Keyword Analyzer</strong> is a web-based tool designed for marketing teams to uncover actionable insights from health and wellness app listings on the Apple App Store. By uploading a <code>.txt</code> file of App Store URLs, users can analyze each app's title, subtitle, and description to extract high-impact keywords using advanced NLP and TF-IDF techniques. The tool identifies dominant tones (e.g., clinical, wellness, lifestyle), visualizes keyword trends, and highlights opportunities for improved App Store Optimization (ASO). Results are presented through a clear, branded dashboard with downloadable reports—perfect for marketers focused on competitive positioning and organic discoverability.
        </p>
        <form method="post" enctype="multipart/form-data" class="file-upload">
            <input type="file" name="file" accept=".txt" required>
            <br><br>
            <input type="submit" value="Run Analyzer">
        </form>

        {% if status %}
            <h2>Status:</h2>
            <pre>{{ status }}</pre>
        {% endif %}

        {% if summary %}
            <h2>📈 Summary Metrics</h2>
            <table>{{ summary|safe }}</table>
        {% endif %}

        {% if top_keywords_table %}
            <h2>💡 Top Keyword Opportunities</h2>
            <table>{{ top_keywords_table|safe }}</table>
            <p class="chart-caption">
                This table lists the top 20 keywords based on their TF-IDF score, indicating which words are most distinctively used across analyzed apps.
                Keywords with high TF-IDF scores and broad usage (many apps) are strong candidates for organic discovery and search optimization.
            </p>
        {% endif %}

        {% if tone_chart %}
            <h2>🎨 Tone Distribution</h2>
            <img src="{{ tone_chart }}" width="500">
            <p class="chart-caption">
                This bar chart shows the tone breakdown of app messaging based on common wellness, clinical, or lifestyle language.
                It helps identify whether apps are positioning themselves as more empathetic, medical, or motivational.
            </p>
        {% endif %}

        {% if keyword_chart %}
            <h2>🔑 Top Keywords</h2>
            <img src="{{ keyword_chart }}" width="700">
            <p class="chart-caption">
                A visual ranking of the top 20 keywords weighted by TF-IDF. This helps highlight strategic themes that apps in this category are emphasizing.
            </p>
        {% endif %}

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
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    status = "Waiting for file upload..."
    download_buttons = []
    summary = ""
    tone_chart = ""
    keyword_chart = ""
    top_keywords_table = ""

    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename.endswith('.txt'):
            uploaded_file.save("app.txt")
            status = "📥 File uploaded. Running analyzer..."

            try:
                subprocess.run(["python", "keywordanalyzer2.py"], check=True)

                # Locate the latest analysis directory
                report_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and ("analysis_" in d)]
                latest_dir = max(report_dirs, key=os.path.getctime)
                files = os.listdir(latest_dir)

                # Load the app data
                app_df = pd.read_csv(os.path.join(latest_dir, [f for f in files if "apps" in f][0]))

                # Summaries
                summary_data = {
                    'Metric': [],
                    'Value': []
                }

                # Add total apps analyzed
                summary_data['Metric'].insert(0, 'Apps Analyzed')
                summary_data['Value'].insert(0, len(app_df))

                # Keyword effectiveness metrics
                if 'Keyword Count' in app_df.columns:
                    valid_keywords = app_df['Keyword Count'][app_df['Keyword Count'] > 0]
                    summary_data['Metric'].append('Avg Keyword Count')
                    summary_data['Value'].append(round(valid_keywords.mean(), 2) if not valid_keywords.empty else 0)

                if 'Keyword Ratio' in app_df.columns:
                    valid_ratios = app_df['Keyword Ratio'][app_df['Keyword Ratio'] > 0]
                    summary_data['Metric'].append('Avg Keyword Density')
                    summary_data['Value'].append(f"{round(valid_ratios.mean() * 100, 2)}%" if not valid_ratios.empty else "0%")

                # Tone distribution (for positioning analysis)
                if 'Tone' in app_df.columns:
                    tone_counts = app_df['Tone'].value_counts()
                    for tone, count in tone_counts.items():
                        summary_data['Metric'].append(f'{tone} Apps')
                        summary_data['Value'].append(count)

                summary_df = pd.DataFrame(summary_data)
                summary = summary_df.to_html(index=False)

                # Tone chart
                tone_counts = app_df['Tone'].value_counts()
                tone_plot = tone_counts.plot(kind="bar", color="#0A9396", title="Tone Distribution").get_figure()
                tone_chart_path = os.path.join(latest_dir, "tone_chart.png")
                tone_plot.savefig(tone_chart_path)
                tone_plot.clf()
                tone_chart = tone_chart_path

                # Keyword analysis
                kw_df = pd.read_csv(os.path.join(latest_dir, [f for f in files if "keywords" in f][0]))
                top_kw = kw_df.sort_values(by="TF-IDF Score", ascending=False).head(20)
                top_keywords_table = top_kw[['Keyword', 'TF-IDF Score', 'Raw Count', '# Apps Used In']].to_html(index=False)

                keyword_plot = top_kw.plot.barh(x="Keyword", y="TF-IDF Score", color="#005F73", title="Top Keywords").get_figure()
                keyword_chart_path = os.path.join(latest_dir, "keywords_chart.png")
                keyword_plot.savefig(keyword_chart_path)
                keyword_plot.clf()
                keyword_chart = keyword_chart_path

                # Download buttons
                for f in files:
                    path = os.path.join(latest_dir, f)
                    download_buttons.append(f'<a href="/{path}" class="btn" download>{f}</a>')

                if os.path.exists("failed_urls.txt"):
                    download_buttons.append(f'<a href="/failed_urls.txt" class="btn" download>failed_urls.txt</a>')

                status = "✅ Analysis complete. See results below."

            except subprocess.CalledProcessError as e:
                status = f"❌ Script failed: {e}"
            except Exception as ex:
                status = f"⚠️ Unexpected error: {ex}"

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
    app.run(debug=True, port=5000)
