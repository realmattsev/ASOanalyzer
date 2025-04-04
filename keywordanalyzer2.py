import requests
from bs4 import BeautifulSoup
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from datetime import datetime
import os
import matplotlib.pyplot as plt
import shutil
import json

nlp = spacy.load("en_core_web_sm")

DOMAIN_WHITELIST = set()

failed_urls = []

def scrape_app_metadata(url):
    if "apps.apple.com" not in url:
        print(f"Skipping non-App Store URL: {url}")
        failed_urls.append(url)
        return "", "", ""

    print(f"Scraping metadata for: {url}".encode('utf-8', errors='ignore').decode())
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the title from the schema.org metadata
        title = ""
        schema_script = soup.find("script", {"type": "application/ld+json"})
        if schema_script:
            try:
                schema_data = json.loads(schema_script.string)
                title = schema_data.get("name", "")
                # Clean up the title by removing HTML entities and extra spaces
                title = title.replace("&amp;", "&").strip()
            except:
                pass

        # If no title found in schema, try meta tags
        if not title:
            title_tag = soup.find("meta", {"property": "og:title"})
            if title_tag:
                title = title_tag.get("content", "").strip()
                # Remove "on the App Store" suffix if present
                title = title.replace(" on the App Store", "").strip()

        # Find subtitle from the app description section
        subtitle = ""
        desc_tag = soup.select_one("div.section__description, div.we-clamp, div.app-description, div[itemprop='description']")
        if desc_tag:
            # Get the first paragraph or first line of the description as subtitle
            first_p = desc_tag.find("p")
            if first_p:
                subtitle = first_p.get_text(strip=True)
            else:
                # Split by newlines and get the first non-empty line
                lines = [line.strip() for line in desc_tag.get_text(strip=True).split('\n') if line.strip()]
                subtitle = lines[0] if lines else ""

        # Get full description
        description = desc_tag.get_text(strip=True) if desc_tag else ""

        print(f"Title: {title[:30]}\nSubtitle: {subtitle[:30]}\nDescription length: {len(description)}")
        return title, subtitle, description
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        failed_urls.append(url)
        return "", "", ""

def classify_tone(text):
    wellness_terms = {"calm", "habit", "mind", "mood", "breathe", "relax", "sleep"}
    clinical_terms = {"therapy", "cbt", "depression", "anxiety", "diagnosis", "mental", "disorder"}
    lifestyle_terms = {"track", "coach", "routine", "goal", "daily", "step", "plan"}

    text = text.lower()
    w = sum(word in text for word in wellness_terms)
    c = sum(word in text for word in clinical_terms)
    l = sum(word in text for word in lifestyle_terms)

    if c > w and c > l:
        return "Clinical"
    elif w > c and w > l:
        return "Wellness"
    elif l > c and l > w:
        return "Lifestyle"
    else:
        return "Mixed"

def advanced_keyword_extraction(data_map, min_tfidf=0.03, min_keywords=100):
    print("Starting advanced keyword extraction...")
    stopwords = set(nlp.Defaults.stop_words)
    raw_counts = defaultdict(int)
    keyword_sources = defaultdict(set)
    app_metadata, texts, app_vectors = [], [], []
    app_urls = []

    for url, (primary, secondary) in data_map.items():
        # Clean and prepare the text
        primary = primary.strip() if primary else ""
        secondary = secondary.strip() if secondary else ""
        
        # Combine text for keyword extraction
        combined_text = f"{primary} {secondary}".strip()
        if not combined_text:
            continue

        doc = nlp(combined_text)
        keywords = []
        for token in doc:
            word = token.lemma_.lower().strip()
            if token.pos_ in {"NOUN", "VERB", "ADJ"} and len(word) > 2 and word.isalpha() and word not in stopwords:
                raw_counts[word] += 1
                keyword_sources[word].add(url)
                keywords.append(word)

        if keywords:
            texts.append(" ".join(keywords))
            app_urls.append(url)

        # Calculate metrics
        title_words = len(primary.split()) if primary else 0
        subtitle_words = len(secondary.split()) if secondary else 0
        total_words = len(combined_text.split())
        keyword_count = len(keywords)
        keyword_ratio = round(keyword_count / total_words, 3) if total_words > 0 else 0

        app_metadata.append({
            "App URL": url,
            "Primary Text": primary,
            "Secondary Text": secondary,
            "Tone": classify_tone(combined_text),
            "Title Length": title_words,
            "Subtitle Length": subtitle_words,
            "Total Words": total_words,
            "Keyword Count": keyword_count,
            "Keyword Ratio": keyword_ratio,
            "Title Char Length": len(primary),
            "Subtitle Char Length": len(secondary)
        })

    if not texts:
        print("No valid texts found. Exiting.")
        return [], app_metadata

    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X = vectorizer.fit_transform(texts)
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))

    # Save cosine similarity matrix for similarity analysis
    similarity_matrix = cosine_similarity(X)
    similarity_df = pd.DataFrame(similarity_matrix, index=app_urls, columns=app_urls)
    similarity_df.to_csv("competitor_similarity_matrix.csv")

    pos_weights = defaultdict(float)
    for doc in nlp.pipe(texts):
        for token in doc:
            word = token.lemma_.lower().strip()
            if word in tfidf_scores:
                pos_weights[word] += 1.5 if token.pos_ == "VERB" else 1.0

    filtered_data = [
        {"Keyword": word, "TF-IDF Score": round(tfidf_scores[word]*pos_weights[word],4),
         "Raw Count": raw_counts[word], "# Apps Used In": len(keyword_sources[word])}
        for word in tfidf_scores
        if tfidf_scores[word]*pos_weights[word] >= min_tfidf and (not DOMAIN_WHITELIST or word in DOMAIN_WHITELIST)
    ]

    sorted_keywords = sorted(filtered_data, key=lambda x: x["TF-IDF Score"], reverse=True)[:min_keywords]
    print(f"Extracted {len(sorted_keywords)} keywords.")
    return sorted_keywords, app_metadata

def save_keywords_to_csv(keyword_data, app_data, analysis_name="analysis"):
    if not keyword_data or not app_data:
        print("No data to save.")
        return None, None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"{analysis_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(keyword_data).to_csv(os.path.join(output_dir, f"keywords_{timestamp}.csv"), index=False)
    pd.DataFrame(app_data).to_csv(os.path.join(output_dir, f"apps_{timestamp}.csv"), index=False)

    zip_path = shutil.make_archive(output_dir, 'zip', output_dir)
    print(f"Report saved and zipped at: {zip_path}")
    return output_dir, zip_path

if __name__ == "__main__":
    with open("app.txt") as file:
        urls = [line.strip() for line in file if line.strip()]

    title_subtitle_map = {}
    description_map = {}
    
    for url in urls:
        title, subtitle, description = scrape_app_metadata(url)
        if title or subtitle:
            # Store title and subtitle separately
            title_subtitle_map[url] = (title, subtitle)
        if description:
            description_map[url] = (description, "")

    # Process title and subtitle data
    kw_titles, app_metadata_titles = advanced_keyword_extraction(title_subtitle_map)
    
    # Process description data separately
    kw_descriptions, app_metadata_descriptions = advanced_keyword_extraction(description_map)

    # Save both analyses
    save_keywords_to_csv(kw_titles, app_metadata_titles, "title_subtitle_analysis")
    save_keywords_to_csv(kw_descriptions, app_metadata_descriptions, "description_analysis")

    if failed_urls:
        with open("failed_urls.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(failed_urls))
        print(f"Saved {len(failed_urls)} failed URLs to failed_urls.txt")
