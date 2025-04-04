# Google Play Store Keyword Analyzer
# This script analyzes Google Play Store app listings to extract keywords and metadata
# for App Store Optimization (ASO) purposes.

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
import re

# Load the spaCy English language model for NLP tasks
nlp = spacy.load("en_core_web_sm")

# Broader whitelist of domain-specific terms
# These terms are important in the mental health and wellness domain
# and will be prioritized in keyword extraction
DOMAIN_WHITELIST = {
    # Mental health terms
    "mental", "health", "wellness", "wellbeing", "therapy", "counseling", "counselling",
    "anxiety", "depression", "stress", "meditation", "mindfulness", "self-care", "selfcare",
    "emotional", "psychology", "psychiatrist", "psychologist", "psychotherapy", "cbt", "dbt",
    "trauma", "ptsd", "ocd", "adhd", "add", "bipolar", "schizophrenia", "panic", "phobia",
    
    # Wellness terms
    "fitness", "exercise", "workout", "yoga", "meditation", "breathing", "breath", "relaxation",
    "sleep", "insomnia", "dream", "nightmare", "nightmare", "nightmare", "nightmare", "nightmare",
    "nutrition", "diet", "food", "eating", "weight", "obesity", "anorexia", "bulimia", "binge",
    
    # Lifestyle terms
    "habit", "routine", "schedule", "plan", "goal", "track", "monitor", "progress", "journal",
    "diary", "log", "record", "note", "reminder", "alert", "notification", "calendar", "event",
    "task", "todo", "checklist", "list", "item", "project", "activity", "exercise", "workout",
    
    # General health terms
    "health", "medical", "doctor", "physician", "nurse", "patient", "symptom", "diagnosis",
    "treatment", "cure", "heal", "healing", "recovery", "recover", "prevention", "prevent",
    "risk", "safety", "emergency", "crisis", "suicide", "self-harm", "selfharm", "addiction",
    
    # MindNation specific terms
    "mindnation", "mind", "nation", "teletherapy", "chat", "sos", "check-in", "checkin",
    "wellbeing", "productivity", "minded", "self-paced", "guided", "premium", "partner",
    "corporate", "carenow", "plan", "organization", "community", "team", "individual",
    "holistic", "needs-based", "impactful", "secure", "confidential", "data-based", "proactive",
    "culture", "driven", "approach", "solution", "customized", "organization", "talk", "training"
}

# Default parameters for keyword extraction
# These can be customized using the set_keyword_params function
DEFAULT_PARAMS = {
    "min_tfidf": 0.02,  # Lower threshold for TF-IDF score (was 0.03)
    "min_keywords": 200,  # Increased from 100 to capture more keywords
    "use_whitelist": True,  # Whether to prioritize domain-specific terms
    "include_pos": ["NOUN", "VERB", "ADJ"],  # Parts of speech to include
    "min_word_length": 3,  # Minimum word length to consider
    "max_ngram": 2  # Maximum n-gram size (1=unigrams, 2=bigrams, etc.)
}

# List to store URLs that failed during scraping
failed_urls = []

def scrape_google_play_metadata(url):
    """
    Scrape metadata from a Google Play Store app listing.
    
    Args:
        url (str): The URL of the Google Play Store app listing
        
    Returns:
        tuple: (title, subtitle, description) - The app's title, subtitle, and description
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0"
    }
    
    try:
        # Send a GET request to the URL with the specified headers
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try multiple selectors for title
        title_selectors = [
            "h1[itemprop='name']",  # Common class for app title
            "h1.AndroidAppTitle",   # Another common class
            "h1[class*='title']",   # Any h1 with 'title' in class
            "meta[property='og:title']",  # Open Graph meta tag
            "meta[name='title']"    # Standard meta tag
        ]
        
        title = None
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get('content', element.text).strip()
                if title:
                    break
        
        # Remove "on Google Play" suffix if present
        if title and "on Google Play" in title:
            title = title.replace("on Google Play", "").strip()
        
        # Try multiple selectors for description
        desc_selectors = [
            "div[itemprop='description']",  # Common class for description
            "div[class*='description']",    # Any div with 'description' in class
            "div[class*='desc']",           # Any div with 'desc' in class
            "div[class*='app-desc']",       # Another common class
            "div[class*='appDesc']"         # Another variation
        ]
        
        description = None
        for selector in desc_selectors:
            element = soup.select_one(selector)
            if element:
                description = element.text.strip()
                if description:
                    break
        
        # Extract subtitle (first line of description)
        subtitle = ""
        if description:
            # Split by newlines and get the first non-empty line
            lines = [line.strip() for line in description.split('\n') if line.strip()]
            if lines:
                subtitle = lines[0]
        
        return title, subtitle, description
        
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        failed_urls.append(url)
        return None, None, None

def classify_tone(text):
    """
    Classify the tone of a text as clinical, wellness, or lifestyle.
    
    Args:
        text (str): The text to classify
        
    Returns:
        str: The classified tone
    """
    if not text:
        return "Unknown"
    
    # Define keywords for each tone
    clinical_keywords = {"medical", "clinical", "doctor", "treatment", "diagnosis", "symptom", 
                        "therapy", "counseling", "psychiatrist", "psychologist", "patient"}
    wellness_keywords = {"wellness", "wellbeing", "health", "meditation", "mindfulness", 
                        "self-care", "selfcare", "balance", "holistic", "natural"}
    lifestyle_keywords = {"lifestyle", "habit", "routine", "daily", "everyday", "life", 
                         "living", "lifestyle", "work-life", "worklife", "productivity"}
    
    # Count occurrences of each tone's keywords
    text_lower = text.lower()
    clinical_count = sum(1 for word in clinical_keywords if word in text_lower)
    wellness_count = sum(1 for word in wellness_keywords if word in text_lower)
    lifestyle_count = sum(1 for word in lifestyle_keywords if word in text_lower)
    
    # Determine the dominant tone
    counts = {
        "Clinical": clinical_count,
        "Wellness": wellness_count,
        "Lifestyle": lifestyle_count
    }
    
    # If no tone is dominant, return "Mixed"
    if max(counts.values()) == 0:
        return "Mixed"
    
    # Return the tone with the highest count
    return max(counts, key=counts.get)

def advanced_keyword_extraction(app_data, min_tfidf=None, min_keywords=None):
    """
    Extract keywords from app metadata using TF-IDF and NLP techniques.
    
    Args:
        app_data (dict): Dictionary mapping app URLs to (primary_text, secondary_text) tuples
        min_tfidf (float, optional): Minimum TF-IDF score threshold
        min_keywords (int, optional): Maximum number of keywords to extract
        
    Returns:
        tuple: (keywords, app_metadata) - List of keywords and app metadata
    """
    # Use default parameters if not provided
    if min_tfidf is None:
        min_tfidf = DEFAULT_PARAMS["min_tfidf"]
    if min_keywords is None:
        min_keywords = DEFAULT_PARAMS["min_keywords"]
    
    # Initialize variables
    texts = []
    app_urls = []
    app_metadata = []
    raw_counts = defaultdict(int)
    keyword_sources = defaultdict(set)
    
    # Process each app's data
    for url, (primary, secondary) in app_data.items():
        # Clean and prepare text
        primary = primary.strip() if primary else ""
        secondary = secondary.strip() if secondary else ""
        
        # Combine primary and secondary text for keyword extraction
        combined_text = f"{primary} {secondary}"
        
        if combined_text:
            # Add to texts list for TF-IDF analysis
            texts.append(combined_text)
            app_urls.append(url)
            
            # Count raw occurrences of words
            doc = nlp(combined_text)
            for token in doc:
                word = token.lemma_.lower().strip()
                if len(word) >= DEFAULT_PARAMS["min_word_length"] and token.pos_ in DEFAULT_PARAMS["include_pos"]:
                    raw_counts[word] += 1
                    keyword_sources[word].add(url)
            
            # Calculate metrics
            title_word_count = len(primary.split()) if primary else 0
            subtitle_word_count = len(secondary.split()) if secondary else 0
            total_word_count = title_word_count + subtitle_word_count
            
            # Classify tone
            tone = classify_tone(combined_text)
            
            # Add metadata
            app_metadata.append({
                "App URL": url,
                "Primary Text": primary,
                "Secondary Text": secondary,
                "Tone": tone,
                "Title Length": title_word_count,
                "Subtitle Length": subtitle_word_count,
                "Total Words": total_word_count,
                "Title Char Length": len(primary) if primary else 0,
                "Subtitle Char Length": len(secondary) if secondary else 0
            })

    # If no valid texts found, return empty results
    if not texts:
        print("No valid texts found. Exiting.")
        return [], app_metadata

    # Calculate TF-IDF scores
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X = vectorizer.fit_transform(texts)
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))

    # Save cosine similarity matrix for similarity analysis
    similarity_matrix = cosine_similarity(X)
    similarity_df = pd.DataFrame(similarity_matrix, index=app_urls, columns=app_urls)
    similarity_df.to_csv("google_play_competitor_similarity_matrix.csv")

    # Apply part-of-speech weighting
    pos_weights = defaultdict(float)
    for doc in nlp.pipe(texts):
        for token in doc:
            word = token.lemma_.lower().strip()
            if word in tfidf_scores:
                pos_weights[word] += 1.5 if token.pos_ == "VERB" else 1.0

    # Apply whitelist filter if enabled
    whitelist_filter = lambda word: (not DEFAULT_PARAMS["use_whitelist"] or word in DOMAIN_WHITELIST)
    
    # Filter and sort keywords
    filtered_data = [
        {"Keyword": word, "TF-IDF Score": round(tfidf_scores[word]*pos_weights[word],4),
         "Raw Count": raw_counts[word], "# Apps Used In": len(keyword_sources[word])}
        for word in tfidf_scores
        if tfidf_scores[word]*pos_weights[word] >= min_tfidf and whitelist_filter(word)
    ]

    # Sort by TF-IDF score and limit to min_keywords
    sorted_keywords = sorted(filtered_data, key=lambda x: x["TF-IDF Score"], reverse=True)[:min_keywords]
    print(f"Extracted {len(sorted_keywords)} keywords.")
    return sorted_keywords, app_metadata

def save_keywords_to_csv(keyword_data, app_data, analysis_name="google_play_analysis"):
    """
    Save keyword and app data to CSV files.
    
    Args:
        keyword_data (list): List of keyword dictionaries
        app_data (list): List of app metadata dictionaries
        analysis_name (str): Base name for the analysis directory
        
    Returns:
        tuple: (output_dir, zip_path) - Paths to the output directory and zip file
    """
    if not keyword_data or not app_data:
        print("No data to save.")
        return None, None

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"{analysis_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save data to CSV files
    pd.DataFrame(keyword_data).to_csv(os.path.join(output_dir, f"google_play_keywords_{timestamp}.csv"), index=False)
    pd.DataFrame(app_data).to_csv(os.path.join(output_dir, f"google_play_apps_{timestamp}.csv"), index=False)

    # Create a zip archive of the output directory
    zip_path = shutil.make_archive(output_dir, 'zip', output_dir)
    print(f"Report saved and zipped at: {zip_path}")
    return output_dir, zip_path

def set_keyword_params(min_tfidf=None, min_keywords=None, use_whitelist=None, include_pos=None, min_word_length=None, max_ngram=None):
    """
    Set custom parameters for keyword extraction.
    
    Args:
        min_tfidf (float, optional): Minimum TF-IDF score threshold
        min_keywords (int, optional): Maximum number of keywords to extract
        use_whitelist (bool, optional): Whether to prioritize domain-specific terms
        include_pos (list, optional): Parts of speech to include
        min_word_length (int, optional): Minimum word length to consider
        max_ngram (int, optional): Maximum n-gram size
        
    Returns:
        dict: Updated parameters
    """
    global DEFAULT_PARAMS
    
    if min_tfidf is not None:
        DEFAULT_PARAMS["min_tfidf"] = min_tfidf
    if min_keywords is not None:
        DEFAULT_PARAMS["min_keywords"] = min_keywords
    if use_whitelist is not None:
        DEFAULT_PARAMS["use_whitelist"] = use_whitelist
    if include_pos is not None:
        DEFAULT_PARAMS["include_pos"] = include_pos
    if min_word_length is not None:
        DEFAULT_PARAMS["min_word_length"] = min_word_length
    if max_ngram is not None:
        DEFAULT_PARAMS["max_ngram"] = max_ngram
        
    return DEFAULT_PARAMS

def add_to_whitelist(terms):
    """
    Add terms to the domain whitelist.
    
    Args:
        terms (list): List of terms to add to the whitelist
        
    Returns:
        int: Updated size of the whitelist
    """
    global DOMAIN_WHITELIST
    
    for term in terms:
        DOMAIN_WHITELIST.add(term.lower())
        
    return len(DOMAIN_WHITELIST)

def test_analyzer():
    """
    Test the analyzer with a sample URL to verify it's working correctly.
    
    Returns:
        bool: True if test was successful, False otherwise
    """
    print("\n=== TESTING ANALYZER ===")
    test_url = "https://play.google.com/store/apps/details?id=com.MindNation.app&hl=en_US"
    print(f"Testing with URL: {test_url}")
    
    try:
        # Test scraping
        title, subtitle, description = scrape_google_play_metadata(test_url)
        print(f"Title: {title}")
        print(f"Subtitle: {subtitle}")
        print(f"Description length: {len(description) if description else 0}")
        
        # Test keyword extraction
        test_map = {test_url: (title, subtitle)}
        kw, metadata = advanced_keyword_extraction(test_map)
        print(f"Extracted {len(kw)} keywords")
        
        # Test saving
        output_dir, zip_path = save_keywords_to_csv(kw, metadata, "google_play_test_analysis")
        print(f"Test output saved to: {output_dir}")
        
        print("=== TEST COMPLETE ===")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting Google Play Store Keyword Analyzer...")
    print(f"Using parameters: {DEFAULT_PARAMS}")
    print(f"Whitelist contains {len(DOMAIN_WHITELIST)} terms")
    
    # Run test first
    test_analyzer()
    
    try:
        # Read URLs from gapp.txt
        with open("gapp.txt") as file:
            urls = [line.strip() for line in file if line.strip()]
        
        print(f"Found {len(urls)} URLs to analyze")
        
        # Initialize data structures
        title_subtitle_map = {}
        description_map = {}
        
        # Process each URL
        for i, url in enumerate(urls, 1):
            print(f"\nProcessing URL {i}/{len(urls)}: {url}")
            title, subtitle, description = scrape_google_play_metadata(url)
            
            # Store title and subtitle data
            if title or subtitle:
                print(f"Found title: '{title[:30]}...' and subtitle: '{subtitle[:30]}...'")
                title_subtitle_map[url] = (title, subtitle)
            else:
                print("No title or subtitle found")
                
            # Store description data
            if description:
                print(f"Found description with {len(description)} characters")
                description_map[url] = (description, "")
            else:
                print("No description found")

        print(f"\nProcessed {len(title_subtitle_map)} apps with title/subtitle data")
        print(f"Processed {len(description_map)} apps with description data")

        # Process title and subtitle data
        print("\nAnalyzing title and subtitle data...")
        kw_titles, app_metadata_titles = advanced_keyword_extraction(title_subtitle_map)
        
        # Process description data separately
        print("\nAnalyzing description data...")
        kw_descriptions, app_metadata_descriptions = advanced_keyword_extraction(description_map)

        # Save both analyses
        print("\nSaving analysis results...")
        save_keywords_to_csv(kw_titles, app_metadata_titles, "google_play_title_subtitle_analysis")
        save_keywords_to_csv(kw_descriptions, app_metadata_descriptions, "google_play_description_analysis")

        # Save failed URLs if any
        if failed_urls:
            with open("google_play_failed_urls.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(failed_urls))
            print(f"Saved {len(failed_urls)} failed URLs to google_play_failed_urls.txt")
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc() 