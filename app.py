# streamlit_app.py
import streamlit as st
import pandas as pd
import re
import time
from serpapi.google_search import GoogleSearch  # Correct import
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64
from collections import Counter

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import scipy.sparse as sp

# Page configuration MUST be the first Streamlit command
st.set_page_config(page_title="VibeChek AI Dashboard", layout="wide")

# Define standard figure sizes for consistent display
FIGURE_SIZES = {
    "large": (7, 3.5),      # For main visualizations
    "medium": (4, 2.5),     # For secondary visualizations 
    "small": (3.5, 2.3),    # For compact visualizations
    "pie": (3, 2.5)         # Specifically for pie charts
}

# Add custom CSS for better spacing and containment
st.markdown("""
<style>
    .plot-container {
        max-width: 90%;
        margin: 0 auto;
    }
    .section-divider {
        margin-top: 2em;
        margin-bottom: 1em;
    }
    .subsection-divider {
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    /* Make all charts and visualizations more compact */
    .stPlotlyChart, .stChart {
        max-width: 90% !important;
        margin: 0 auto !important;
    }
    /* Add custom scaling for better fit */
    div[data-testid="stImage"] img {
        max-width: 90% !important;
        display: block !important;
        margin: 0 auto !important;
    }
</style>
""", unsafe_allow_html=True)

# ML functions for smart recommendations
def prepare_review_features(reviews_df):
    """Transform reviews into feature vectors for ML"""
    
    # Skip if no reviews
    if len(reviews_df) < 10:
        return None
    
    # Text features
    reviews_df["word_count"] = reviews_df["Cleaned_Review"].apply(lambda x: len(str(x).split()))
    reviews_df["char_count"] = reviews_df["Cleaned_Review"].apply(lambda x: len(str(x)))
    
    # Create TF-IDF features from review text
    tfidf = TfidfVectorizer(
        max_features=200,  # Limit features to avoid overfitting
        min_df=2,          # Ignore terms that appear in fewer than 2 documents
        max_df=0.85,       # Ignore terms that appear in more than 85% of documents
        stop_words='english'
    )
    
    # Create a sparse matrix of TF-IDF features
    tfidf_matrix = tfidf.fit_transform(reviews_df["Cleaned_Review"].fillna(""))
    
    # Create a feature for sentiment (0 for Negative, 1 for Neutral, 2 for Positive)
    sentiment_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
    reviews_df["sentiment_numeric"] = reviews_df["Sentiment"].map(sentiment_map)
    
    # Create feature names
    feature_names = tfidf.get_feature_names_out().tolist()
    
    return {
        "feature_matrix": tfidf_matrix,
        "feature_names": feature_names,
        "vectorizer": tfidf,
        "reviews_with_features": reviews_df
    }

def discover_review_aspects(feature_data, num_aspects=6):
    """Use NMF to discover key aspects in reviews"""
    
    if feature_data is None or feature_data["feature_matrix"].shape[0] < 10:
        return None
    
    # Adjust number of aspects based on data size
    num_aspects = min(num_aspects, max(2, feature_data["feature_matrix"].shape[0] // 10))
    
    # Non-negative Matrix Factorization for topic modeling/aspect extraction
    nmf = NMF(n_components=num_aspects, random_state=42)
    nmf_features = nmf.fit_transform(feature_data["feature_matrix"])
    
    # Get top terms for each aspect
    aspects = []
    feature_names = feature_data["feature_names"]
    
    for aspect_idx, aspect in enumerate(nmf.components_):
        # Get top 10 terms for this aspect
        top_terms_idx = aspect.argsort()[-10:][::-1]
        top_terms = [feature_names[i] for i in top_terms_idx if i < len(feature_names)]
        
        # Generate aspect name from top terms (simplified)
        aspect_name = f"Aspect {aspect_idx+1}: {top_terms[0].capitalize()}"
        
        aspects.append({
            "id": aspect_idx,
            "name": aspect_name,
            "top_terms": top_terms,
            "weight": np.sum(aspect)  # Importance of this aspect
        })
    
    # Sort aspects by weight
    aspects.sort(key=lambda x: x["weight"], reverse=True)
    
    return {
        "aspects": aspects,
        "nmf_model": nmf,
        "aspect_features": nmf_features
    }

def analyze_aspect_sentiment(reviews_df, aspects_data):
    """Analyze sentiment for each aspect"""
    
    if aspects_data is None:
        return None
        
    # For each aspect, calculate sentiment distribution
    aspect_sentiments = []
    
    for aspect_idx, aspect in enumerate(aspects_data["aspects"]):
        # Get aspect weights for each review
        aspect_weights = aspects_data["aspect_features"][:, aspect_idx]
        
        # Find reviews where this aspect is prominent (weight > mean + std)
        threshold = np.mean(aspect_weights) + np.std(aspect_weights)
        relevant_indices = np.where(aspect_weights > threshold)[0]
        
        # Get sentiments for these reviews
        relevant_sentiments = reviews_df.iloc[relevant_indices]["Sentiment"].value_counts(normalize=True)
        
        # Calculate a sentiment score (-1 to 1)
        sentiment_score = 0
        if "Positive" in relevant_sentiments:
            sentiment_score += relevant_sentiments["Positive"]
        if "Negative" in relevant_sentiments:
            sentiment_score -= relevant_sentiments["Negative"]
            
        # Store results
        aspect_sentiments.append({
            "aspect_id": aspect_idx,
            "aspect_name": aspect["name"],
            "sentiment_score": sentiment_score,
            "sentiment_dist": relevant_sentiments.to_dict() if not relevant_sentiments.empty else {},
            "relevant_reviews": len(relevant_indices),
            "top_terms": aspect["top_terms"]
        })
    
    return aspect_sentiments

def cluster_reviews(reviews_df, aspects_data, num_clusters=4):
    """Cluster reviews based on aspect weights"""
    
    if aspects_data is None:
        return None
        
    # Use aspect features for clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(aspects_data["aspect_features"])
    
    # Add cluster to dataframe
    reviews_df_with_clusters = reviews_df.copy()
    reviews_df_with_clusters["cluster"] = clusters
    
    # Analyze each cluster
    cluster_insights = []
    
    for cluster_id in range(num_clusters):
        # Get reviews in this cluster
        cluster_reviews = reviews_df_with_clusters[reviews_df_with_clusters["cluster"] == cluster_id]
        
        if len(cluster_reviews) == 0:
            continue
            
        # Check sentiment distribution
        sentiment_dist = cluster_reviews["Sentiment"].value_counts(normalize=True)
        
        # Find dominant aspects for this cluster
        cluster_centroid = kmeans.cluster_centers_[cluster_id]
        dominant_aspects = []
        
        for aspect_idx, weight in enumerate(cluster_centroid):
            if weight > np.mean(cluster_centroid) + 0.5 * np.std(cluster_centroid):
                dominant_aspects.append({
                    "aspect_id": aspect_idx,
                    "aspect_name": aspects_data["aspects"][aspect_idx]["name"],
                    "weight": weight
                })
        
        # Get sample reviews
        sample_reviews = cluster_reviews["snippet"].head(3).tolist()
        
        # Determine cluster sentiment
        pos_pct = sentiment_dist.get("Positive", 0)
        neg_pct = sentiment_dist.get("Negative", 0)
        
        if pos_pct > 0.6:
            cluster_sentiment = "Positive"
        elif neg_pct > 0.4:
            cluster_sentiment = "Negative"
        else:
            cluster_sentiment = "Mixed"
        
        # Store cluster insights
        cluster_insights.append({
            "cluster_id": cluster_id,
            "size": len(cluster_reviews),
            "sentiment": cluster_sentiment,
            "sentiment_dist": sentiment_dist.to_dict(),
            "dominant_aspects": dominant_aspects,
            "sample_reviews": sample_reviews
        })
    
    return cluster_insights

def generate_ml_recommendations(reviews_df):
    """Generate smart recommendations using ML"""
    
    # Check if we have enough data
    if len(reviews_df) < 20:
        return {
            "status": "insufficient_data",
            "message": "Need at least 20 reviews for ML-based recommendations"
        }
    
    try:
        # 1. Prepare features
        feature_data = prepare_review_features(reviews_df)
        
        if feature_data is None:
            return {
                "status": "error",
                "message": "Error preparing features"
            }
        
        # 2. Discover aspects
        num_aspects = min(8, len(reviews_df) // 10)  # Scale with data size
        aspects_data = discover_review_aspects(feature_data, num_aspects)
        
        if aspects_data is None:
            return {
                "status": "error",
                "message": "Error discovering aspects"
            }
        
        # 3. Analyze sentiment for each aspect
        aspect_sentiments = analyze_aspect_sentiment(reviews_df, aspects_data)
        
        # 4. Cluster reviews
        num_clusters = min(5, len(reviews_df) // 20)  # Scale with data size
        cluster_insights = cluster_reviews(reviews_df, aspects_data, num_clusters)
        
        # 5. Generate recommendations
        positive_insights = []
        negative_insights = []
        action_items = []
        
        # Add aspect-based insights
        if aspect_sentiments:
            # Positive aspect insights
            for aspect in aspect_sentiments:
                if aspect["sentiment_score"] > 0.3 and aspect["relevant_reviews"] >= 3:
                    terms = ", ".join(aspect["top_terms"][:3])
                    insight = f"Customers value your {terms} (mentioned in {aspect['relevant_reviews']} reviews with positive sentiment)"
                    positive_insights.append(insight)
            
            # Negative aspect insights
            for aspect in aspect_sentiments:
                if aspect["sentiment_score"] < -0.2 and aspect["relevant_reviews"] >= 2:
                    terms = ", ".join(aspect["top_terms"][:3])
                    insight = f"Customers have concerns about {terms} (mentioned in {aspect['relevant_reviews']} reviews with negative sentiment)"
                    negative_insights.append(insight)
                    
                    # Also generate an action item
                    action = f"Address issues related to {terms} based on customer feedback"
                    action_items.append(action)
        
        # Add cluster-based insights
        if cluster_insights:
            # Positive cluster insights
            for cluster in cluster_insights:
                if cluster["sentiment"] == "Positive" and cluster["size"] >= 5:
                    aspects_text = ", ".join([a["aspect_name"].split(": ")[1] for a in cluster["dominant_aspects"][:2]])
                    if aspects_text:
                        insight = f"A significant group of customers ({cluster['size']}) praise your {aspects_text}"
                        positive_insights.append(insight)
            
            # Negative cluster insights
            for cluster in cluster_insights:
                if cluster["sentiment"] == "Negative" and cluster["size"] >= 3:
                    aspects_text = ", ".join([a["aspect_name"].split(": ")[1] for a in cluster["dominant_aspects"][:2]])
                    if aspects_text:
                        insight = f"A group of {cluster['size']} reviews show concerns about {aspects_text}"
                        negative_insights.append(insight)
                        
                        # Also generate an action item
                        action = f"Investigate and improve {aspects_text} based on the identified customer segment"
                        action_items.append(action)
        
        # Add general insights based on overall sentiment
        sentiment_counts = reviews_df["Sentiment"].value_counts(normalize=True)
        pos_pct = sentiment_counts.get("Positive", 0)
        neg_pct = sentiment_counts.get("Negative", 0)
        
        if pos_pct > 0.75:
            positive_insights.append(f"Overall sentiment is very positive ({pos_pct*100:.1f}%). Consider using reviews in marketing.")
            action_items.append("Develop a testimonial program featuring your most enthusiastic customer reviews")
        elif neg_pct > 0.4:
            negative_insights.append(f"Overall sentiment shows concerns ({neg_pct*100:.1f}% negative). Consider a review of operations.")
            action_items.append("Conduct a comprehensive review of business operations to address core customer concerns")
        
        # Add standard action items
        action_items.extend([
            "Respond to negative reviews promptly and professionally",
            "Track sentiment trends monthly to measure improvement",
            "Implement a customer feedback system to catch issues before they result in negative reviews"
        ])
        
        # Deduplicate and limit insights
        positive_insights = list(dict.fromkeys(positive_insights))[:5]  # Take top 5 unique insights
        negative_insights = list(dict.fromkeys(negative_insights))[:5]  # Take top 5 unique insights
        action_items = list(dict.fromkeys(action_items))[:5]  # Take top 5 unique actions
        
        return {
            "status": "success",
            "positive_insights": positive_insights,
            "negative_insights": negative_insights,
            "action_items": action_items,
            "aspects_data": aspects_data["aspects"],
            "cluster_insights": cluster_insights,
            "aspect_sentiments": aspect_sentiments
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

def visualize_aspects(aspects_data, aspect_sentiments, figure_sizes):
    """Create visualizations for discovered aspects and their sentiments"""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    if not aspects_data or not aspect_sentiments:
        return None, None
    
    # 1. Create aspect importance visualization
    aspect_names = [a["name"].split(": ")[1] for a in aspects_data]
    aspect_weights = [a["weight"] for a in aspects_data]
    
    # Normalize weights for better visualization
    aspect_weights = [w / max(aspect_weights) for w in aspect_weights]
    
    # Create horizontal bar chart of aspect importance
    fig1, ax1 = plt.subplots(figsize=figure_sizes["medium"])
    y_pos = np.arange(len(aspect_names))
    
    bars = ax1.barh(y_pos, aspect_weights, align='center', color='skyblue')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(aspect_names)
    ax1.invert_yaxis()  # Labels read top-to-bottom
    ax1.set_title('Key Aspects in Reviews', fontsize=10)
    ax1.set_xlabel('Relative Importance', fontsize=9)
    
    # Add labels with top terms for each aspect
    for i, bar in enumerate(bars):
        width = bar.get_width()
        terms = ", ".join(aspects_data[i]["top_terms"][:3])
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                 terms, va='center', fontsize=7)
    
    plt.tight_layout()
    
    # 2. Create aspect sentiment visualization
    sentiment_scores = [a["sentiment_score"] for a in aspect_sentiments]
    aspect_names = [a["aspect_name"].split(": ")[1] for a in aspect_sentiments]
    
    # Sort by sentiment score
    sorted_indices = np.argsort(sentiment_scores)
    sorted_scores = [sentiment_scores[i] for i in sorted_indices]
    sorted_names = [aspect_names[i] for i in sorted_indices]
    
    # Set colors based on sentiment (red for negative, green for positive)
    colors = ['red' if s < 0 else 'green' for s in sorted_scores]
    
    fig2, ax2 = plt.subplots(figsize=figure_sizes["medium"])
    y_pos = np.arange(len(sorted_names))
    
    bars = ax2.barh(y_pos, sorted_scores, align='center', color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_names)
    ax2.invert_yaxis()  # Labels read top-to-bottom
    ax2.set_title('Aspect Sentiment Analysis', fontsize=10)
    ax2.set_xlabel('Sentiment Score (-1 to 1)', fontsize=9)
    
    # Add a vertical line at 0
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels with review counts
    for i, bar in enumerate(bars):
        width = bar.get_width()
        aspect_idx = sorted_indices[i]
        review_count = aspect_sentiments[aspect_idx]["relevant_reviews"]
        
        label_x = width + 0.01 if width >= 0 else width - 0.01
        ha = 'left' if width >= 0 else 'right'
        
        ax2.text(label_x, bar.get_y() + bar.get_height()/2, 
                 f"{review_count} reviews", va='center', ha=ha, fontsize=7)
    
    plt.tight_layout()
    
    return fig1, fig2

# Helper functions
def clean_text(text):
    """Clean text by removing URLs, special characters, and converting to lowercase"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text.strip().lower()

def vader_sentiment(text):
    """Basic VADER sentiment analysis"""
    if not text:
        return "Neutral"
    score = sia.polarity_scores(text)["compound"]
    return "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"

def enhanced_business_sentiment(text):
    """Enhanced business-specific sentiment analysis"""
    if not text:
        return "Neutral"
    
    # Get the base VADER scores
    score = sia.polarity_scores(text)["compound"]
    
    # Business-specific sentiment boosters
    business_positive = [
        'recommend', 'excellent', 'amazing', 'love', 'best', 
        'friendly', 'helpful', 'clean', 'professional', 'fresh',
        'worth', 'perfect', 'fantastic', 'awesome', 'definitely'
    ]
    
    business_negative = [
        'waste', 'overpriced', 'rude', 'slow', 'dirty',
        'terrible', 'horrible', 'avoid', 'disappointing', 'cold',
        'manager', 'complained', 'waiting', 'problem', 'never again'
    ]
    
    # Check for business-specific terms and adjust score
    text_lower = text.lower()
    
    # Apply modest boosts to the compound score for business-specific terms
    for term in business_positive:
        if term in text_lower:
            score = min(1.0, score + 0.05)
            
    for term in business_negative:
        if term in text_lower:
            score = max(-1.0, score - 0.05)
    
    # Adjust thresholds for business reviews (they tend to be more polarized)
    if score >= 0.1:
        return "Positive"
    elif score <= -0.1:
        return "Negative"
    else:
        return "Neutral"

def get_top_words(reviews, n=10):
    """Extract and count the most common words in reviews"""
    if not hasattr(reviews, 'any') or not reviews.any():
        return []
            
    # Combine all review text
    all_text = " ".join(reviews)
    
    # Split into words and count
    words = re.findall(r'\b\w+\b', all_text.lower())
    
    # Simple stopwords filtering
    stopwords = {'the', 'a', 'an', 'and', 'is', 'in', 'it', 'to', 'was', 'for', 
                 'of', 'with', 'on', 'at', 'by', 'this', 'that', 'but', 'are', 
                 'be', 'or', 'have', 'has', 'had', 'not', 'what', 'all', 'were', 
                 'when', 'where', 'who', 'which', 'their', 'they', 'them', 'there',
                 'from', 'out', 'some', 'would', 'about', 'been', 'many', 'us', 'we'}
    
    # Filter out stopwords and short words
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Count word frequency
    word_counts = Counter(filtered_words)
    
    # Return top N words
    return word_counts.most_common(n)

@st.cache_data
def convert_df_to_csv(df):
    """Convert dataframe to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

# Download NLTK data at startup
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# App title and description
st.title("🧠 VibeChek: Google Review Analyzer")

st.markdown("""
**Don't know your Place ID?**
🔗 [Find your Google Place ID here](https://developers.google.com/maps/documentation/places/web-service/place-id)
Search for your business and copy the Place ID.
""")

# Get API Key from secrets
try:
    SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
except Exception:
    SERPAPI_KEY = st.text_input("Enter your SerpAPI Key", type="password")
    if not SERPAPI_KEY:
        st.warning("Please enter your SerpAPI Key to continue")
        st.stop()

# Initialize session state for storing data
if "reviews_df" not in st.session_state:
    st.session_state.reviews_df = None

# Input Place ID
place_id = st.text_input("📍 Enter Google Place ID")

# Max Reviews
max_reviews = st.slider("🔄 How many reviews to fetch?", min_value=50, max_value=500, step=50, value=150)

if st.button("🚀 Fetch & Analyze Reviews") and place_id:
    try:
        with st.spinner("Fetching reviews from Google Maps..."):
            # Create params with error handling
            params = {
                "engine": "google_maps_reviews",
                "place_id": place_id,
                "api_key": SERPAPI_KEY,
            }
            
            # Test API connection with just one request first
            try:
                # Explicitly use the correct import path
                test_search = GoogleSearch(params)
                test_results = test_search.get_dict()
                
                if "error" in test_results:
                    st.error(f"API Error: {test_results['error']}")
                    st.stop()
                    
                if "reviews" not in test_results:
                    st.warning("No reviews found for this Place ID. Please verify the ID is correct.")
                    st.stop()
                    
            except Exception as e:
                st.error(f"Error connecting to SerpAPI: {str(e)}")
                st.stop()
            
            # Now fetch all reviews
            all_reviews = []
            start = 0
            
            progress_bar = st.progress(0)
            
            # Make multiple requests with pagination
            while len(all_reviews) < max_reviews:
                params["start"] = start
                
                try:
                    search = GoogleSearch(params)
                    results = search.get_dict()
                    reviews = results.get("reviews", [])
                    
                    if not reviews:
                        break
                        
                    all_reviews.extend(reviews)
                    start += len(reviews)
                    
                    # Update progress
                    progress = min(len(all_reviews) / max_reviews, 1.0)
                    progress_bar.progress(progress)
                    
                    # Sleep to respect API rate limits
                    time.sleep(2)
                    
                except Exception as e:
                    st.warning(f"Error during pagination (fetched {len(all_reviews)} reviews so far): {str(e)}")
                    break
            
            if not all_reviews:
                st.error("No reviews could be fetched. Please check your Place ID and API key.")
                st.stop()
                
            df = pd.DataFrame(all_reviews[:max_reviews])
            
            # Handle missing columns - sometimes SerpAPI response structure varies
            for col in ['snippet', 'rating', 'time']:
                if col not in df.columns:
                    df[col] = None
            
            # Basic data validation
            df = df.dropna(subset=['snippet'])
            
            if len(df) == 0:
                st.error("No valid reviews found after filtering.")
                st.stop()
                
            st.session_state.reviews_df = df
            st.success(f"✅ {len(df)} reviews fetched!")
    
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.stop()
    
    # Process the data
    try:
        with st.spinner("Processing reviews..."):
            # Clean reviews
            df["Cleaned_Review"] = df["snippet"].apply(clean_text)
            
            # Apply enhanced sentiment analysis
            df["Sentiment"] = df["Cleaned_Review"].apply(enhanced_business_sentiment)
            
            # Simple ratings analysis with colorful bars
            if "rating" in df.columns and df["rating"].notna().any():
                # Create a figure with adjusted height to accommodate the legend
                fig, ax = plt.subplots(figsize=(FIGURE_SIZES["medium"][0], FIGURE_SIZES["medium"][1] + 0.5))
                
                # Ensure we're working with numeric ratings and convert to integers if needed
                df['rating_num'] = pd.to_numeric(df['rating'], errors='coerce')
                rating_counts = df['rating_num'].value_counts().sort_index()
                
                # Define color map - use both integer and float keys to handle both types
                color_map = {
                    1: '#d73027', 1.0: '#d73027',     # Red
                    2: '#fc8d59', 2.0: '#fc8d59',     # Orange
                    3: '#ffffbf', 3.0: '#ffffbf',     # Yellow
                    4: '#91cf60', 4.0: '#91cf60',     # Light green
                    5: '#1a9850', 5.0: '#1a9850',     # Dark green
                }
                
                # Explicitly create arrays for plotting rather than using pandas series directly
                ratings = rating_counts.index.tolist()
                counts = rating_counts.values.tolist()
                
                # Create a list of colors for each bar - with fallback default color
                bar_colors = []
                for rating in ratings:
                    if rating in color_map:
                        bar_colors.append(color_map[rating])
                    else:
                        # Fallback to blue if we somehow get an unexpected rating
                        bar_colors.append('#4575b4')
                
                # Create the plot
                bars = ax.bar(ratings, counts, color=bar_colors)
                
                # Add rating values on top of each bar - with smaller font
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            str(int(count)), ha='center', va='bottom', fontweight='bold', fontsize=8)
                
                # Set labels and title with smaller font
                ax.set_xlabel("Rating", fontsize=9)
                ax.set_ylabel("Count", fontsize=9)
                ax.set_title("Rating Distribution", fontsize=10)
                
                # Set x-axis ticks - explicitly use only integer ratings from 1-5
                ax.set_xticks([1, 2, 3, 4, 5])
                ax.tick_params(axis='both', which='major', labelsize=8)
                
                # Set y-axis to start at 0
                ax.set_ylim(bottom=0)
                
                # Add a legend explaining the color scheme - with smaller font
                # MOVED BELOW THE PLOT
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#d73027', label='1 Star'),
                    Patch(facecolor='#fc8d59', label='2 Stars'),
                    Patch(facecolor='#ffffbf', label='3 Stars'),
                    Patch(facecolor='#91cf60', label='4 Stars'),
                    Patch(facecolor='#1a9850', label='5 Stars')
                ]
                
                # Place legend below the plot
                ax.legend(handles=legend_elements, 
                          title="Rating Colors", 
                          loc='upper center', 
                          bbox_to_anchor=(0.5, -0.15),
                          ncol=5, 
                          fontsize=7, 
                          title_fontsize=8)
                
                # Adjust figure to make room for the legend
                plt.subplots_adjust(bottom=0.2)
                
                # Improve layout
                plt.tight_layout()
                
                # Show the plot with container
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # Show sentiment distribution
            st.subheader("📊 Sentiment Analysis")
            sentiment_counts = df["Sentiment"].value_counts()
            
            # RESIZED pie chart and smaller fonts
            fig, ax = plt.subplots(figsize=FIGURE_SIZES["pie"])
            colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}
            wedges, texts, autotexts = ax.pie(
                sentiment_counts, 
                labels=sentiment_counts.index, 
                autopct='%1.1f%%',
                colors=[colors.get(x, 'blue') for x in sentiment_counts.index],
                textprops={'fontsize': 8}
            )
            for autotext in autotexts:
                autotext.set_fontsize(8)
            
            ax.set_title("Sentiment Distribution", fontsize=10)
            plt.tight_layout()
            
            # Show the plot with container
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show the data
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.subheader("📋 Review Data")
            st.dataframe(df[["snippet", "Sentiment"]].head(10))
    
    except Exception as e:
        st.error(f"Error in data processing: {str(e)}")
        st.stop()
    
    
    # Word Clouds - only if we have enough data
    try:
        if len(df) > 5:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.subheader("☁️ Word Clouds")
            col1, col2 = st.columns(2)
            
            with col1:
                pos_reviews = df[df["Sentiment"] == "Positive"]["Cleaned_Review"].dropna()
                if len(pos_reviews) > 0:
                    pos_text = " ".join(pos_reviews)
                    if len(pos_text) > 50:  # Ensure we have enough text
                        # Smaller word cloud
                        wc_pos = WordCloud(width=300, height=200, background_color="white").generate(pos_text)
                        # RESIZED word cloud
                        fig, ax = plt.subplots(figsize=FIGURE_SIZES["small"])
                        ax.imshow(wc_pos, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                        st.caption("Positive Reviews")
                    else:
                        st.info("Not enough positive review text for word cloud")
                else:
                    st.info("No positive reviews found")
            
            with col2:
                neg_reviews = df[df["Sentiment"] == "Negative"]["Cleaned_Review"].dropna()
                if len(neg_reviews) > 0:
                    neg_text = " ".join(neg_reviews)
                    if len(neg_text) > 50:  # Ensure we have enough text
                        # Smaller word cloud
                        wc_neg = WordCloud(width=300, height=200, background_color="black", colormap="Reds").generate(neg_text)
                        # RESIZED word cloud
                        fig, ax = plt.subplots(figsize=FIGURE_SIZES["small"])
                        ax.imshow(wc_neg, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                        st.caption("Negative Reviews")
                    else:
                        st.info("Not enough negative review text for word cloud")
                else:
                    st.info("No negative reviews found")
    except Exception as e:
        st.warning(f"Error generating word clouds: {str(e)}")
    
    # Top words analysis - simple word frequency analysis
    try:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("🔍 Common Words Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Positive Review Keywords")
            pos_words = get_top_words(df[df["Sentiment"] == "Positive"]["Cleaned_Review"])
            
            if pos_words:
                # Create a bar chart with RESIZED dimensions and smaller fonts
                pos_df = pd.DataFrame(pos_words, columns=['Word', 'Count'])
                fig, ax = plt.subplots(figsize=FIGURE_SIZES["small"])
                bars = ax.barh(pos_df['Word'][::-1], pos_df['Count'][::-1], color='green')
                
                # Add count values next to each bar
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                            str(int(width)), va='center', fontsize=7)
                            
                ax.set_title("Top Words in Positive Reviews", fontsize=9)
                ax.tick_params(axis='both', which='major', labelsize=8)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Not enough data for positive keyword analysis")
        
        with col2:
            st.markdown("#### Negative Review Keywords")
            neg_words = get_top_words(df[df["Sentiment"] == "Negative"]["Cleaned_Review"])
            
            if neg_words:
                # Create a bar chart with RESIZED dimensions and smaller fonts
                neg_df = pd.DataFrame(neg_words, columns=['Word', 'Count'])
                fig, ax = plt.subplots(figsize=FIGURE_SIZES["small"])
                bars = ax.barh(neg_df['Word'][::-1], neg_df['Count'][::-1], color='red')
                
                # Add count values next to each bar
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                            str(int(width)), va='center', fontsize=7)
                            
                ax.set_title("Top Words in Negative Reviews", fontsize=9)
                ax.tick_params(axis='both', which='major', labelsize=8)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Not enough data for negative keyword analysis")
    except Exception as e:
        st.warning(f"Error in keyword analysis: {str(e)}")
        
    # Smart Recommendations with ML only
    try:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("🤖 ML-Based Smart Recommendations")
        
        # Only run if we have enough data
        if len(df) >= 20:  # Need reasonable amount for ML
            with st.spinner("Training machine learning models on your reviews..."):
                # Run the ML recommendation system
                ml_recommendations = generate_ml_recommendations(df)
                
                if ml_recommendations["status"] == "success":
                    # Show discovered aspects
                    st.markdown("### 🔍 Key Aspects Discovered in Reviews")
                    
                    # Create visualizations
                    aspect_fig, sentiment_fig = visualize_aspects(
                        ml_recommendations["aspects_data"],
                        ml_recommendations["aspect_sentiments"],
                        FIGURE_SIZES
                    )
                    
                    if aspect_fig:
                        # Show aspect importance visualization
                        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                        st.pyplot(aspect_fig)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    if sentiment_fig:
                        # Show aspect sentiment visualization
                        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                        st.pyplot(sentiment_fig)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display recommendations
                    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 🟢 Strengths to Maintain")
                        if ml_recommendations["positive_insights"]:
                            for i, insight in enumerate(ml_recommendations["positive_insights"], 1):
                                st.markdown(f"{i}. {insight}")
                        else:
                            st.info("ML model couldn't identify significant strengths")
                    
                    with col2:
                        st.markdown("### 🔴 Areas for Improvement")
                        if ml_recommendations["negative_insights"]:
                            for i, insight in enumerate(ml_recommendations["negative_insights"], 1):
                                st.markdown(f"{i}. {insight}")
                        else:
                            st.info("ML model couldn't identify significant concerns")
                    
                    # Add a section for actionable next steps
                    st.markdown('<div class="subsection-divider"></div>', unsafe_allow_html=True)
                    st.markdown("### 📝 Actionable Next Steps")
                    
                    if ml_recommendations["action_items"]:
                        for i, action in enumerate(ml_recommendations["action_items"], 1):
                            st.markdown(f"{i}. {action}")
                    else:
                        st.info("ML model couldn't generate specific action items")
                        
                    # Add explanation of ML approach
                    with st.expander("ℹ️ How the ML Recommendation System Works"):
                        st.markdown("""
                        This recommendation system uses several machine learning techniques:
                        
                        1. **Text Vectorization**: Converts review text into numerical features
                        2. **Topic Modeling**: Uses Non-negative Matrix Factorization (NMF) to discover key aspects in reviews
                        3. **Sentiment Analysis**: Analyzes sentiment for each discovered aspect
                        4. **Clustering**: Groups similar reviews to identify customer segments and their concerns
                        5. **Insight Generation**: Combines all analyses to create data-driven recommendations
                        
                        Unlike rule-based systems, this approach adapts to your specific business and discovers patterns in your unique customer feedback.
                        """)
                else:
                    st.warning(f"ML model error: {ml_recommendations.get('message', 'Unknown error')}")
                    st.info("Try increasing the number of reviews for better ML performance.")
        else:
            st.info(f"""
            ML recommendation system requires at least 20 reviews to work effectively.
            You currently have {len(df)} valid reviews. Try fetching more reviews to enable this feature.
            
            With more reviews, the system can:
            - Discover key themes in customer feedback
            - Analyze sentiment for each theme
            - Generate tailored, data-driven recommendations
            """)
    except Exception as e:
        st.warning(f"Error in ML recommendation system: {str(e)}")
        st.info("""
        Basic recommendations:
        1. Respond to negative reviews promptly and professionally
        2. Track sentiment trends monthly to measure improvement
        3. Implement a customer feedback system to catch issues early
        4. Train staff on customer service best practices
        """)
    
    # Download Results
    try:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("📎 Download Your Results")
        
        if "reviews_df" in st.session_state and st.session_state.reviews_df is not None:
            csv = convert_df_to_csv(df)
            st.download_button(
                label="📥 Download Reviews CSV",
                data=csv,
                file_name="review_analysis.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.warning(f"Error with download functionality: {str(e)}")

else:
    # Show a placeholder or example when no data is loaded
    if place_id:
        st.info("Click 'Fetch & Analyze Reviews' to start the analysis.")
    else:
        st.info("Enter a Google Place ID and click 'Fetch & Analyze Reviews' to start.")
        
        # Show an example of what the app does
        st.subheader("📱 App Features")
        st.markdown("""
        - 🤖 **Automated Review Analysis**: Get insights from hundreds of reviews in seconds
        - 📈 **Sentiment Timeline**: Track how sentiment changes over time
        - 🔍 **Common Words Analysis**: Discover what words customers mention most often
        - ☁️ **Word Clouds**: Visualize common words in positive and negative reviews
        - 📊 **Sentiment Analysis**: AI-powered sentiment detection using VADER
        - 💡 **Smart Recommendations**: Get ML-powered business advice based on customer feedback
        """)
