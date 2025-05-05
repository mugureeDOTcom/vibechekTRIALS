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
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

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

@st.cache_resource
def load_nlp_models():
    """Load and cache the HuggingFace models for sentiment analysis and summarization"""
    try:
        # Load sentiment analysis model
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            return_all_scores=True
        )
        
        # Load summarization model
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            max_length=100,
            min_length=30,
            do_sample=False
        )
        
        return sentiment_model, summarizer
    except Exception as e:
        st.error(f"Error loading NLP models: {str(e)}")
        return None, None

def ml_sentiment_analysis(text, sentiment_model):
    """Perform sentiment analysis using the Hugging Face model"""
    if not text or len(text) < 10:
        return "Neutral"
    
    try:
        # Get model prediction
        result = sentiment_model(text[:512])[0]  # Limit text length to avoid token limit errors
        
        # Extract scores - model returns [{'label': 'NEGATIVE', 'score': 0.xxx}, {'label': 'POSITIVE', 'score': 0.xxx}]
        scores = {item['label']: item['score'] for item in result}
        
        # Determine sentiment based on confidence scores
        if 'POSITIVE' in scores and scores['POSITIVE'] > 0.6:
            return "Positive"
        elif 'NEGATIVE' in scores and scores['NEGATIVE'] > 0.6:
            return "Negative"
        else:
            return "Neutral"
    except Exception as e:
        st.warning(f"Sentiment model error: {str(e)}")
        return "Neutral"

def summarize_reviews(reviews, summarizer, sentiment_type="all"):
    """Summarize a collection of reviews using the Hugging Face summarization model"""
    if not reviews or len(reviews) < 3:  # Need at least a few reviews to summarize
        return f"Not enough {sentiment_type.lower()} reviews to generate insights."
    
    # Combine reviews into one text (limit to avoid token limits)
    combined_text = " ".join(reviews[:50])
    
    # Ensure the text is substantial enough
    if len(combined_text) < 100:
        return f"Not enough {sentiment_type.lower()} review content to generate insights."
    
    try:
        # Generate summary
        summary = summarizer(combined_text, 
                           max_length=100,
                           min_length=30,
                           do_sample=False)
        
        # Extract the summary text and capitalize the first letter
        summary_text = summary[0]['summary_text']
        summary_text = summary_text[0].upper() + summary_text[1:] if summary_text else ""
        
        return summary_text
    except Exception as e:
        st.warning(f"Summarization error: {str(e)}")
        return f"Unable to generate {sentiment_type.lower()} review summary due to an error."

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
    
    # Smart Recommendations with ML models
    try:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("🤖 AI-Powered Business Insights")
        
        # Load the models
        sentiment_model, summarizer = load_nlp_models()
        
        if sentiment_model is None or summarizer is None:
            st.error("Could not load NLP models. Please check your internet connection and try again.")
            st.stop()
        
        # Apply ML-based sentiment analysis if we have enough reviews
        if len(df) > 5:
            with st.spinner("Generating AI insights..."):
                # Use batch processing for efficiency (optional)
                
                # If you want to refresh sentiment with the ML model:
                sample_size = min(100, len(df))  # Limit processing to 100 reviews for performance
                sampled_df = df.sample(sample_size) if sample_size < len(df) else df
                sampled_df["ML_Sentiment"] = sampled_df["Cleaned_Review"].apply(
                    lambda x: ml_sentiment_analysis(x, sentiment_model)
                )
                df = df.copy()
                df.loc[sampled_df.index, "Sentiment"] = sampled_df["ML_Sentiment"]
                
                # Extract positive and negative reviews for summarization
                positive_reviews = df[df["Sentiment"] == "Positive"]["snippet"].tolist()
                negative_reviews = df[df["Sentiment"] == "Negative"]["snippet"].tolist()
                
                # Get summaries
                positive_summary = summarize_reviews(positive_reviews, summarizer, "positive")
                negative_summary = summarize_reviews(negative_reviews, summarizer, "negative")
                
                # Display summaries
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🟢 Strengths & Positive Feedback")
                    if len(positive_reviews) > 3:
                        st.markdown(f"**AI Summary:** {positive_summary}")
                        
                        # Add some context about what customers like
                        pos_words = get_top_words(df[df["Sentiment"] == "Positive"]["Cleaned_Review"], 5)
                        if pos_words:
                            top_pos_words = [word for word, count in pos_words]
                            st.markdown(f"**Top positive mentions:** {', '.join(top_pos_words)}")
                    else:
                        st.info("Not enough positive reviews to generate AI insights")
                
                with col2:
                    st.markdown("### 🔴 Areas for Improvement")
                    if len(negative_reviews) > 3:
                        st.markdown(f"**AI Summary:** {negative_summary}")
                        
                        # Add context about customer pain points
                        neg_words = get_top_words(df[df["Sentiment"] == "Negative"]["Cleaned_Review"], 5)
                        if neg_words:
                            top_neg_words = [word for word, count in neg_words]
                            st.markdown(f"**Top negative mentions:** {', '.join(top_neg_words)}")
                    else:
                        st.info("Not enough negative reviews to generate AI insights")
                
                # Add actionable recommendations based on the summaries
                st.markdown('<div class="subsection-divider"></div>', unsafe_allow_html=True)
                st.markdown("### 📋 Strategic Recommendations")
                
                # Get sentiment distribution
                sentiment_counts = df["Sentiment"].value_counts()
                total_reviews = len(df)
                positive_pct = sentiment_counts.get("Positive", 0) / total_reviews * 100 if total_reviews > 0 else 0
                negative_pct = sentiment_counts.get("Negative", 0) / total_reviews * 100 if total_reviews > 0 else 0
                neutral_pct = sentiment_counts.get("Neutral", 0) / total_reviews * 100 if total_reviews > 0 else 0
                
                # Create a recommendations framework based on sentiment distribution
                recommendations = []
                
                # Overall sentiment health
                if positive_pct >= 70:
                    recommendations.append("**Overall Brand Health:** Your business has a strong positive reputation. Focus on maintaining quality while addressing specific negative feedback areas.")
                elif positive_pct >= 50:
                    recommendations.append("**Overall Brand Health:** Your business has a moderately positive reputation. Address key negative themes while reinforcing positive attributes.")
                else:
                    recommendations.append("**Overall Brand Health:** Your business needs significant improvement in customer satisfaction. Prioritize addressing the negative feedback themes immediately.")
                
                # Response strategy
                if negative_pct > 20:
                    recommendations.append("**Response Strategy:** Implement a proactive review response protocol for all negative reviews within 24-48 hours.")
                else:
                    recommendations.append("**Response Strategy:** Continue responding to selected reviews, focusing on the most detailed feedback.")
                
                # Marketing recommendations
                if positive_pct >= 60:
                    recommendations.append("**Marketing Opportunity:** Leverage positive reviews in marketing materials and consider implementing a customer testimonial program.")
                
                # Operational focus
                if len(negative_reviews) >= 5:
                    recommendations.append(f"**Operational Focus:** Based on negative feedback patterns, prioritize improvements in: {', '.join(top_neg_words[:3]) if 'top_neg_words' in locals() else 'customer experience areas'}.")
                
                # Review acquisition strategy
                if total_reviews < 50:
                    recommendations.append("**Review Acquisition:** Implement a systematic review request process to gather more customer feedback.")
                
                # Display recommendations
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {rec}")
        else:
            st.info("Need more reviews to generate meaningful AI insights. Try fetching more reviews.")

    except Exception as e:
        st.warning(f"Error generating AI insights: {str(e)}")
        # Fallback to simpler analysis if AI models fail
        st.markdown("Falling back to basic analysis. The AI models could not be loaded.")
    
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
        - 💡 **Smart Recommendations**: Get actionable business advice based on customer feedback
        """)
