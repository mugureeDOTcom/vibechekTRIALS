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
    
    # Smart Recommendations
    try:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("🤖 Smart Recommendations")
        
        # Define common business issues and solutions based on keywords
        business_insights = {
            "service": {
                "positive": "Your service is praised by customers. Continue training staff on excellent customer service techniques.",
                "negative": "Service issues appear in negative reviews. Consider staff training or reviewing service protocols."
            },
            "price": {
                "positive": "Customers find your pricing reasonable and fair. Maintain this pricing strategy.",
                "negative": "Price concerns appear in negative reviews. Consider reviewing your pricing strategy or better communicating value."
            },
            "quality": {
                "positive": "Product/service quality is appreciated. Maintain your quality standards.",
                "negative": "Quality concerns appear in reviews. Review quality control processes."
            },
            "wait": {
                "positive": "Customers appreciate your efficient timing/waiting periods.",
                "negative": "Wait times appear to be an issue. Consider operational efficiency improvements."
            },
            "clean": {
                "positive": "Cleanliness is noted positively. Maintain your cleanliness standards.",
                "negative": "Cleanliness concerns appear in reviews. Review cleaning protocols."
            },
            "staff": {
                "positive": "Your staff receives positive mentions. Recognize and reward good employees.",
                "negative": "Staff-related concerns appear in reviews. Consider additional training or reviewing hiring practices."
            },
            "location": {
                "positive": "Your location is mentioned positively. Highlight this in marketing materials.",
                "negative": "Location issues appear in reviews. Consider improving signage, access, or parking if possible."
            },
            "food": {
                "positive": "Food quality is praised. Maintain your food preparation standards.",
                "negative": "Food quality issues appear in reviews. Review kitchen operations and quality control."
            },
            "atmosphere": {
                "positive": "Customers enjoy your atmosphere/ambiance. Maintain this environment.",
                "negative": "Atmosphere concerns appear in reviews. Consider refreshing your space's design or ambiance."
            },
            "parking": {
                "positive": "Parking is mentioned positively. Continue to maintain good parking options.",
                "negative": "Parking issues appear in reviews. Consider improving parking options or providing clearer instructions."
            },
            "menu": {
                "positive": "Your menu receives positive attention. Continue with your current menu strategy.",
                "negative": "Menu concerns appear in reviews. Consider updating options or improving descriptions."
            },
            "value": {
                "positive": "Customers find good value in your offerings. Maintain this balance of price and quality.",
                "negative": "Value concerns appear in reviews. Review pricing or improve quality to increase perceived value."
            },
            "friendly": {
                "positive": "Friendliness is mentioned positively. Continue encouraging friendly customer interactions.",
                "negative": "Consider emphasizing a more friendly approach to customer service."
            },
            "recommend": {
                "positive": "Customers are recommending your business - excellent! Consider a referral program.",
                "negative": "Work on issues that prevent customers from recommending your business."
            },
            "management": {
                "positive": "Management is mentioned positively. Maintain these management practices.",
                "negative": "Management concerns appear in reviews. Consider reviewing management training or approaches."
            },
            "reservation": {
                "positive": "Your reservation system works well for customers.",
                "negative": "Reservation issues appear in reviews. Review your reservation system or processes."
            },
            "bathroom": {
                "positive": "Restrooms are mentioned positively. Maintain cleanliness standards.",
                "negative": "Bathroom concerns appear in reviews. Improve cleaning frequency or facilities."
            },
            "noisy": {
                "positive": "Customers appreciate the sound levels in your establishment.",
                "negative": "Noise issues appear in reviews. Consider acoustic improvements or music volume adjustments."
            },
            "portion": {
                "positive": "Portion sizes are mentioned positively. Maintain current portion standards.",
                "negative": "Portion size concerns appear in reviews. Review your portion sizing strategy."
            },
            "delivery": {
                "positive": "Delivery service is praised by customers. Maintain delivery standards.",
                "negative": "Delivery issues appear in reviews. Review delivery processes and timing."
            },
            "return": {
                "positive": "Customers mention returning to your business - excellent repeat business!",
                "negative": "Consider addressing issues that prevent customers from returning."
            }
        }
        
        # Check for keywords in positive and negative reviews
        positive_insights = []
        negative_insights = []
        
        positive_text = " ".join(df[df["Sentiment"] == "Positive"]["Cleaned_Review"]).lower()
        negative_text = " ".join(df[df["Sentiment"] == "Negative"]["Cleaned_Review"]).lower()
        
        for keyword, insights in business_insights.items():
            if keyword in positive_text:
                positive_insights.append(insights["positive"])
            if keyword in negative_text:
                negative_insights.append(insights["negative"])
        
        # Add insights based on overall sentiment
        if len(df) > 0:
            sentiment_counts = df["Sentiment"].value_counts()
            positive_pct = sentiment_counts.get("Positive", 0) / len(df) * 100 if len(df) > 0 else 0
            negative_pct = sentiment_counts.get("Negative", 0) / len(df) * 100 if len(df) > 0 else 0
            
            if positive_pct > 75:
                positive_insights.append("Overall sentiment is very positive. Consider using positive reviews in marketing materials.")
            elif positive_pct < 50:
                negative_insights.append("Overall sentiment is concerning. Consider a comprehensive review of business operations.")
        
        # Add insights based on most common words
        pos_words = get_top_words(df[df["Sentiment"] == "Positive"]["Cleaned_Review"], 5)
        neg_words = get_top_words(df[df["Sentiment"] == "Negative"]["Cleaned_Review"], 5)
        
        if pos_words:
            top_pos_words = [word for word, count in pos_words]
            positive_insights.append(f"Customers often mention '{', '.join(top_pos_words)}' positively. Emphasize these aspects in marketing.")
            
        if neg_words:
            top_neg_words = [word for word, count in neg_words]
            negative_insights.append(f"Customers often mention '{', '.join(top_neg_words)}' negatively. Address these areas for improvement.")
        
        # Display recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🟢 Strengths to Maintain")
            if positive_insights:
                for i, insight in enumerate(positive_insights[:5], 1):  # Limit to top 5
                    st.markdown(f"{i}. {insight}")
            else:
                st.info("Not enough data to generate strength recommendations")
        
        with col2:
            st.markdown("### 🔴 Areas for Improvement")
            if negative_insights:
                for i, insight in enumerate(negative_insights[:5], 1):  # Limit to top 5
                    st.markdown(f"{i}. {insight}")
            else:
                st.info("Not enough data to generate improvement recommendations")
                
        # Add a section for actionable next steps
        if positive_insights or negative_insights:
            st.markdown('<div class="subsection-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📝 Actionable Next Steps")
            
            action_items = []
            
            if negative_insights:
                # Prioritize addressing top negative issues
                action_items.append("Address the top negative themes in customer reviews with specific improvement plans")
                
            if positive_insights:
                # Leverage strengths
                action_items.append("Highlight positive aspects in marketing materials and train staff to emphasize these strengths")
            
            # Add general best practices
            action_items.extend([
                "Respond to negative reviews promptly and professionally",
                "Track sentiment trends monthly to measure improvement",
                "Implement a customer feedback system to catch issues before they result in negative reviews",
                "Train staff on common customer pain points identified in the analysis"
            ])
            
            for i, action in enumerate(action_items, 1):
                st.markdown(f"{i}. {action}")
    except Exception as e:
        st.warning(f"Error generating recommendations: {str(e)}")
    
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
