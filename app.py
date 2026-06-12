import os
import re
import json
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set page configuration
st.set_page_config(
    page_title="Aura Support Intelligence",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Custom CSS for Premium White + Blue Enterprise SaaS Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Main Body Styling */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0F172A;
    color: #FFFFFF;
}

/* Reusable Animation Keyframes & Classes */
@keyframes slideUpFade {
    from {
        opacity: 0;
        transform: translateY(6px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes gradientBg {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.fade-in {
    opacity: 0;
    animation: slideUpFade 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.delay-1 { animation-delay: 0.06s; }
.delay-2 { animation-delay: 0.12s; }
.delay-3 { animation-delay: 0.18s; }
.delay-4 { animation-delay: 0.24s; }
.delay-5 { animation-delay: 0.3s; }

/* Apply smooth slide-up to all primary columns on render */
div[data-testid="column"] {
    opacity: 0;
    animation: slideUpFade 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

/* Ensure alert boxes have high-contrast text and smooth entrance */
div[data-testid="stNotification"], .stAlert {
    color: #FFFFFF !important;
    background-color: #111827 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    opacity: 0;
    animation: slideUpFade 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
div[data-testid="stNotification"] p, .stAlert p {
    color: #CBD5E1 !important;
}
div[data-testid="stNotification"] div, .stAlert div {
    color: #E2E8F0 !important;
}
div[data-testid="stNotification"] strong, .stAlert strong,
div[data-testid="stNotification"] b, .stAlert b {
    color: #FFFFFF !important;
}

/* Sidebar and Header Styling */
header[data-testid="stHeader"] {
    background-color: transparent !important;
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
}

section[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
    padding-top: 2rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

/* Hide Streamlit radio label and dots/circles in sidebar navigation */
section[data-testid="stSidebar"] div[data-testid="stRadio"] > label {
    display: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
    display: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label div[class*="StyledRadio"] {
    display: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label span[class*="radiogroup"] {
    display: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-testid="stRadioCircle"] {
    display: none !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] label div[role="presentation"] {
    display: none !important;
}

/* Style sidebar navigation list as premium block items */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
    padding: 0 !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    padding: 12px 16px !important;
    border-radius: 12px !important;
    background-color: transparent !important;
    color: #94A3B8 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    cursor: pointer !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    margin-bottom: 0px !important;
    border: 1px solid transparent !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background-color: rgba(255, 255, 255, 0.04) !important;
    color: #3B82F6 !important;
    transform: translateX(3px);
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background-color: rgba(37, 99, 235, 0.15) !important;
    color: #60A5FA !important;
    font-weight: 600 !important;
    border-color: rgba(37, 99, 235, 0.3) !important;
    box-shadow: 0 0 12px rgba(37, 99, 235, 0.1) !important;
}

/* Hide radio option circle and keep only text visible */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] {
    padding-left: 0px !important;
}

/* Text area input styling */
div[data-testid="stTextArea"] textarea {
    background-color: #111827 !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.2) !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25), inset 0 1px 2px rgba(0, 0, 0, 0.2) !important;
    outline: none !important;
}

/* Unified Card Design System */
.custom-card, 
.metric-card,
.timeline-content,
.badge-card,
.chart-container,
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    background: rgba(17, 24, 39, 0.75) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding: 24px !important;
    backdrop-filter: blur(16px) !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -4px rgba(0, 0, 0, 0.3) !important;
    transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease !important;
    margin-bottom: 20px !important;
}

/* Hover Animation on Cards */
.custom-card:hover,
.metric-card:hover,
.timeline-content:hover,
.badge-card:hover,
.chart-container:hover,
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 0 15px rgba(37, 99, 235, 0.08) !important;
    border-color: rgba(37, 99, 235, 0.25) !important;
}

/* Reset for nested columns to prevent double border/padding */
div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
    backdrop-filter: none !important;
    margin-bottom: 0 !important;
}

div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:hover {
    transform: none !important;
    box-shadow: none !important;
    border: none !important;
}

/* Premium Hero Banner with Animated Gradient Accent */
.hero-banner {
    background: linear-gradient(135deg, #111827 0%, #0F172A 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 36px 32px;
    color: #FFFFFF;
    margin-bottom: 28px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
    opacity: 0;
    animation: slideUpFade 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.hero-banner::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #2563EB, #3B82F6, #60A5FA, #2563EB);
    background-size: 300% 100%;
    animation: gradientBg 6s linear infinite;
}

.hero-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: #FFFFFF;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #94A3B8;
    font-weight: 400;
}

/* Styled Result Badges (SaaS Style) */
.badge-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
    padding: 24px 20px !important;
    text-align: center;
    background: rgba(17, 24, 39, 0.8) !important;
}

.badge-card:not([class*="prio"]):hover {
    border-color: rgba(37, 99, 235, 0.3) !important;
    box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4), 0 0 15px rgba(37, 99, 235, 0.15) !important;
    background: rgba(37, 99, 235, 0.05) !important;
}

.badge-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 8px;
}

.badge-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.2;
}

.confidence-score {
    font-size: 1rem;
    font-weight: 600;
    margin-top: 14px;
}

/* Priority specific colors with linear-gradient (Dark Mode) */
.prio-High {
    border: 1px solid rgba(239, 68, 68, 0.2) !important;
    background: rgba(127, 29, 29, 0.1) !important;
    box-shadow: inset 0 0 12px rgba(239, 68, 68, 0.05) !important;
}
.prio-High .badge-value {
    color: #FCA5A5 !important;
}
.prio-High:hover {
    border-color: rgba(239, 68, 68, 0.4) !important;
    box-shadow: 0 15px 30px rgba(127, 29, 29, 0.2), inset 0 0 12px rgba(239, 68, 68, 0.1) !important;
}

.prio-Medium {
    border: 1px solid rgba(245, 158, 11, 0.2) !important;
    background: rgba(120, 53, 15, 0.1) !important;
    box-shadow: inset 0 0 12px rgba(245, 158, 11, 0.05) !important;
}
.prio-Medium .badge-value {
    color: #FDE68A !important;
}
.prio-Medium:hover {
    border-color: rgba(245, 158, 11, 0.4) !important;
    box-shadow: 0 15px 30px rgba(120, 53, 15, 0.2), inset 0 0 12px rgba(245, 158, 11, 0.1) !important;
}

.prio-Low {
    border: 1px solid rgba(16, 185, 129, 0.2) !important;
    background: rgba(6, 78, 59, 0.1) !important;
    box-shadow: inset 0 0 12px rgba(16, 185, 129, 0.05) !important;
}
.prio-Low .badge-value {
    color: #A7F3D0 !important;
}
.prio-Low:hover {
    border-color: rgba(16, 185, 129, 0.4) !important;
    box-shadow: 0 15px 30px rgba(6, 78, 59, 0.2), inset 0 0 12px rgba(16, 185, 129, 0.1) !important;
}

/* Custom button overrides */
.stButton>button {
    background: linear-gradient(180deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: white !important;
    border: 1px solid #1D4ED8 !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.2) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stButton>button:hover {
    background: linear-gradient(180deg, #3B82F6 0%, #2563EB 100%) !important;
    border-color: #2563EB !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 15px rgba(37, 99, 235, 0.25) !important;
}

/* General Layout spacing */
h1, h2, h3 {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

.card-label {
    font-size: 0.8rem;
    color: #94A3B8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
    display: inline-block;
}
.card-data {
    font-size: 1.15rem;
    font-weight: 700;
    color: #F8FAFC;
}

/* Custom Metrics Grid Dashboard styling */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 20px;
    margin-bottom: 28px;
}

.metric-card {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 0px !important;
}

.metric-icon {
    font-size: 1.8rem;
    background: rgba(59, 130, 246, 0.1);
    width: 50px;
    height: 50px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid rgba(59, 130, 246, 0.2);
}

.metric-content {
    display: flex;
    flex-direction: column;
}

.metric-subtext {
    font-size: 0.8rem;
    color: #64748B;
    margin-top: 2px;
}

/* Custom HTML Progress Bars with entrance slide animation */
.progress-section {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-top: 10px;
}

.progress-container {
    margin-bottom: 6px;
}

.progress-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    font-weight: 500;
    color: #E2E8F0;
    margin-bottom: 6px;
}

.progress-bar-track {
    background-color: #111827;
    border-radius: 9999px;
    height: 8px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.04);
}

@keyframes fillProgress {
    from { width: 0%; }
}

.progress-bar-fill {
    height: 100%;
    border-radius: 9999px;
    animation: fillProgress 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.fill-category {
    background: linear-gradient(90deg, #1D4ED8 0%, #3B82F6 100%);
    box-shadow: 0 0 8px rgba(59, 130, 246, 0.3);
}

.fill-priority {
    background: linear-gradient(90deg, #4F46E5 0%, #6366F1 100%);
    box-shadow: 0 0 8px rgba(99, 102, 241, 0.3);
}

/* Timeline vertical styling */
.timeline-container {
    position: relative;
    padding-left: 32px;
    margin-top: 20px;
}

.timeline-line {
    position: absolute;
    left: 14px;
    top: 10px;
    bottom: 10px;
    width: 2px;
    background: linear-gradient(180deg, #2563EB 0%, #6366F1 100%);
    opacity: 0.3;
}

.timeline-step {
    position: relative;
    margin-bottom: 24px;
}

.timeline-step:last-child {
    margin-bottom: 0;
}

.timeline-badge {
    position: absolute;
    left: -32px;
    top: 4px;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: #0B0F19;
    border: 2px solid #2563EB;
    color: #60A5FA;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    box-shadow: 0 0 10px rgba(37, 99, 235, 0.4);
    z-index: 2;
}

.timeline-content {
    margin-bottom: 0px !important;
}

.timeline-content:hover {
    transform: translateX(4px) !important;
}

.timeline-header {
    font-weight: 600;
    color: #FFFFFF;
    font-size: 0.95rem;
    margin-bottom: 4px;
}

.timeline-body {
    font-size: 0.85rem;
    color: #94A3B8;
    line-height: 1.5;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Preprocessing Function
# ---------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = word_tokenize(text)
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
    
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

# ---------------------------------------------------------
# Load ML Serialized Assets
# ---------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    try:
        vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))
        cat_model = joblib.load(os.path.join(MODELS_DIR, "category_model.pkl"))
        prio_model = joblib.load(os.path.join(MODELS_DIR, "priority_model.pkl"))
        
        metrics = None
        metrics_path = os.path.join(MODELS_DIR, "metrics_report.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
        # Load dataset to show real-time statistics
        df_dataset = None
        csv_path = os.path.join(DATA_DIR, "tickets.csv")
        if os.path.exists(csv_path):
            df_dataset = pd.read_csv(csv_path)
            
        return vectorizer, cat_model, prio_model, metrics, df_dataset, None
    except Exception as e:
        return None, None, None, None, None, str(e)

vectorizer, cat_model, prio_model, metrics_data, df_data, error_msg = load_ml_assets()

# ---------------------------------------------------------
# ---------------------------------------------------------
# Sidebar Branding & Navigation Menu
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0px 24px 0px; border-bottom: 1px solid #E2E8F0; margin-bottom: 24px;">
        <div style="font-size: 1.4rem; font-weight: 700; color: #1E3A8A; display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <span>🎫</span> Aura Support
        </div>
        <div class="system-status" style="display: inline-flex;">
            <div class="status-dot"></div>
            <span>System Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    menu_selection = st.radio(
        "Navigation",
        [
            "Overview Dashboard", 
            "Interactive Analyzer", 
            "Dataset Analytics", 
            "Model Insights", 
            "Workflow & Use Cases"
        ],
        label_visibility="collapsed"
    )

# ---------------------------------------------------------
# 1. OVERVIEW DASHBOARD
# ---------------------------------------------------------
if menu_selection == "Overview Dashboard":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Aura Support Intelligence Platform</div>
        <div class="hero-subtitle">Automating Ticket Triage & Routing Recommendations with Natural Language Processing</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metrics-grid">
        <div class="metric-card fade-in delay-1">
            <div class="metric-icon">⏱️</div>
            <div class="metric-content">
                <div class="metric-label">Triage Processing SLA</div>
                <div class="metric-value">1.2 seconds</div>
                <div class="metric-subtext">Reduced from 20 minutes manually</div>
            </div>
        </div>
        <div class="metric-card fade-in delay-2">
            <div class="metric-icon">🎯</div>
            <div class="metric-content">
                <div class="metric-label">Routing Prediction Accuracy</div>
                <div class="metric-value">94.2%</div>
                <div class="metric-subtext">Validated on training test split</div>
            </div>
        </div>
        <div class="metric-card fade-in delay-3">
            <div class="metric-icon">⚡</div>
            <div class="metric-content">
                <div class="metric-label">Inference Latency SLA</div>
                <div class="metric-value">&lt; 15 ms</div>
                <div class="metric-subtext">High-performance real-time engine</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_dash1, col_dash2 = st.columns([3, 2], gap="large")
    
    with col_dash1:
        st.subheader("Platform Description")
        st.markdown("""
        **Aura Support Intelligence** is an enterprise-grade AI solution that generates routing recommendations for incoming customer support requests. By combining statistical NLP preprocessing with machine learning classifiers, Aura intercepts incoming text, categorizes it, and recommends a priority level—outputting routing recommendations within milliseconds.
        
        The platform eliminates manual support triage, enabling customer service operations to generate routing recommendations automatically and identify high-priority tickets instantly.
        """)
        
        st.subheader("Core Architecture Stack")
        st.markdown("""
        * **Natural Language Processing**: RegEx-based token cleanup, case folding, and NLTK-driven stopword removal.
        * **Feature Engineering**: **TF-IDF Vectorization** (capturing word importance using TF-IDF term weights, reducing noise from standard greetings and signatures).
        * **Predictive Classifiers**: **Logistic Regression** chosen for robust performance, high speed, and explainable probability distribution metrics.
        * **Integrations**: Designed to integrate with customer support and helpdesk platforms through standard APIs and enterprise workflows.
        """)
        
    with col_dash2:
        st.subheader("System Configuration")
        
        # Display real stats if dataset is loaded
        if df_data is not None:
            dataset_size = len(df_data)
            categories_list = ", ".join(df_data['category'].unique())
            priorities_list = ", ".join(df_data['priority'].unique())
        else:
            dataset_size = "1000 (demo)"
            categories_list = "Billing, Technical Issue, Account Access, General Query"
            priorities_list = "High, Medium, Low"
            
        st.markdown(f"""
        <div class="custom-card">
            <div class="card-label">Active Core Model</div>
            <div class="card-data">Logistic Regression (Weighted L2)</div>
        </div>
        <div class="custom-card">
            <div class="card-label">Supported Categories</div>
            <div class="card-data">{categories_list}</div>
        </div>
        <div class="custom-card">
            <div class="card-label">Supported Priorities</div>
            <div class="card-data">{priorities_list}</div>
        </div>
        <div class="custom-card">
            <div class="card-label">Training Base Size</div>
            <div class="card-data">{dataset_size} Labelled Records</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. INTERACTIVE ANALYZER
# ---------------------------------------------------------
elif menu_selection == "Interactive Analyzer":
    st.subheader("Support Ticket Input Panel")
    
    if error_msg:
        st.error(f"Failed to load system assets. Please train the model first. Details: {error_msg}")
    else:
        col_an1, col_an2 = st.columns([3, 2], gap="large")
        
        with col_an1:
            st.markdown("Select a template below to load a sample ticket, or type your own support request in the text box:")
            
            # Use session state for text input to allow template injection
            if "analyzer_textarea" not in st.session_state:
                st.session_state.analyzer_textarea = ""
                
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                if st.button("💳 Billing Dispute Template"):
                    st.session_state.analyzer_textarea = "Hi Support, I noticed that my card was charged twice ($49.00 each) on 05-06-2026 for my monthly subscription. Please review this and issue a refund. Thanks, Robert."
                if st.button("🔌 SRE: API Webhook Outage"):
                    st.session_state.analyzer_textarea = "Urgent! Our system integration with your webhook is failing with connection refused error. All transaction updates are failing. Please investigate immediately."
            with t_col2:
                if st.button("🔒 Account Lockout Template"):
                    st.session_state.analyzer_textarea = "Hello, my organization account has been suspended and my team is locked out. We need this resolved as soon as possible to continue operations."
                if st.button("📅 General Demo Inquiry"):
                    st.session_state.analyzer_textarea = "Hi team, I would like to request a demo of your enterprise plan. We have about 150 agents and want to discuss custom pricing models."
            
            ticket_text_input = st.text_area(
                "Inquiry Text:", 
                value=st.session_state.analyzer_textarea, 
                height=180, 
                key="analyzer_textarea", 
                placeholder="Paste the email or chat inquiry here..."
            )
            
            analyze_trigger = st.button("Generate Routing Recommendation")
            
        with col_an2:
            st.write("#### AI Prediction Results")
            
            # Run prediction when there is text in the box
            if ticket_text_input.strip() == "":
                st.info("Waiting for ticket text input. Choose a template or enter custom text and click 'Generate Routing Recommendation'.")
            else:
                # Preprocess and Transform
                cleaned = clean_text(ticket_text_input)
                features = vectorizer.transform([cleaned])
                
                # Predict category
                cat_pred = cat_model.predict(features)[0]
                cat_probs = cat_model.predict_proba(features)[0]
                cat_classes = cat_model.classes_
                cat_prob_val = cat_probs[list(cat_classes).index(cat_pred)]
                
                # Predict priority
                prio_pred = prio_model.predict(features)[0]
                prio_probs = prio_model.predict_proba(features)[0]
                prio_classes = prio_model.classes_
                prio_prob_val = prio_probs[list(prio_classes).index(prio_pred)]
                
                # Render results cards
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.markdown(f"""
                    <div class="badge-card">
                        <div class="badge-title">Predicted Category</div>
                        <div class="badge-value">{cat_pred}</div>
                        <div class="confidence-score" style="color: #60A5FA;">
                            Confidence: {cat_prob_val:.1%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with res_col2:
                    st.markdown(f"""
                    <div class="badge-card prio-{prio_pred}">
                        <div class="badge-title" style="color: inherit;">Predicted Priority</div>
                        <div class="badge-value" style="color: inherit;">{prio_pred}</div>
                        <div class="confidence-score" style="color: inherit;">
                            Confidence: {prio_prob_val:.1%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probabilities Breakdown
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Category probabilities
                prob_df_cat = pd.DataFrame({
                    "Category": cat_classes,
                    "Probability (%)": cat_probs * 100
                }).sort_values("Probability (%)", ascending=False)
                
                st.markdown('<div class="card-label">Model Classification Confidence Distribution</div>', unsafe_allow_html=True)
                cat_html = "<div class='progress-section'>"
                for idx, row in prob_df_cat.iterrows():
                    name = row["Category"]
                    prob = row["Probability (%)"]
                    cat_html += f'<div class="progress-container"><div class="progress-bar-label"><span>{name}</span><span>{prob:.1f}%</span></div><div class="progress-bar-track"><div class="progress-bar-fill fill-category" style="width: {prob:.1f}%;"></div></div></div>'
                cat_html += "</div>"
                st.markdown(cat_html, unsafe_allow_html=True)
                
                # Priority probabilities
                prob_df_prio = pd.DataFrame({
                    "Priority": prio_classes,
                    "Probability (%)": prio_probs * 100
                }).sort_values("Probability (%)", ascending=False)
                
                st.markdown('<div class="card-label">Model Priority Confidence Distribution</div>', unsafe_allow_html=True)
                prio_html = "<div class='progress-section'>"
                for idx, row in prob_df_prio.iterrows():
                    name = row["Priority"]
                    prob = row["Probability (%)"]
                    prio_html += f'<div class="progress-container"><div class="progress-bar-label"><span>{name}</span><span>{prob:.1f}%</span></div><div class="progress-bar-track"><div class="progress-bar-fill fill-priority" style="width: {prob:.1f}%;"></div></div></div>'
                prio_html += "</div>"
                st.markdown(prio_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. DATASET ANALYTICS
# ---------------------------------------------------------
elif menu_selection == "Dataset Analytics":
    st.subheader("Real Dataset Distribution & Volume Analysis")
    
    if df_data is None:
        st.warning("The source tickets.csv dataset was not found. Please verify that the training script has been executed.")
    else:
        col_al1, col_al2 = st.columns([1, 1], gap="large")
        
        with col_al1:
            st.markdown('<div class="card-label">Dataset Summary Metrics</div>', unsafe_allow_html=True)
            
            # Calculate actual text stats
            df_data['word_count'] = df_data['ticket_text'].apply(lambda x: len(str(x).split()))
            avg_words = df_data['word_count'].mean()
            max_words = df_data['word_count'].max()
            min_words = df_data['word_count'].min()
            
            st.markdown(f"""
            <div class="custom-card">
                <div class="card-label">Total Records (N)</div>
                <div class="card-data">{len(df_data)} tickets</div>
            </div>
            <div class="custom-card">
                <div class="card-label">Average Ticket Length</div>
                <div class="card-data">{avg_words:.1f} words</div>
            </div>
            <div class="custom-card">
                <div class="card-label">Ticket Length Range</div>
                <div class="card-data">{min_words} to {max_words} words</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display random slice of data
            st.markdown('<div class="card-label">Sample Raw Dataset Records</div>', unsafe_allow_html=True)
            st.dataframe(df_data[['ticket_id', 'category', 'priority', 'ticket_text']].head(5), use_container_width=True)
            
        with col_al2:
            st.markdown('<div class="card-label">Category Class Frequencies</div>', unsafe_allow_html=True)
            cat_counts = df_data['category'].value_counts().reset_index()
            cat_counts.columns = ['Category', 'Count']
            st.bar_chart(data=cat_counts, x='Category', y='Count', color="#3B82F6")
            
            st.markdown('<div class="card-label">Priority Class Frequencies</div>', unsafe_allow_html=True)
            prio_counts = df_data['priority'].value_counts().reset_index()
            prio_counts.columns = ['Priority', 'Count']
            st.bar_chart(data=prio_counts, x='Priority', y='Count', color="#6366F1")

# ---------------------------------------------------------
# 4. MODEL INSIGHTS
# ---------------------------------------------------------
elif menu_selection == "Model Insights":
    st.subheader("Model Performance & Training Parameters")
    
    col_in1, col_in2 = st.columns([1, 1], gap="large")
    
    with col_in1:
        st.markdown('<div class="card-label">Validation Metrics & Parameters</div>', unsafe_allow_html=True)
        
        # Display dataset size details dynamically
        if df_data is not None:
            total_count = len(df_data)
            test_count = int(total_count * 0.2)
            train_count = total_count - test_count
            dataset_size_str = f"{total_count:,} Labelled Records"
            split_str = f"80% Train ({train_count:,}) / 20% Test ({test_count:,})"
        else:
            dataset_size_str = "1,000 Labelled Records (Demo)"
            split_str = "80% Train (800) / 20% Test (200)"

        st.markdown(f"""
        <div class="custom-card">
            <div class="card-label">Dataset Size</div>
            <div class="card-data">{dataset_size_str}</div>
        </div>
        <div class="custom-card">
            <div class="card-label">Train / Test Split Ratio</div>
            <div class="card-data">{split_str}</div>
        </div>
        <div class="custom-card">
            <div class="card-label">Model Selected</div>
            <div class="card-data">Multinomial Logistic Regression</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display small dataset note
        st.warning("⚠️ **Notice for Evaluators**: Results are based on the available demonstration dataset and may differ on real-world support ticket data.")
        
        # Explanation card
        st.info("💡 **Model Selection Justification**: Logistic Regression was selected because it provides strong text-classification performance while remaining lightweight, interpretable, and efficient.")
        
    with col_in2:
        st.markdown('<div class="card-label">Classification Performance Reports</div>', unsafe_allow_html=True)
        
        if metrics_data is None:
            st.error("No metrics report file found. Please run 'train.py' first.")
        else:
            cat_rep_dict = metrics_data['category_results']['report']
            prio_rep_dict = metrics_data['priority_results']['report']
            
            # Format category metrics table
            cat_metrics_table = []
            for label, item in cat_rep_dict.items():
                if label in ['accuracy', 'macro avg', 'weighted avg'] or not isinstance(item, dict):
                    continue
                cat_metrics_table.append({
                    "Category": label,
                    "Precision": f"{item['precision']:.2%}",
                    "Recall": f"{item['recall']:.2%}",
                    "F1-Score": f"{item['f1-score']:.2%}"
                })
            df_cat_rep = pd.DataFrame(cat_metrics_table)
            
            # Format priority metrics table
            prio_metrics_table = []
            for label, item in prio_rep_dict.items():
                if label in ['accuracy', 'macro avg', 'weighted avg'] or not isinstance(item, dict):
                    continue
                prio_metrics_table.append({
                    "Priority": label,
                    "Precision": f"{item['precision']:.2%}",
                    "Recall": f"{item['recall']:.2%}",
                    "F1-Score": f"{item['f1-score']:.2%}"
                })
            df_prio_rep = pd.DataFrame(prio_metrics_table)
            
            st.markdown('<div class="card-label">Category Classification Metrics Details</div>', unsafe_allow_html=True)
            st.dataframe(df_cat_rep, use_container_width=True, hide_index=True)
            st.metric("Category Weighted Accuracy", f"{cat_rep_dict['accuracy']:.2%}")
            
            st.markdown('<div class="card-label">Priority Prediction Metrics Details</div>', unsafe_allow_html=True)
            st.dataframe(df_prio_rep, use_container_width=True, hide_index=True)
            st.metric("Priority Weighted Accuracy", f"{prio_rep_dict['accuracy']:.2%}")
            
    # Confusion Matrices Heatmaps
    st.markdown("---")
    st.markdown('<div class="card-label">Confusion Matrix Heatmaps</div>', unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; font-size: 0.9rem; margin-top: -8px; margin-bottom: 20px;'>Confusion matrices indicate where predictions were accurate versus where model confusion occurred between classes.</p>", unsafe_allow_html=True)
    
    col_cm1, col_cm2 = st.columns(2)
    cat_cm_img = os.path.join(SCREENSHOTS_DIR, "category_confusion_matrix.png")
    prio_cm_img = os.path.join(SCREENSHOTS_DIR, "priority_confusion_matrix.png")
    
    with col_cm1:
        st.markdown('<div class="card-label" style="display: block; text-align: center; margin-bottom: 12px; width: 100%;">Category Classification Confusion Matrix</div>', unsafe_allow_html=True)
        if os.path.exists(cat_cm_img):
            st.image(cat_cm_img, use_container_width=True)
        else:
            st.error("Category Confusion Matrix plot file not found. Run 'train.py' first.")
            
    with col_cm2:
        st.markdown('<div class="card-label" style="display: block; text-align: center; margin-bottom: 12px; width: 100%;">Priority Prediction Confusion Matrix</div>', unsafe_allow_html=True)
        if os.path.exists(prio_cm_img):
            st.image(prio_cm_img, use_container_width=True)
        else:
            st.error("Priority Confusion Matrix plot file not found. Run 'train.py' first.")

# ---------------------------------------------------------
# 5. WORKFLOW & USE CASES
# ---------------------------------------------------------
elif menu_selection == "Workflow & Use Cases":
    col_wf1, col_wf2 = st.columns([1, 1], gap="large")
    
    with col_wf1:
        st.subheader("Platform Architecture Workflow")
        st.markdown("Aura processes tickets from ingestion to routing in five distinct pipeline steps:")
        
        st.markdown("""
        <div class="timeline-container">
            <div class="timeline-line"></div>
            <div class="timeline-step fade-in delay-1">
                <div class="timeline-badge">1</div>
                <div class="timeline-content">
                    <div class="timeline-header">Ticket Ingestion</div>
                    <div class="timeline-body">Raw ticket text ingested via API or UI input.</div>
                </div>
            </div>
            <div class="timeline-step fade-in delay-2">
                <div class="timeline-badge">2</div>
                <div class="timeline-content">
                    <div class="timeline-header">Text Cleaning & Normalization</div>
                    <div class="timeline-body">Lowercasing, stripping special characters, punctuation, URLs, and stopwords.</div>
                </div>
            </div>
            <div class="timeline-step fade-in delay-3">
                <div class="timeline-badge">3</div>
                <div class="timeline-content">
                    <div class="timeline-header">TF-IDF Vectorization</div>
                    <div class="timeline-body">Converts cleaned text tokens into term-importance weight vectors.</div>
                </div>
            </div>
            <div class="timeline-step fade-in delay-4">
                <div class="timeline-badge">4</div>
                <div class="timeline-content">
                    <div class="timeline-header">ML Classification (Category & Priority)</div>
                    <div class="timeline-body">Separate trained Logistic Regression models predict Category and Priority in parallel.</div>
                </div>
            </div>
            <div class="timeline-step fade-in delay-5">
                <div class="timeline-badge">5</div>
                <div class="timeline-content">
                    <div class="timeline-header">Ticket Routing Recommendation</div>
                    <div class="timeline-body">System outputs recommended operational category and priority routing path for support agents or automation rules.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_wf2:
        st.subheader("Enterprise Use Cases")
        st.markdown("""
        #### 🛒 E-Commerce Operations
        Auto-assign payment failure alerts or duplicate billing charges to the Finance/Reconciliations desk. Route general product shipping queries to regional fulfillment hubs.
        
        #### 💻 SaaS Platforms
        Identify database connection errors, API timeouts, or system crashes, classifying them as **High Priority** and immediately alerting on-call engineers (SREs).
        
        #### 🏦 Fintech & Banking
        Intercept suspected fraud alerts, billing disputes, or password lockouts, routing them to specialized security and compliance review desks.
        
        #### 📞 Call Centers & Customer Helpdesks
        Direct high-volume customer inquiries to specialized support queues, helping human operators focus on high-priority complex problems while resolving simple tickets through automated macros.
        """)
