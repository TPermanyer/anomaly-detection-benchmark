
import streamlit as st

def inject_custom_css():
    """Injects custom CSS for better styling."""
    st.markdown("""
<style>
    /* Global Font & Colors */
    :root {
        --primary-color: #4F8BF9;
        --secondary-color: #2E3B55;
        --background-color: #FFFFFF;
        --text-color: #31333F;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 2.5rem; 
        font-weight: 800; 
        color: var(--primary-color);
        margin-bottom: 0rem;
    }
    
    .sub-header {
        font-size: 1.25rem;
        color: var(--secondary-color);
        margin-bottom: 2rem;
    }

    /* Metric Cards */
    .metric-card {
        background: #F8F9FB; 
        padding: 1.5rem; 
        border-radius: 0.75rem; 
        margin: 0.5rem 0;
        border: 1px solid #E6E9EF;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Streamlit Metric Overrides */
    [data-testid="stMetric"] {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #eee;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    [data-testid="stMetric"] label {
        color: #666 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #111 !important;
        font-weight: 600 !important;
    }

</style>
""", unsafe_allow_html=True)

def render_header():
    """Renders the main application header."""
    st.markdown('<h1 class="main-header">🔍 TFG Anomaly Detection</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Workbench for benchmarking and analyzing anomaly detection algorithms.</div>', unsafe_allow_html=True)

def show_quick_start():
    with st.expander("📚 Quick Start Guide"):
        st.markdown("""
        ### Getting Started
        
        1. **Upload Data**: CSV or Parquet file with your dataset
        2. **Map Columns**: Select features, labels (optional), and timestamp (optional)
        3. **Configure Preprocessing**: Choose between dense or sparse pipelines
        4. **Select Methods**: Pick one or more anomaly detection algorithms
        5. **Run Evaluation**: Click "Run Evaluation" and review results
        
        ### Tips
        - Enable CV for more robust evaluation (requires labels)
        - Use sparse mode for high-cardinality categorical data
        - Compare multiple methods to find the best fit
        - Download scores for further analysis
        """)
