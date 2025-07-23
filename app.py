import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components
from src.models.yoga_recommender import YogaRecommenderSystem
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Yoga Recommender",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern and impressive styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    
    /* Root variables for consistent theming */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --accent-color: #f093fb;
        --success-color: #4ade80;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --dark-bg: #0f172a;
        --card-bg: rgba(255, 255, 255, 0.95);
        --glass-bg: rgba(255, 255, 255, 0.1);
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --shadow-light: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-medium: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        --shadow-heavy: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
    }
    
    /* Global styles */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }
    
    /* Background with gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Header styles with glassmorphism */
    .hero-section {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: var(--shadow-heavy);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: float 6s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #ffffff, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 400;
        margin-bottom: 2rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin-top: 2rem;
        position: relative;
        z-index: 1;
    }
    
    .stat-item {
        text-align: center;
        color: white;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar enhancements */
    .css-1d391kg {
        background: var(--glass-bg) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .sidebar .stSelectbox > div > div {
        background: var(--card-bg);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .sidebar .stNumberInput > div > div > input {
        background: var(--card-bg) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
    }
    
    .sidebar .stMultiSelect > div > div {
        background: var(--card-bg);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Form button enhancement */
    .stFormSubmitButton > button {
        background: linear-gradient(45deg, var(--primary-color), var(--secondary-color)) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--shadow-medium) !important;
        width: 100% !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--shadow-heavy) !important;
    }
    
    /* Recommendation cards with glassmorphism */
    .recommendation-card {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: var(--shadow-medium);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-heavy);
    }
    
    .recommendation-card:hover::before {
        transform: scaleX(1);
    }
    
    .card-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-content {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .info-item {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }
    
    .info-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .info-value {
        font-size: 1rem;
        color: var(--text-primary);
        font-weight: 500;
    }
    
    .score-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(45deg, var(--success-color), #22c55e);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .safety-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(45deg, var(--warning-color), #f59e0b);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .focus-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
    }
    
    .focus-tag {
        background: rgba(102, 126, 234, 0.1);
        color: var(--primary-color);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .body-tag {
        background: rgba(118, 75, 162, 0.1);
        color: var(--secondary-color);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid rgba(118, 75, 162, 0.2);
    }
    
    .reason-box {
        background: rgba(240, 147, 251, 0.1);
        border-left: 4px solid var(--accent-color);
        padding: 1rem;
        border-radius: 0 12px 12px 0;
        margin-top: 1rem;
    }
    
    .reason-text {
        color: var(--text-primary);
        font-style: italic;
        margin: 0;
    }
    
    /* Plotly charts within recommendation cards */
    .recommendation-card .stPlotlyChart {
        background: transparent !important;
        border-radius: 15px !important;
        overflow: hidden !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .recommendation-card .stPlotlyChart > div {
        background: transparent !important;
        border-radius: 15px !important;
        width: 100% !important;
        max-width: 100% !important;
        overflow: hidden !important;
    }
    
    .recommendation-card .js-plotly-plot {
        width: 100% !important;
        height: 500px !important;
        overflow: hidden !important;
        border-radius: 15px !important;
        background: transparent !important;
    }
    
    .recommendation-card .plotly {
        width: 100% !important;
        max-width: 100% !important;
        overflow: hidden !important;
    }

    /* Ensure plotly chart fits within the card */
    .recommendation-card div[data-testid="stPlotlyChart"] {
        width: 100% !important;
        max-width: 100% !important;
        overflow: hidden !important;
    }
    
    /* Fix for plotly modebar positioning */
    .recommendation-card .modebar {
        right: 10px !important;
        top: 10px !important;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        .hero-stats {
            flex-direction: column;
            gap: 1.5rem;
        }
        .card-content {
            grid-template-columns: 1fr;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recommender' not in st.session_state:
    try:
        st.session_state.recommender = YogaRecommenderSystem()
        st.session_state.recommender.load_model('src/models/yoga_recommender_model_final1.pkl')
        # Load synthetic asana data if not included in pickle
        if not hasattr(st.session_state.recommender, 'asanas_processed'):
            st.session_state.recommender.load_and_preprocess_data()  # Generates synthetic data
        st.session_state.model_loaded = True
    except Exception as e:
        st.session_state.model_loaded = False
        st.error(f"Error loading model: {e}")


def create_hero_section():
    """Create the hero section with stats"""
    st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">🧘‍♀️ AI Yoga Recommender</h1>
            <h3 class="hero-title">Start Yoga, Start a New Life 🌸</h3>
            <p class="hero-subtitle">Discover personalized yoga poses powered by advanced machine learning</p>
            <div class="hero-stats">
                <div class="stat-item">
                    <span class="stat-number">141</span>
                    <span class="stat-label">Yoga Poses</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">99%</span>
                    <span class="stat-label">Safety Score</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">AI</span>
                    <span class="stat-label">Powered</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Load asana dataset for additional information
@st.cache_data
def load_asana_dataset():
    """Load the asana dataset with additional pose information"""
    try:
        df = pd.read_csv('data/raw/asana_dataset.csv')
        return df
    except Exception as e:
        st.error(f"Error loading asana dataset: {e}")
        return None

def get_age_appropriate_reps(repetition_str, age):
    """Parse repetition string and return age-appropriate repetitions"""
    try:
        if not repetition_str or pd.isna(repetition_str):
            return "Hold comfortably"
        
        # Handle different formats in the repetition column
        if ':' in str(repetition_str):
            parts = str(repetition_str).split(',')
            for part in parts:
                if ':' in part:
                    age_range, reps = part.split(':')
                    age_range = age_range.strip()
                    
                    if '-' in age_range:
                        min_age, max_age = map(int, age_range.split('-'))
                        if min_age <= age <= max_age:
                            return reps.strip()
                    elif '+' in age_range:
                        min_age = int(age_range.replace('+', ''))
                        if age >= min_age:
                            return reps.strip()
        
        return str(repetition_str)
    except:
        return "Hold comfortably"

def format_duration(duration_secs):
    """Format duration in seconds to readable format"""
    try:
        duration = int(duration_secs)
        if duration >= 60:
            minutes = duration // 60
            seconds = duration % 60
            if seconds > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{minutes}m"
        else:
            return f"{duration}s"
    except:
        return "30s"

def create_recommendation_card(rec, index, asana_df=None, user_age=35):
    """Create a modern recommendation card with additional asana information"""
    # Load asana dataset if not provided
    if asana_df is None:
        asana_df = load_asana_dataset()
    
    # Get difficulty icons
    difficulty_icons = {
        'Beginner': '🌱',
        'Intermediate': '🌿', 
        'Advanced': '🌳'
    }
    
    # Get additional information from asana dataset
    description = "A beneficial yoga pose for your practice."
    benefits = rec.get('benefits', [])
    duration = "30s"
    repetitions = "Hold comfortably"
    
    if asana_df is not None:
        # Find matching pose in the dataset with flexible matching
        pose_name_lower = rec['pose_name'].lower().strip()
        
        # Try exact match first
        pose_data = asana_df[asana_df['asana_name'].str.lower().str.strip() == pose_name_lower]
        
        # If no exact match, try partial match
        if pose_data.empty:
            pose_data = asana_df[asana_df['asana_name'].str.lower().str.contains(pose_name_lower, na=False, regex=False)]
        
        if not pose_data.empty:
            pose_info = pose_data.iloc[0]
            
            # Get description
            if pd.notna(pose_info.get('description')):
                description = str(pose_info['description']).strip()
            
            # Get duration and repetitions
            if pd.notna(pose_info.get('duration_secs')):
                duration = format_duration(pose_info['duration_secs'])
            
            if pd.notna(pose_info.get('repetition')):
                repetitions = get_age_appropriate_reps(pose_info['repetition'], user_age)
    
    # Create focus area tags
    focus_tags = ''.join([f'<span class="focus-tag">{area}</span>' for area in rec['focus_areas']])
    body_tags = ''.join([f'<span class="body-tag">{part}</span>' for part in rec['body_parts']])
    
    # Create benefits tags
    benefits_tags = ''.join([f'<span class="focus-tag">{benefit}</span>' for benefit in benefits]) if benefits else ''
    
    # Clean and escape the description text properly
    if description and str(description).strip():
        clean_description = str(description).replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;').strip()
    else:
        clean_description = "A beneficial yoga pose for your practice."
    
    # Build the benefits section
    benefits_section = ""
    if benefits_tags:
        benefits_section = f'<div class="focus-tags"><strong class="info-label">Benefits:</strong><br>{benefits_tags}</div>'
    
    # Build the precautions section
    precautions_section = ""
    if rec.get('precautions'):
        precaution_tags = ', '.join([f'<span class="body-tag">{str(prec)}</span>' for prec in rec['precautions'] if prec])
        if precaution_tags:
            precautions_section = f'<div class="focus-tags"><strong class="info-label">Precautions:</strong><br>{precaution_tags}</div>'
    
    return f"""
        <div class="recommendation-card">
            <div class="card-title">
                {difficulty_icons.get(rec['difficulty_level'], '🧘')} {rec['pose_name']}
            </div>
            <div class="card-content">
                <div class="info-item">
                    <span class="info-label">Difficulty</span>
                    <span class="info-value">{rec['difficulty_level']}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Recommendation Score</span>
                    <span class="score-badge">⭐ {rec['score']:.1%}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Safety Score</span>
                    <span class="safety-badge">🛡️ {rec['safety_score']:.1%}</span>
                </div>
            </div>
            <div class="card-content">
                <div class="info-item">
                    <span class="info-label">Duration</span>
                    <span class="info-value">⏱️ {duration}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Repetitions</span>
                    <span class="info-value">🔄 {repetitions}</span>
                </div>
            </div>
            <div class="reason-box" style="margin-top: 1rem; border-left-color: #667eea;">
                <p class="reason-text" style="color: var(--text-primary); font-style: normal;">📝 <strong>Description:</strong> {clean_description}</p>
            </div>
            <div class="focus-tags">
                <strong class="info-label">Focus Areas:</strong><br>
                {focus_tags}
            </div>
            <div class="focus-tags">
                <strong class="info-label">Body Parts:</strong><br>
                {body_tags}
            </div>
            {benefits_section}
            {precautions_section}
            <div class="reason-box">
                <p class="reason-text">💡 <strong>Why this pose?</strong> {rec.get('recommendation_reason', 'This pose is recommended for your practice.')}</p>
            </div>
        </div>
    """

def create_interactive_chart(recommendations):
    """Create an interactive Plotly chart"""
    scores = [rec['score'] for rec in recommendations]
    pose_names = [rec['pose_name'] for rec in recommendations]
    difficulties = [rec['difficulty_level'] for rec in recommendations]
    safety_scores = [rec['safety_score'] for rec in recommendations]
    
    # Create color mapping for difficulties
    color_map = {'Beginner': '#4ade80', 'Intermediate': '#f59e0b', 'Advanced': '#ef4444'}
    colors = [color_map.get(diff, '#667eea') for diff in difficulties]
    
    fig = go.Figure()
    
    # Add main bars
    fig.add_trace(go.Bar(
        x=pose_names,
        y=scores,
        name='Recommendation Score',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        hovertemplate='<b>%{x}</b><br>Score: %{y:.1%}<br>Difficulty: %{customdata}<extra></extra>',
        customdata=difficulties
    ))
    
    # Add safety score as a secondary trace
    fig.add_trace(go.Scatter(
        x=pose_names,
        y=safety_scores,
        mode='markers+lines',
        name='Safety Score',
        marker=dict(size=10, color='#f093fb', line=dict(color='white', width=2)),
        line=dict(color='#f093fb', width=3, dash='dot'),
        yaxis='y2',
        hovertemplate='<b>%{x}</b><br>Safety: %{y:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='📊 Your Personalized Recommendations',
            font=dict(size=24, family='Inter', color='#1e293b'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Yoga Poses',
            title_font=dict(size=14, family='Inter'),
            tickfont=dict(size=12, family='Inter'),
            tickangle=-45
        ),
        yaxis=dict(
            title='Recommendation Score',
            title_font=dict(size=14, family='Inter', color='#667eea'),
            tickformat='.1%',
            tickfont=dict(size=12, family='Inter')
        ),
        yaxis2=dict(
            title='Safety Score',
            title_font=dict(size=14, family='Inter', color='#f093fb'),
            overlaying='y',
            side='right',
            tickformat='.1%',
            tickfont=dict(size=12, family='Inter')
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        ),
        margin=dict(t=100, r=80, l=80, b=120),
        height=500,
        autosize=True
    )
    
    # Update grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    
    return fig

def main():
    # Create hero section
    create_hero_section()
    
    # Sidebar for user input
    with st.sidebar:
        st.markdown("### 🐱 Your Yoga Profile")
        st.markdown("Fill in your details to get personalized recommendations")
        
        with st.form(key='user_input_form'):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("🎂 Age", min_value=8, max_value=100, value=35, step=1)
            with col2:
                height = st.number_input("📏 Height (cm)", min_value=100, max_value=250, value=170, step=1)
            
            weight = st.number_input("⚖️ Weight (kg)", min_value=30, max_value=200, value=70, step=1)

            # Physical level
            physical_level = st.selectbox(
                "💪 Physical Fitness Level",
                options=["Beginner", "Intermediate", "Advanced"],
                help="Select your current fitness level"
            )

            # Focus area
            focus_area = st.selectbox(
                "🎯 Primary Goal",
                options=[
                    'Flexibility/Stretching', 'Strength Building', 'Balance/Stability',
                    'Stress Relief/Calming', 'Posture Improvement', 'Meditation/Focus',
                    'Digestion Support', 'Cardiovascular Fitness', 'Endurance Building',
                    'Energy Building', 'Circulation Enhancement', 'Coordination',
                    'Relaxation/Restorative', 'Emotional Release', 'Detoxification',
                    'Breathing Improvement'
                ],
                help="Choose your main focus for yoga practice"
            )

            # Health conditions and injuries
            # health_conditions = st.multiselect(
            #     "⚕️ Health Conditions (Optional)",
            #     options=[
            #         'High blood pressure', 'Low blood pressure', 'Back injuries',
            #         'Knee problems/injuries', 'Neck injuries', 'Shoulder injuries',
            #         'Ankle injuries', 'Wrist injuries', 'Hip injuries', 'Heart conditions',
            #         'Pregnancy', 'Glaucoma/eye conditions', 'Carpal tunnel',
            #         'Hamstring injuries', 'Asthma', 'Migraine', 'Insomnia',
            #         'Diarrhea', 'Menstruation', 'Balance disorders'
            #     ],
            #     help="Select any relevant health conditions or injuries"
            # )

            submit_button = st.form_submit_button(
                label="✨ Get AI Recommendations",
                help="Generate personalized yoga pose recommendations using AI"
            )

    # Main content
    if submit_button and st.session_state.model_loaded:
        # Prepare user input
        user_input = {
            'age': age,
            'height': height,
            'weight': weight,
            'physical_level': physical_level,
            'focus_area': focus_area,
            'health_conditions': [],
            'injuries': [],
            'bp_systolic': 120,
            'bp_diastolic': 80
        }

        # Generate recommendations
        with st.spinner("Generating your personalized yoga recommendations..."):
            recommendations = st.session_state.recommender.recommend_poses(user_input, top_k=5)

        if not recommendations:
            st.markdown("""
                <div class="recommendation-card">
                    <div class="card-title">❌ No Recommendations Found</div>
                    <div class="reason-box">
                        <p class="reason-text">
                            No suitable poses found for your profile. This might be due to specific health conditions 
                            or a very restrictive combination of criteria. Please try adjusting your inputs or consult 
                            a yoga professional for personalized guidance.
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Load asana dataset for additional information
            asana_df = load_asana_dataset()
            
            # Success message
            st.markdown("""
                <div style="text-align: center; margin-bottom: 2rem;">
                    <h2 style="color: white; font-family: 'Inter', sans-serif; font-size: 2rem; margin-bottom: 0.5rem;">
                        ✨ Your AI-Powered Yoga Recommendations
                    </h2>
                    <p style="color: rgba(255,255,255,0.8); font-size: 1.1rem;">
                        Based on your profile, here are the top yoga poses selected just for you
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display recommendation cards with additional information
            for i, rec in enumerate(recommendations, 1):
                try:
                    card_html = create_recommendation_card(rec, i, asana_df, age)
                    st.markdown(card_html, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error creating recommendation card {i}: {e}")
                    # Fallback: display basic information
                    st.markdown(f"""
                        <div class="recommendation-card">
                            <div class="card-title">
                                🧘 {rec.get('pose_name', 'Unknown Pose')}
                            </div>
                            <div class="card-content">
                                <div class="info-item">
                                    <span class="info-label">Difficulty</span>
                                    <span class="info-value">{rec.get('difficulty_level', 'Unknown')}</span>
                                </div>
                                <div class="info-item">
                                    <span class="info-label">Score</span>
                                    <span class="score-badge">⭐ {rec.get('score', 0):.1%}</span>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            # Interactive chart section using streamlit components
            # Extract data from recommendations
            poses = [rec['pose_name'] for rec in recommendations]
            rec_scores = [rec['score'] * 100 for rec in recommendations]  # Convert to percentage
            safety_scores = [rec['safety_score'] * 100 for rec in recommendations]  # Convert to percentage
            difficulties = [rec['difficulty_level'] for rec in recommendations]
            
            # Create color mapping for difficulties
            color_map = {'Beginner': '#4ade80', 'Intermediate': '#f59e0b', 'Advanced': '#ef4444'}
            colors = [color_map.get(diff, '#667eea') for diff in difficulties]
            
            # Create chart with actual data
            fig = go.Figure()
            
            # Add recommendation score bars
            fig.add_bar(
                x=poses,
                y=rec_scores,
                name="Recommendation Score",
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(255,255,255,0.2)', width=1)
                ),
                hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}%<br>Difficulty: %{customdata}<extra></extra>',
                customdata=difficulties
            )
            
            # Add safety score line
            fig.add_scatter(
                x=poses,
                y=safety_scores,
                name="Safety Score",
                mode="markers+lines",
                marker=dict(size=10, color='#f093fb', line=dict(color='white', width=2)),
                line=dict(color='#f093fb', width=3, dash='dot'),
                yaxis='y2',
                hovertemplate='<b>%{x}</b><br>Safety: %{y:.1f}%<extra></extra>'
            )

            fig.update_layout(
                xaxis=dict(
                    title='Yoga Poses',
                    title_font=dict(size=14, family='Inter'),
                    tickfont=dict(size=12, family='Inter'),
                    tickangle=-45
                ),
                yaxis=dict(
                    title='Recommendation Score (%)',
                    title_font=dict(size=14, family='Inter', color='#667eea'),
                    tickfont=dict(size=12, family='Inter'),
                    range=[0, 110]
                ),
                yaxis2=dict(
                    title='Safety Score (%)',
                    title_font=dict(size=14, family='Inter', color='#f093fb'),
                    overlaying='y',
                    side='right',
                    tickfont=dict(size=12, family='Inter'),
                    range=[0, 110]
                ),
                margin=dict(t=30, r=80, l=60, b=120),
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='rgba(0,0,0,0.1)',
                    borderwidth=1
                ),
                showlegend=True
            )
            
            # Update grid
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')

            # Convert chart to HTML
            plot_html = fig.to_html(include_plotlyjs='cdn', full_html=False)

            # Custom CSS + Chart inside HTML
            container_html = f"""
            <style>
            .chart-section {{
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(20px);
                border-radius: 20px;
                padding: 2rem;
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
                margin: 2rem 0;
                position: relative;
                overflow: hidden;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }}

            .chart-section::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #667eea, #764ba2);
                transform: scaleX(0);
                transition: transform 0.3s ease;
            }}

            .chart-section:hover::before {{
                transform: scaleX(1);
            }}

            .chart-section:hover {{
                transform: translateY(-5px);
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            }}

            .chart-title {{
                font-family: 'Inter', sans-serif;
                font-size: 1.5rem;
                font-weight: 600;
                color: #1e293b;
                margin-bottom: 1rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}
            </style>

            <div class="chart-section">
                <div class="chart-title">📊 Your Personalized Recommendations</div>
                {plot_html}
            </div>
            """

            # Render HTML with chart embedded inside styled container
            components.html(container_html, height=650)

            # Summary insights
            st.markdown("""
                <div class="recommendation-card">
                    <div class="card-title">📋 Practice Summary & Tips</div>
                    <div class="card-content">
                        <div class="info-item">
                            <span class="info-label">Average Recommendation Score</span>
                            <span class="score-badge">⭐ {:.1%}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Average Safety Score</span>
                            <span class="safety-badge">🛡️ {:.1%}</span>
                        </div>
                        <div class="info-item">
                            <span class="info-label">Focus Area</span>
                            <span class="info-value">{}</span>
                        </div>
                    </div>
                    <div class="reason-box">
                        <p class="reason-text">
                            💡 <strong>Practice Tips:</strong> Start with the highest-rated poses and gradually work your way through 
                            the list. Always listen to your body and modify poses as needed. Consider practicing in a 
                            quiet, comfortable space with a yoga mat for the best experience.
                        </p>
                    </div>
                </div>
            """.format(
                sum(rec['score'] for rec in recommendations) / len(recommendations),
                sum(rec['safety_score'] for rec in recommendations) / len(recommendations),
                focus_area
            ), unsafe_allow_html=True)

            st.markdown("""
                    <div class="hero-section">
                        <div>            
                            <p class="hero-title">🔬 Built with 💜 by Aditi Salvi</p> 
                            <p> My Notebook - https://www.kaggle.com/code/aditisalvi04/yoga-recommender-system </p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

    elif not st.session_state.model_loaded:
        st.error(
            "Model failed to load. Please ensure the 'yoga_recommender_model_final.pkl' file is in the correct directory.")


if __name__ == "__main__":
    main()