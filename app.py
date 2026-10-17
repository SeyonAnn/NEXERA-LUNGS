import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(__file__))
from utils.preprocessing import preprocess_image
from utils.prediction import predict

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="NEXERA — Lung AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# Custom CSS
# ==============================
st.markdown("""
<style>
    .stApp { background-color: #0a0a0f; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #0d0d1a; border-right: 1px solid #1a1a3a; }
    .nexera-card { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid #1a2a4a; border-radius: 12px; padding: 20px; margin: 10px 0; }
    .metric-card { background: #0d1117; border: 1px solid #1a2a4a; border-radius: 10px;
        padding: 15px; text-align: center; }
    .result-normal { background: linear-gradient(135deg, #0a2a0a 0%, #0d3b0d 100%);
        border: 2px solid #00ff88; border-radius: 12px; padding: 25px; text-align: center; }
    .result-pneumonia { background: linear-gradient(135deg, #2a0a0a 0%, #3b0d0d 100%);
        border: 2px solid #ff3333; border-radius: 12px; padding: 25px; text-align: center; }
    .nexera-title { font-size: 3rem; font-weight: 800;
        background: linear-gradient(90deg, #00ff88, #00aaff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; letter-spacing: 8px; }
    .nexera-subtitle { font-size: 0.95rem; color: #888888; text-align: center;
        letter-spacing: 3px; text-transform: uppercase; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #00aaff;
        border-bottom: 1px solid #1a2a4a; padding-bottom: 8px; margin-bottom: 15px;
        letter-spacing: 2px; text-transform: uppercase; }
    .conf-bar-container { background: #1a1a2a; border-radius: 10px; height: 12px;
        margin: 10px 0; overflow: hidden; }
    .disclaimer { background: #0d1117; border: 1px solid #2a2a4a; border-radius: 8px;
        padding: 12px; font-size: 0.8rem; color: #666666; text-align: center; margin-top: 20px; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .stButton > button { background: linear-gradient(90deg, #00aaff, #00ff88);
        color: #000000; font-weight: 700; border: none; border-radius: 8px;
        padding: 10px 30px; font-size: 1rem; letter-spacing: 1px; width: 100%; }
    .stButton > button:hover { background: linear-gradient(90deg, #00ff88, #00aaff); }
</style>
""", unsafe_allow_html=True)


# ==============================
# Logo Path
# ==============================
logo_path = os.path.join(os.path.dirname(__file__), 'nexera_logo.png')


# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_nexera_model():
    model_path = os.path.join(
        os.path.dirname(__file__), 'model', 'nexera_mobilenet.keras'
    )
    return tf.keras.models.load_model(model_path)


# ==============================
# Sidebar
# ==============================
with st.sidebar:
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    st.markdown("""
    <div style='text-align:center; padding: 10px 0;'>
        <div style='font-size:2rem; font-weight:800;
                    background: linear-gradient(90deg, #00ff88, #00aaff);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    letter-spacing: 4px;'>NEXERA</div>
        <div style='font-size:0.7rem; color:#555555; letter-spacing:2px; margin-top:4px;'>
            AI HEALTHCARE PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div class='section-header'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio(
        "",
        ["🫁  Single Scan Analysis", "📊  Batch Analysis", "📈  Model Performance"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("<div class='section-header'>Model Info</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.8rem; color:#888888; line-height:1.8;'>
        Architecture: MobileNetV2<br>
        Classes: Normal / Pneumonia<br>
        Input: 224 × 224 × 3<br>
        Accuracy: 97.97%<br>
        Precision: 99.22%<br>
        Recall: 96.70%<br>
        F1 Score: 97.94%
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div class='disclaimer'>⚕️ For research and assistance only.<br>
        Not a replacement for professional medical diagnosis.</div>
    """, unsafe_allow_html=True)


# ==============================
# Header with Logo
# ==============================
col_left, col_center, col_right = st.columns([1, 3, 1])
with col_center:
    if os.path.exists(logo_path):
        logo_col1, logo_col2, logo_col3 = st.columns([1, 1, 1])
        with logo_col2:
            st.image(logo_path, width=120)
    st.markdown("""
    <div style='padding: 10px 0 10px 0;'>
        <div class='nexera-title'>NEXERA</div>
        <div class='nexera-subtitle'>AI-Powered Lung Disease Detection Platform</div>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# Metrics Row
# ==============================
col1, col2, col3, col4 = st.columns(4)
metrics_data = [
    ("97.97%", "ACCURACY", "#00ff88"),
    ("99.22%", "PRECISION", "#00aaff"),
    ("96.70%", "RECALL", "#00ff88"),
    ("97.94%", "F1 SCORE", "#00aaff")
]
for col, (val, label, color) in zip([col1, col2, col3, col4], metrics_data):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='font-size:1.8rem; font-weight:800; color:{color};'>{val}</div>
            <div style='font-size:0.7rem; color:#888888; letter-spacing:2px;'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==============================
# PAGE 1 — Single Scan Analysis
# ==============================
if "Single" in page:
    st.markdown(
        "<div class='section-header'>🫁 CT Scan Analysis</div>",
        unsafe_allow_html=True
    )
    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.markdown("""
        <div class='nexera-card'>
            <div style='font-size:0.85rem; color:#00aaff;
                        letter-spacing:2px; margin-bottom:15px;'>
                UPLOAD CT SCAN</div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload a chest CT scan image",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )

        if uploaded_file:
            img_array, img = preprocess_image(uploaded_file)
            st.image(img, caption="Uploaded CT Scan", use_container_width=True)
            st.markdown("""
            <div style='color:#00ff88; font-size:0.8rem; margin-top:8px;'>
                ✅ Image loaded successfully<br>
                ✅ Preprocessing complete — 224×224×3<br>
                ✅ Ready for AI analysis
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        analyze = False
        if uploaded_file:
            analyze = st.button("🔬  ANALYSE CT SCAN")

    with col_result:
        if uploaded_file and analyze:
            with st.spinner("Running AI analysis..."):
                model = load_nexera_model()
                label, score, normal_prob, pneumonia_prob = predict(
                    model, img_array
                )
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if label == 'NORMAL':
                icon, card_class, color = "✅", "result-normal", "#00ff88"
                message = "No pneumonia-related patterns detected in this CT scan."
            else:
                icon, card_class, color = "⚠️", "result-pneumonia", "#ff3333"
                message = "Patterns consistent with pneumonia detected in this CT scan."

            st.markdown(f"""
            <div class='{card_class}'>
                <div style='font-size:2.5rem;'>{icon}</div>
                <div style='font-size:2rem; font-weight:800; color:{color};
                            letter-spacing:4px; margin:10px 0;'>{label}</div>
                <div style='font-size:1.3rem; color:#ffffff; font-weight:600;'>
                    Confidence: {score:.1f}%</div>
                <div class='conf-bar-container' style='margin:15px 0;'>
                    <div style='width:{score:.1f}%; height:100%;
                                background:{color}; border-radius:10px;'></div>
                </div>
                <div style='font-size:0.85rem; color:#aaaaaa;
                            margin-top:10px;'>{message}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            fig = go.Figure(go.Bar(
                x=['Normal', 'Pneumonia'],
                y=[normal_prob, pneumonia_prob],
                marker_color=['#00ff88', '#ff3333'],
                text=[f'{normal_prob:.1f}%', f'{pneumonia_prob:.1f}%'],
                textposition='outside',
                textfont=dict(color='white', size=14)
            ))
            fig.update_layout(
                title='Probability Distribution',
                plot_bgcolor='#0d1117',
                paper_bgcolor='#0d1117',
                font=dict(color='white'),
                yaxis=dict(range=[0, 110], gridcolor='#1a2a4a'),
                xaxis=dict(gridcolor='#1a2a4a'),
                showlegend=False,
                height=280
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"""
            <div class='nexera-card'>
                <div class='section-header'>Analysis Report</div>
                <table style='width:100%; font-size:0.85rem;
                              color:#cccccc; border-collapse:collapse;'>
                    <tr style='border-bottom:1px solid #1a2a4a;'>
                        <td style='padding:8px 0; color:#888888;'>Prediction</td>
                        <td style='padding:8px 0; color:{color};
                                   font-weight:bold;'>{label}</td></tr>
                    <tr style='border-bottom:1px solid #1a2a4a;'>
                        <td style='padding:8px 0; color:#888888;'>Confidence</td>
                        <td style='padding:8px 0;'>{score:.2f}%</td></tr>
                    <tr style='border-bottom:1px solid #1a2a4a;'>
                        <td style='padding:8px 0; color:#888888;'>Normal prob.</td>
                        <td style='padding:8px 0;'>{normal_prob:.2f}%</td></tr>
                    <tr style='border-bottom:1px solid #1a2a4a;'>
                        <td style='padding:8px 0; color:#888888;'>Pneumonia prob.</td>
                        <td style='padding:8px 0;'>{pneumonia_prob:.2f}%</td></tr>
                    <tr style='border-bottom:1px solid #1a2a4a;'>
                        <td style='padding:8px 0; color:#888888;'>Model</td>
                        <td style='padding:8px 0;'>MobileNetV2</td></tr>
                    <tr>
                        <td style='padding:8px 0; color:#888888;'>Timestamp</td>
                        <td style='padding:8px 0;'>{timestamp}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)


# ==============================
# PAGE 2 — Batch Analysis
# ==============================
elif "Batch" in page:
    st.markdown(
        "<div class='section-header'>📊 Batch CT Scan Analysis</div>",
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class='nexera-card'>
        <div style='font-size:0.85rem; color:#00aaff;
                    letter-spacing:2px; margin-bottom:15px;'>
            UPLOAD MULTIPLE CT SCANS</div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload multiple CT scan images",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_files:
        st.markdown(f"""
        <div style='color:#00ff88; font-size:0.85rem; margin:10px 0;'>
            ✅ {len(uploaded_files)} images loaded</div>
        """, unsafe_allow_html=True)

        if st.button("🔬  RUN BATCH ANALYSIS"):
            model = load_nexera_model()
            results = []
            progress = st.progress(0)
            status = st.empty()

            for i, file in enumerate(uploaded_files):
                status.text(f"Analysing {file.name}...")
                img_array, _ = preprocess_image(file)
                label, score, normal_prob, pneumonia_prob = predict(
                    model, img_array
                )
                results.append({
                    'Image Name': file.name,
                    'Prediction': label,
                    'Confidence (%)': f'{score:.2f}',
                    'Normal (%)': f'{normal_prob:.2f}',
                    'Pneumonia (%)': f'{pneumonia_prob:.2f}'
                })
                progress.progress((i + 1) / len(uploaded_files))

            status.text("✅ Batch analysis complete")
            df = pd.DataFrame(results)

            st.markdown(
                "<div class='section-header' style='margin-top:20px;'>Results</div>",
                unsafe_allow_html=True
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

            normal_count = sum(1 for r in results if r['Prediction'] == 'NORMAL')
            pneumonia_count = len(results) - normal_count

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size:1.5rem; font-weight:800;
                                color:#00aaff;'>{len(results)}</div>
                    <div style='font-size:0.7rem; color:#888888;'>TOTAL SCANS</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size:1.5rem; font-weight:800;
                                color:#00ff88;'>{normal_count}</div>
                    <div style='font-size:0.7rem; color:#888888;'>NORMAL</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='font-size:1.5rem; font-weight:800;
                                color:#ff3333;'>{pneumonia_count}</div>
                    <div style='font-size:0.7rem; color:#888888;'>PNEUMONIA</div>
                </div>""", unsafe_allow_html=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️  Download Analysis Report (CSV)",
                data=csv,
                file_name=f'nexera_batch_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )


# ==============================
# PAGE 3 — Model Performance
# ==============================
elif "Performance" in page:
    st.markdown(
        "<div class='section-header'>📈 Model Performance Report</div>",
        unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)

    with col1:
        fig1 = go.Figure(go.Bar(
            x=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            y=[97.97, 99.22, 96.70, 97.94],
            marker_color=['#00aaff', '#00ff88', '#ffaa00', '#ff6688'],
            text=['97.97%', '99.22%', '96.7%', '97.94%'],
            textposition='outside',
            textfont=dict(color='white', size=13)
        ))
        fig1.update_layout(
            title='Model Metrics',
            plot_bgcolor='#0d1117',
            paper_bgcolor='#0d1117',
            font=dict(color='white'),
            yaxis=dict(range=[90, 102], gridcolor='#1a2a4a'),
            xaxis=dict(gridcolor='#1a2a4a'),
            height=350
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure(go.Heatmap(
            z=[[391, 3], [13, 381]],
            x=['Predicted Normal', 'Predicted Pneumonia'],
            y=['Actual Normal', 'Actual Pneumonia'],
            colorscale='Blues',
            text=[[391, 3], [13, 381]],
            texttemplate='%{text}',
            textfont=dict(size=18, color='white'),
            showscale=False
        ))
        fig2.update_layout(
            title='Confusion Matrix (Test Set)',
            plot_bgcolor='#0d1117',
            paper_bgcolor='#0d1117',
            font=dict(color='white'),
            height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        "<div class='section-header'>Training History — MobileNetV2</div>",
        unsafe_allow_html=True
    )

    acc = [0.7279, 0.8889, 0.9187, 0.9435, 0.9550, 0.9490,
           0.9602, 0.9700, 0.9733, 0.9730, 0.9733, 0.9735, 0.9743]
    val_acc = [0.9349, 0.9566, 0.9630, 0.9668, 0.9656, 0.9668,
               0.9681, 0.9707, 0.9681, 0.9643, 0.9668, 0.9668, 0.9668]
    epochs = list(range(1, len(acc) + 1))

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=epochs, y=[a*100 for a in acc],
        mode='lines+markers', name='Training Accuracy',
        line=dict(color='#00aaff', width=2), marker=dict(size=6)
    ))
    fig3.add_trace(go.Scatter(
        x=epochs, y=[a*100 for a in val_acc],
        mode='lines+markers', name='Validation Accuracy',
        line=dict(color='#00ff88', width=2), marker=dict(size=6)
    ))
    fig3.add_vline(
        x=8, line_dash="dash", line_color="#ff3333",
        annotation_text="Best Epoch (8)",
        annotation_font_color="#ff3333"
    )
    fig3.update_layout(
        title='Training vs Validation Accuracy',
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='white'),
        yaxis=dict(range=[60, 102], gridcolor='#1a2a4a', title='Accuracy (%)'),
        xaxis=dict(gridcolor='#1a2a4a', title='Epoch'),
        legend=dict(bgcolor='#0d1117'),
        height=350
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    <div class='nexera-card'>
        <div class='section-header'>Model Summary</div>
        <table style='width:100%; font-size:0.85rem;
                      color:#cccccc; border-collapse:collapse;'>
            <tr style='border-bottom:1px solid #1a2a4a;'>
                <td style='padding:10px 0; color:#888888;'>Architecture</td>
                <td style='padding:10px 0; color:#00aaff;'>
                    MobileNetV2 + Custom Head</td></tr>
            <tr style='border-bottom:1px solid #1a2a4a;'>
                <td style='padding:10px 0; color:#888888;'>Total Parameters</td>
                <td style='padding:10px 0;'>2,435,393</td></tr>
            <tr style='border-bottom:1px solid #1a2a4a;'>
                <td style='padding:10px 0; color:#888888;'>Trainable Parameters</td>
                <td style='padding:10px 0;'>174,849</td></tr>
            <tr style='border-bottom:1px solid #1a2a4a;'>
                <td style='padding:10px 0; color:#888888;'>Frozen Parameters</td>
                <td style='padding:10px 0;'>2,260,544</td></tr>
            <tr style='border-bottom:1px solid #1a2a4a;'>
                <td style='padding:10px 0; color:#888888;'>Training Images</td>
                <td style='padding:10px 0;'>3,664</td></tr>
            <tr style='border-bottom:1px solid #1a2a4a;'>
                <td style='padding:10px 0; color:#888888;'>Test Images</td>
                <td style='padding:10px 0;'>788</td></tr>
            <tr style='border-bottom:1px solid #1a2a4a;'>
                <td style='padding:10px 0; color:#888888;'>Best Epoch</td>
                <td style='padding:10px 0;'>8 / 13</td></tr>
            <tr>
                <td style='padding:10px 0; color:#888888;'>Model Size</td>
                <td style='padding:10px 0;'>11.2 MB</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ==============================
# Footer
# ==============================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class='disclaimer'>
    ⚕️ NEXERA AI HEALTHCARE PLATFORM — Powered by Deep Learning<br>
    This system is developed for research and assistance purposes only.
    It is not a replacement for professional medical diagnosis.
    Always consult a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)