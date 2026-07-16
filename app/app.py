import pandas as pd
import joblib
import streamlit as st


st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------
# Load Saved Model
# -----------------------------------------

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "insurance_lasso_model.pkl")
feature_columns = joblib.load(BASE_DIR / "models" / "feature_columns.pkl")


# -----------------------------------------
# HTML Renderer
# -----------------------------------------

def render_html(markup):
    if hasattr(st, "html"):
        st.html(markup)
    else:
        st.markdown(markup, unsafe_allow_html=True)


# -----------------------------------------
# Page Styling
# -----------------------------------------

st.markdown(
    """
    <style>
        :root {
            --app-blue: #2563eb;
            --app-blue-dark: #1d4ed8;
            --app-blue-soft: #eff6ff;
            --ink: #0f172a;
            --text: #111827;
            --muted: #667085;
            --soft: #f8fafc;
            --panel: #ffffff;
            --line: #e5e7eb;
            --line-strong: #d0d5dd;
            --success: #059669;
        }

        @keyframes enter-up {
            from {
                opacity: 0;
                transform: translateY(14px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fade-scale {
            from {
                opacity: 0;
                transform: scale(0.985);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }

        @keyframes shimmer {
            0% {
                transform: translateX(-110%);
            }
            100% {
                transform: translateX(110%);
            }
        }

        .stApp {
            background:
                linear-gradient(180deg, #f8fafc 0%, #ffffff 36%, #f8fafc 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1220px;
            padding: 1.15rem 1.25rem 2.5rem;
        }

        #MainMenu,
        header,
        footer {
            visibility: hidden;
        }

        .app-shell {
            animation: enter-up 520ms cubic-bezier(.2, .8, .2, 1) both;
        }

        .topbar {
            align-items: center;
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid var(--line);
            border-radius: 1.15rem;
            box-shadow: 0 16px 45px rgba(15, 23, 42, 0.06);
            display: flex;
            justify-content: space-between;
            margin-bottom: 1rem;
            padding: 0.85rem 1rem;
        }

        .brand {
            align-items: center;
            display: flex;
            gap: 0.78rem;
        }

        .brand-mark {
            align-items: center;
            background: var(--ink);
            border-radius: 0.85rem;
            color: #ffffff;
            display: flex;
            font-size: 0.9rem;
            font-weight: 800;
            height: 2.65rem;
            justify-content: center;
            letter-spacing: 0;
            width: 2.65rem;
        }

        .brand-title {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 780;
            line-height: 1.2;
        }

        .brand-subtitle {
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.35;
        }

        .topbar-actions {
            align-items: center;
            display: flex;
            gap: 0.55rem;
        }

        .status-pill {
            align-items: center;
            background: #ecfdf3;
            border: 1px solid #abefc6;
            border-radius: 999px;
            color: #067647;
            display: inline-flex;
            font-size: 0.78rem;
            font-weight: 720;
            gap: 0.38rem;
            padding: 0.43rem 0.7rem;
            white-space: nowrap;
        }

        .status-dot {
            background: var(--success);
            border-radius: 999px;
            display: inline-block;
            height: 0.48rem;
            width: 0.48rem;
        }

        .dataset-pill {
            background: var(--app-blue-soft);
            border: 1px solid #bfdbfe;
            border-radius: 999px;
            color: #1e3a8a;
            font-size: 0.78rem;
            font-weight: 720;
            padding: 0.43rem 0.72rem;
            white-space: nowrap;
        }

        .dashboard-hero {
            animation: enter-up 560ms cubic-bezier(.2, .8, .2, 1) 80ms both;
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.08), rgba(255, 255, 255, 0) 48%),
                var(--panel);
            border: 1px solid var(--line);
            border-radius: 1.35rem;
            box-shadow: 0 18px 52px rgba(15, 23, 42, 0.06);
            display: grid;
            grid-template-columns: minmax(0, 1fr) 330px;
            gap: 1.2rem;
            margin-bottom: 1rem;
            overflow: hidden;
            padding: 1.35rem;
            position: relative;
        }

        .dashboard-hero::after {
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.55), transparent);
            content: "";
            height: 100%;
            left: 0;
            pointer-events: none;
            position: absolute;
            top: 0;
            transform: translateX(-110%);
            width: 36%;
        }

        .dashboard-hero:hover::after {
            animation: shimmer 900ms ease;
        }

        .kicker {
            color: var(--app-blue);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            margin-bottom: 0.65rem;
            text-transform: uppercase;
        }

        .dashboard-hero h1 {
            color: var(--ink);
            font-size: clamp(2rem, 4vw, 3.4rem);
            font-weight: 820;
            letter-spacing: 0;
            line-height: 1.03;
            margin: 0 0 0.75rem;
            max-width: 780px;
        }

        .hero-copy {
            color: #475467;
            font-size: 1rem;
            line-height: 1.65;
            margin: 0;
            max-width: 760px;
        }

        .hero-stats {
            display: grid;
            gap: 0.7rem;
        }

        .hero-stat {
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid var(--line);
            border-radius: 1rem;
            padding: 0.9rem 0.95rem;
        }

        .hero-stat span {
            color: var(--muted);
            display: block;
            font-size: 0.76rem;
            font-weight: 650;
            margin-bottom: 0.18rem;
        }

        .hero-stat strong {
            color: var(--ink);
            display: block;
            font-size: 1rem;
            font-weight: 780;
        }

        .section-heading {
            color: var(--ink);
            font-size: 1.03rem;
            font-weight: 780;
            letter-spacing: 0;
            margin: 0 0 0.16rem;
        }

        .section-copy {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.52;
            margin: 0 0 1.05rem;
        }

        .field-group {
            align-items: center;
            display: flex;
            gap: 0.5rem;
            margin: 0.3rem 0 0.5rem;
        }

        .field-index {
            align-items: center;
            background: var(--app-blue-soft);
            border: 1px solid #bfdbfe;
            border-radius: 0.65rem;
            color: #1e40af;
            display: flex;
            font-size: 0.76rem;
            font-weight: 800;
            height: 1.85rem;
            justify-content: center;
            width: 1.85rem;
        }

        .field-title {
            color: var(--ink);
            font-size: 0.88rem;
            font-weight: 760;
        }

        .profile-preview {
            background: #f9fafb;
            border: 1px solid var(--line);
            border-radius: 1rem;
            display: grid;
            gap: 0.6rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin: 1rem 0 1.1rem;
            padding: 0.72rem;
        }

        .preview-item {
            min-width: 0;
            padding: 0.2rem;
        }

        .preview-item span {
            color: var(--muted);
            display: block;
            font-size: 0.72rem;
            font-weight: 650;
            margin-bottom: 0.12rem;
        }

        .preview-item strong {
            color: var(--ink);
            display: block;
            font-size: 0.9rem;
            font-weight: 760;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .result-stage {
            animation: fade-scale 380ms cubic-bezier(.2, .8, .2, 1) both;
            background:
                linear-gradient(180deg, rgba(37, 99, 235, 0.08), rgba(37, 99, 235, 0) 62%),
                #ffffff;
            border: 1px solid #bfdbfe;
            border-radius: 1.15rem;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
            padding: 1.1rem;
        }

        .result-meta {
            align-items: center;
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.75rem;
        }

        .result-label {
            color: #1e40af;
            font-size: 0.76rem;
            font-weight: 820;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .result-chip {
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 999px;
            color: #1e40af;
            font-size: 0.74rem;
            font-weight: 730;
            padding: 0.32rem 0.55rem;
        }

        .result-value {
            color: var(--ink);
            font-size: clamp(2.25rem, 4vw, 3.55rem);
            font-weight: 850;
            letter-spacing: 0;
            line-height: 1;
            margin: 0 0 0.28rem;
        }

        .result-period {
            color: var(--muted);
            font-size: 0.92rem;
            font-weight: 680;
            margin-bottom: 1rem;
        }

        .model-note {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid #dbeafe;
            border-radius: 0.95rem;
            color: #475467;
            font-size: 0.88rem;
            line-height: 1.56;
            padding: 0.86rem 0.92rem;
        }

        .empty-state {
            align-items: center;
            background:
                linear-gradient(180deg, #ffffff, #f9fafb);
            border: 1px dashed var(--line-strong);
            border-radius: 1.15rem;
            color: var(--muted);
            display: grid;
            min-height: 245px;
            padding: 1.2rem;
            text-align: center;
        }

        .empty-state strong {
            color: var(--ink);
            display: block;
            font-size: 1rem;
            margin-bottom: 0.28rem;
        }

        .empty-orbit {
            align-items: center;
            background: var(--app-blue-soft);
            border: 1px solid #bfdbfe;
            border-radius: 999px;
            color: #1e40af;
            display: flex;
            font-size: 1.35rem;
            font-weight: 820;
            height: 4rem;
            justify-content: center;
            margin: 0 auto 0.9rem;
            width: 4rem;
        }

        .insight-row {
            display: grid;
            gap: 0.75rem;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            margin-top: 1rem;
        }

        .insight-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 1rem;
            padding: 0.9rem;
        }

        .insight-card span {
            color: var(--muted);
            display: block;
            font-size: 0.76rem;
            font-weight: 650;
            margin-bottom: 0.16rem;
        }

        .insight-card strong {
            color: var(--ink);
            display: block;
            font-size: 0.96rem;
            font-weight: 780;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            animation: enter-up 560ms cubic-bezier(.2, .8, .2, 1) 140ms both;
            background: rgba(255, 255, 255, 0.96);
            border-color: var(--line);
            border-radius: 1.25rem;
            box-shadow: 0 18px 52px rgba(15, 23, 42, 0.07);
            transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: #cbd5e1;
            box-shadow: 0 24px 66px rgba(15, 23, 42, 0.09);
            transform: translateY(-2px);
        }

        label[data-testid="stWidgetLabel"] p {
            color: #344054;
            font-size: 0.84rem;
            font-weight: 720;
        }

        div[data-testid="stNumberInput"] input,
        div[data-baseweb="select"] > div {
            background: #111827 !important;
            border-color: #1f2937 !important;
            border-radius: 0.82rem;
            color: #ffffff !important;
            min-height: 2.8rem;
            transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
        }

        div[data-testid="stNumberInput"] input,
        div[data-testid="stNumberInput"] input::placeholder,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] div {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        div[data-testid="stNumberInput"] button {
            color: #ffffff !important;
            background: #1f2937 !important;
            border-color: #374151 !important;
        }

        div[data-testid="stNumberInput"] input:focus,
        div[data-baseweb="select"] > div:focus-within {
            border-color: var(--app-blue);
            box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.12);
        }

        div.stButton > button {
            background: var(--app-blue);
            border: 1px solid var(--app-blue);
            border-radius: 0.92rem;
            box-shadow: 0 14px 30px rgba(37, 99, 235, 0.25);
            color: #ffffff;
            font-size: 0.98rem;
            font-weight: 820;
            min-height: 3.25rem;
            transition: transform 160ms ease, box-shadow 160ms ease, background 160ms ease;
        }

        div.stButton > button:hover {
            background: var(--app-blue-dark);
            border-color: var(--app-blue-dark);
            box-shadow: 0 18px 38px rgba(37, 99, 235, 0.32);
            color: #ffffff;
            transform: translateY(-2px);
        }

        div.stButton > button:active {
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.24);
            transform: translateY(1px) scale(0.995);
        }

        @media (max-width: 980px) {
            .dashboard-hero {
                grid-template-columns: 1fr;
            }

            .hero-stats,
            .insight-row {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 720px) {
            .block-container {
                padding-left: 0.85rem;
                padding-right: 0.85rem;
            }

            .topbar,
            .topbar-actions {
                align-items: flex-start;
                flex-direction: column;
            }

            .dashboard-hero {
                padding: 1rem;
            }

            .hero-stats,
            .profile-preview,
            .insight-row {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# App Header
# --------------------------------------------------

render_html(
    """
    <div class="app-shell">
        <div class="topbar">
            <div class="brand">
                <div class="brand-mark">MI</div>
                <div>
                    <div class="brand-title">Insurance Cost Studio</div>
                    <div class="brand-subtitle">Fast charge estimation for US medical insurance profiles</div>
                </div>
            </div>
            <div class="topbar-actions">
                <span class="status-pill"><span class="status-dot"></span>Model loaded</span>
                <span class="dataset-pill">US Medical Cost Dataset</span>
            </div>
        </div>

        <div class="dashboard-hero">
            <div>
                <div class="kicker">Prediction Workspace</div>
                <h1>Medical Insurance Cost Prediction</h1>
                <p class="hero-copy">
                    A clean estimator experience for predicting annual insurance charges
                    from customer demographics, BMI, smoking status, family size, and region.
                </p>
            </div>
            <div class="hero-stats">
                <div class="hero-stat">
                    <span>Algorithm</span>
                    <strong>Lasso Regression</strong>
                </div>
                <div class="hero-stat">
                    <span>Model Selection</span>
                    <strong>GridSearchCV</strong>
                </div>
                <div class="hero-stat">
                    <span>Prediction Unit</span>
                    <strong>Annual charges, USD</strong>
                </div>
            </div>
        </div>
    </div>
    """
)


# --------------------------------------------------
# Main Content
# --------------------------------------------------

left_card, right_card = st.columns([1.1, 0.9], gap="large")

with left_card:
    with st.container(border=True):
        render_html(
            """
            <p class="section-heading">Customer Profile</p>
            <p class="section-copy">
                Enter the policyholder attributes below. The app converts categorical
                inputs with the same encoding structure used by the trained model.
            </p>
            <div class="field-group">
                <div class="field-index">01</div>
                <div class="field-title">Profile Details</div>
            </div>
            """
        )

        # ---------------------------------------------
        # Input Fields
        # ---------------------------------------------

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=25,
            )

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"],
            )

            children = st.number_input(
                "Children",
                min_value=0,
                max_value=10,
                value=0,
            )

        with col2:
            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=60.0,
                value=25.0,
            )

            smoker = st.selectbox(
                "Smoker",
                ["No", "Yes"],
            )

            region = st.selectbox(
                "Region",
                [
                    "Northeast",
                    "Northwest",
                    "Southeast",
                    "Southwest",
                ],
            )

        render_html(
            f"""
            <div class="profile-preview">
                <div class="preview-item">
                    <span>Customer</span>
                    <strong>{age} years · {gender}</strong>
                </div>
                <div class="preview-item">
                    <span>Health profile</span>
                    <strong>BMI {bmi:.1f} · Smoker {smoker}</strong>
                </div>
                <div class="preview-item">
                    <span>Household</span>
                    <strong>{children} children · {region}</strong>
                </div>
            </div>
            """
        )

        predict_button = st.button(
            "Predict Insurance Charges",
            use_container_width=True,
        )

with right_card:
    with st.container(border=True):
        render_html(
            """
            <p class="section-heading">Prediction Result</p>
            <p class="section-copy">
                Run the estimator to generate the annual charge prediction for this profile.
            </p>
            """
        )

        if predict_button:

            input_data = pd.DataFrame({
                "age": [age],
                "bmi": [bmi],
                "children": [children],
                "sex_male": [1 if gender == "Male" else 0],
                "smoker_yes": [1 if smoker == "Yes" else 0],
                "region_northwest": [1 if region == "Northwest" else 0],
                "region_southeast": [1 if region == "Southeast" else 0],
                "region_southwest": [1 if region == "Southwest" else 0],
            })

            # Ensure the column order matches the training data
            input_data = input_data[feature_columns]

            prediction = model.predict(input_data)

            render_html(
                f"""
                <div class="result-stage">
                    <div class="result-meta">
                        <div class="result-label">Estimated Annual Charges</div>
                        <div class="result-chip">Ready</div>
                    </div>
                    <div class="result-value">${prediction[0]:,.2f}</div>
                    <div class="result-period">USD per year</div>
                    <div class="model-note">
                        Predictions are generated using a trained Lasso Regression model and
                        represent insurance charges from the Medical Cost Personal Dataset (US).
                    </div>
                </div>
                """
            )

        else:
            render_html(
                """
                <div class="empty-state">
                    <div>
                        <div class="empty-orbit">$</div>
                        <strong>Enter customer details and click Predict.</strong>
                        <span>The estimated annual insurance charge will appear here.</span>
                    </div>
                </div>
                """
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

render_html(
    """
    <div class="insight-row">
        <div class="insight-card">
            <span>Algorithm</span>
            <strong>Lasso Regression</strong>
        </div>
        <div class="insight-card">
            <span>Encoding</span>
            <strong>pd.get_dummies()</strong>
        </div>
        <div class="insight-card">
            <span>Framework</span>
            <strong>Streamlit</strong>
        </div>
        <div class="insight-card">
            <span>Model Selection</span>
            <strong>GridSearchCV</strong>
        </div>
    </div>
    """
)
