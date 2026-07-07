import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

from src.utils import load_model, prepare_input
from src.recommender import recommend

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Maternal Health AI Dashboard",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------

model, encoder = load_model()

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------

st.markdown("""
<style>

.stApp{
    background:linear-gradient(
    135deg,
    #07111f,
    #0f172a,
    #111827);
}

/* remove top padding */

.block-container{
    padding-top:1rem;
    max-width:1450px;
}

/* hero */

.hero{

padding:30px;

border-radius:20px;

background:rgba(255,255,255,0.05);

border:1px solid rgba(255,255,255,0.08);

box-shadow:0px 10px 25px rgba(0,0,0,0.30);

}

/* cards */

.card{

padding:25px;

border-radius:18px;

background:rgba(255,255,255,0.05);

border:1px solid rgba(255,255,255,0.08);

box-shadow:0 10px 20px rgba(0,0,0,0.25);

}

/* titles */

.big{

font-size:48px;

font-weight:800;

color:white;

}

.subtitle{

font-size:18px;

color:#cbd5e1;

}

.section{

font-size:28px;

font-weight:700;

margin-top:15px;

color:white;

}

.metric{

font-size:18px;

font-weight:bold;

color:#60a5fa;

}

</style>

""", unsafe_allow_html=True)

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------

st.markdown("""

<div class="hero">

<div class="big">

🤰 Maternal Health AI Dashboard

</div>

<div class="subtitle">

Machine Learning based Maternal Risk Prediction and
Personalized Diet • Exercise • Medicine Recommendation

</div>

</div>

""", unsafe_allow_html=True)

st.write("")

# -------------------------------------------------------
# PATIENT DETAILS
# -------------------------------------------------------

st.markdown(
'<p class="section">👩 Patient Information</p>',
unsafe_allow_html=True
)

left,right=st.columns(2)

with left:

    age=st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=25
    )

    sbp=st.number_input(
        "Systolic Blood Pressure",
        min_value=70,
        max_value=220,
        value=120
    )

    dbp=st.number_input(
        "Diastolic Blood Pressure",
        min_value=40,
        max_value=150,
        value=80
    )

with right:

    bs=st.number_input(
        "Blood Sugar",
        min_value=0.0,
        max_value=20.0,
        value=5.5,
        step=0.1
    )

    temp=st.number_input(
        "Body Temperature",
        min_value=95.0,
        max_value=105.0,
        value=98.6,
        step=0.1
    )

    hr=st.number_input(
        "Heart Rate",
        min_value=40,
        max_value=180,
        value=80
    )

trimester=st.selectbox(

"Trimester",

[
"1st Trimester",
"2nd Trimester",
"3rd Trimester"
]

)

st.write("")

analyze=st.button(
"🔍 Analyze Patient",
use_container_width=True
)

# -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------

if analyze:

    sample=prepare_input(
        age,
        sbp,
        dbp,
        bs,
        temp,
        hr
    )

    prediction=model.predict(sample)

    probabilities=model.predict_proba(sample)[0]

    risk=encoder.inverse_transform(prediction)[0]

    probability_map=dict(
        zip(
            encoder.classes_,
            probabilities
        )
    )

    high=probability_map.get(
        "high risk",
        0
    )*100

    mid=probability_map.get(
        "mid risk",
        0
    )*100

    low=probability_map.get(
        "low risk",
        0
    )*100

    recommendation=recommend(

        risk,

        sbp,

        dbp,

        bs,

        temp,

        hr,

        trimester

    )
        # =====================================================
    # RISK ANALYSIS
    # =====================================================

    st.write("")
    st.markdown(
        '<p class="section">📊 Risk Analysis</p>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([2, 1])

    # -------------------------------------
    # LEFT SIDE
    # -------------------------------------

    with col1:

        st.markdown("#### Prediction Confidence")

        st.write("🔴 High Risk")
        st.progress(min(100, int(high)))
        st.caption(f"{high:.2f}%")

        st.write("🟡 Mid Risk")
        st.progress(min(100, int(mid)))
        st.caption(f"{mid:.2f}%")

        st.write("🟢 Low Risk")
        st.progress(min(100, int(low)))
        st.caption(f"{low:.2f}%")

    # -------------------------------------
    # RIGHT SIDE
    # -------------------------------------

    with col2:

        if risk.lower() == "high risk":
            score = high
            color = "#ef4444"

        elif risk.lower() == "mid risk":
            score = mid
            color = "#f59e0b"

        else:
            score = low
            color = "#22c55e"

        fig = go.Figure(
            go.Pie(
                values=[score, 100-score],
                hole=0.78,
                textinfo="none",
                marker=dict(
                    colors=[
                        color,
                        "#1f2937"
                    ]
                )
            )
        )

        fig.update_layout(

            height=320,

            showlegend=False,

            margin=dict(
                t=0,
                b=0,
                l=0,
                r=0
            ),

            paper_bgcolor="rgba(0,0,0,0)",

            annotations=[

                dict(

                    text=f"<b>{score:.1f}%</b><br>{risk.upper()}",

                    x=0.5,

                    y=0.5,

                    showarrow=False,

                    font=dict(
                        size=18,
                        color="white"
                    )

                )

            ]

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # VITAL SUMMARY
    # =====================================================

    st.write("")

    st.markdown(
        '<p class="section">🩺 Patient Vital Summary</p>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Blood Pressure",
            f"{sbp}/{dbp} mmHg"
        )

        st.metric(
            "Blood Sugar",
            f"{bs} mmol/L"
        )

    with c2:

        st.metric(
            "Temperature",
            f"{temp} °F"
        )

        st.metric(
            "Heart Rate",
            f"{hr} bpm"
        )

    with c3:

        st.metric(
            "Trimester",
            trimester
        )

        st.metric(
            "Predicted Risk",
            risk.upper()
        )

    # =====================================================
    # ALERT PANEL
    # =====================================================

    st.write("")

    if risk.lower() == "high risk":

        st.error(
            """
🚨 High Risk Pregnancy Detected

Immediate medical consultation is recommended.

• Monitor BP regularly

• Monitor Blood Sugar

• Follow prescribed medication

• Avoid strenuous activity
            """
        )

    elif risk.lower() == "mid risk":

        st.warning(
            """
⚠ Moderate Risk Pregnancy

Regular monitoring is recommended.

• Maintain healthy diet

• Monitor vitals weekly

• Continue prenatal care
            """
        )

    else:

        st.success(
            """
✅ Low Risk Pregnancy

Continue healthy lifestyle.

• Balanced nutrition

• Prenatal exercise

• Routine antenatal checkups
            """
        )
            # =====================================================
    # PERSONALIZED RECOMMENDATIONS
    # =====================================================

    st.write("")
    st.markdown(
        '<p class="section">🍽 Personalized Recommendations</p>',
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    # ------------------------------
    # BREAKFAST
    # ------------------------------

    with c1:

        st.markdown("""
        <div class="card">
        <h3>🍳 Breakfast</h3>
        </div>
        """, unsafe_allow_html=True)

        st.success(recommendation["breakfast"])

    # ------------------------------
    # LUNCH
    # ------------------------------

    with c2:

        st.markdown("""
        <div class="card">
        <h3>🍛 Lunch</h3>
        </div>
        """, unsafe_allow_html=True)

        st.success(recommendation["lunch"])

    # ------------------------------

    c3, c4 = st.columns(2)

    with c3:

        st.markdown("""
        <div class="card">
        <h3>🍽 Dinner</h3>
        </div>
        """, unsafe_allow_html=True)

        st.success(recommendation["dinner"])

    with c4:

        st.markdown("""
        <div class="card">
        <h3>🏃 Exercise</h3>
        </div>
        """, unsafe_allow_html=True)

        st.info(recommendation["exercise"])

    st.write("")

    st.markdown("""
    <div class="card">
    <h3>💊 Recommended Medicine</h3>
    </div>
    """, unsafe_allow_html=True)

    st.warning(recommendation["medicine"])

    # =====================================================
    # AI SUMMARY
    # =====================================================

    st.write("")
    st.markdown(
        '<p class="section">🧠 AI Health Summary</p>',
        unsafe_allow_html=True
    )

    if risk.lower() == "low risk":

        summary = f"""
Patient appears to be in the **LOW RISK** category.

• Continue routine antenatal visits.

• Follow the recommended healthy diet.

• Perform light exercise.

• Continue nutritional supplements.

• Maintain hydration.

Overall maternal condition appears stable.
"""

    elif risk.lower() == "mid risk":

        summary = f"""
Patient falls under the **MID RISK** category.

• Regular BP and blood sugar monitoring is advised.

• Follow dietary recommendations strictly.

• Continue prescribed medication.

• Moderate physical activity is recommended.

• Schedule frequent antenatal checkups.
"""

    else:

        summary = f"""
Patient falls under the **HIGH RISK** category.

• Immediate medical supervision is strongly recommended.

• Strict medication adherence is required.

• Follow low-sodium / diabetic diet according to recommendations.

• Avoid strenuous physical activity.

• Continuous monitoring of maternal vitals is necessary.
"""

    st.info(summary)

    # =====================================================
    # PATIENT REPORT
    # =====================================================

    st.write("")
    st.markdown(
        '<p class="section">📄 Patient Report</p>',
        unsafe_allow_html=True
    )

    report = f"""
==========================
MATERNAL HEALTH REPORT
==========================

Age : {age}

Trimester : {trimester}

Blood Pressure : {sbp}/{dbp}

Blood Sugar : {bs}

Temperature : {temp}

Heart Rate : {hr}

Predicted Risk :

{risk.upper()}

--------------------------

Breakfast

{recommendation["breakfast"]}

--------------------------

Lunch

{recommendation["lunch"]}

--------------------------

Dinner

{recommendation["dinner"]}

--------------------------

Exercise

{recommendation["exercise"]}

--------------------------

Medicine

{recommendation["medicine"]}

==========================
"""

    st.download_button(

        label="📥 Download Patient Report",

        data=report,

        file_name="Maternal_Health_Report.txt",

        mime="text/plain"

    )

    # =====================================================
    # FOOTER
    # =====================================================

    st.write("")
    st.divider()

    st.caption(
        "Maternal Health AI Recommendation System | Internship Project | "
        "Educational Use Only - Recommendations do not replace clinical judgment."
    )