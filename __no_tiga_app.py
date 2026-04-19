import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_models():
    clf = joblib.load(BASE_DIR / "artifacts" / "clf_pipeline.pkl")
    reg = joblib.load(BASE_DIR / "artifacts" / "reg_pipeline.pkl")
    return clf, reg

clf_model, reg_model = load_models()

def predict(features: dict):
    df = pd.DataFrame([features])
    placed       = int(clf_model.predict(df)[0])
    placed_prob  = float(clf_model.predict_proba(df)[0][1])
    salary       = max(0.0, float(reg_model.predict(df)[0]))
    return placed, placed_prob, salary

def main():
    st.set_page_config(
        page_title="Placement Predictor", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with st.sidebar:
        st.title("Parameters")
        st.markdown("Enter the student's details below:")
        
        with st.form("prediction_form"):
            with st.expander("Academic Performance", expanded=True):
                ssc_percentage        = st.slider("SSC Percentage",        0, 100, 72)
                hsc_percentage        = st.slider("HSC Percentage",        0, 100, 72)
                degree_percentage     = st.slider("Degree Percentage",     0, 100, 72)
                cgpa                  = st.slider("CGPA",                  0.0, 10.0, 7.5, 0.01)
                entrance_exam_score   = st.slider("Entrance Score",        0, 100, 60)
                backlogs              = st.number_input("Backlogs",        0, 20,  1)
                attendance_percentage = st.slider("Attendance (%)",        0, 100, 80)

            with st.expander("Skills & Experience", expanded=False):
                technical_skill_score = st.slider("Technical Skill",       0, 100, 60)
                soft_skill_score      = st.slider("Soft Skill",            0, 100, 60)
                certifications        = st.number_input("Certifications",  0, 20,  2)
                internship_count      = st.number_input("Internships",     0, 10,  1)
                live_projects         = st.number_input("Live Projects",   0, 10,  1)
                work_experience_months= st.slider("Work Exp. (months)",    0, 60,  6)

            with st.expander("Personal Profile", expanded=False):
                gender                     = st.selectbox("Gender", ["Male", "Female"])
                extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])

            submit_btn = st.form_submit_button("Run Prediction", use_container_width=True, type="primary")

    features = {
        "ssc_percentage":            ssc_percentage,
        "hsc_percentage":            hsc_percentage,
        "degree_percentage":         degree_percentage,
        "cgpa":                      cgpa,
        "entrance_exam_score":       entrance_exam_score,
        "technical_skill_score":     technical_skill_score,
        "soft_skill_score":          soft_skill_score,
        "internship_count":          internship_count,
        "live_projects":             live_projects,
        "work_experience_months":    work_experience_months,
        "certifications":            certifications,
        "attendance_percentage":     attendance_percentage,
        "backlogs":                  backlogs,
        "gender":                    gender,
        "extracurricular_activities": extracurricular_activities,
    }

    st.title("Student Placement & Salary Predictor")
    st.markdown("***Frederick Allensius - 2802405030 - Dataset B - UTS Model Deployment 2025/2026***")
    
    st.info("**Adjust the student parameters in the sidebar and click 'Run Prediction' to see the results.**")

    placed, placed_prob, salary = predict(features)

    st.markdown("### Prediction Results")
    res_col1, res_col2 = st.columns(2)
    
    container_kwargs = {"border": True}
        
    with res_col1:
        with st.container(**container_kwargs):
            st.subheader("Placement Status")
            if placed == 1:
                st.success("Likely to be Placed")
                st.metric("Placement Probability", f"{placed_prob:.1%}")
                st.progress(placed_prob)
            else:
                st.error("Unlikely to be Placed")
                st.metric("Placement Probability", f"{placed_prob:.1%}")
                st.progress(placed_prob)
                
    with res_col2:
        with st.container(**container_kwargs):
            st.subheader("Expected Salary")
            if placed == 1:
                st.info("Estimated Compensation")
                st.metric("Salary (LPA)", f"{salary:.2f}")
                st.markdown("*Projected earning potential considering current market trends.*")
            else:
                st.warning("Not Applicable")
                st.metric("Salary (LPA)", "N/A")
                st.markdown("*Salary estimation is only available for placed candidates.*")

    st.divider()

    st.markdown("### Applicant Profile Analysis")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        with st.container(**container_kwargs):
            st.markdown("#### Academic History (%)")
            df_academic = pd.DataFrame({
                "Metric": ["SSC", "HSC", "Degree", "Entrance", "Attendance"],
                "Score": [ssc_percentage, hsc_percentage, degree_percentage, entrance_exam_score, attendance_percentage]
            }).set_index("Metric")
            st.bar_chart(df_academic)
        
    with viz_col2:
        with st.container(**container_kwargs):
            st.markdown("#### Skills & CGPA (Scaled to 100)")
            df_skills = pd.DataFrame({
                "Skill": ["Technical Skill", "Soft Skill", "CGPA (Scaled)"],
                "Score": [technical_skill_score, soft_skill_score, cgpa * 10]
            }).set_index("Skill")
            st.bar_chart(df_skills)

    with st.expander("View Raw Feature Data"):
        df_summary = pd.DataFrame([features]).T.rename(columns={0: "Value"})
        st.dataframe(df_summary, use_container_width=True)

if __name__ == "__main__":
    main()
