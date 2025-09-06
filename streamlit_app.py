import streamlit as st
import pickle
import numpy as np

# ================= Load Saved Model + Scaler =================
student_package = pickle.load(open("student_model.pkl", "rb"))
log_model = student_package["log_model"]   # Logistic Regression (Pass/Fail)
lin_model = student_package["lin_model"]   # Linear Regression (Percentage)
scaler = student_package["scaler"]         # StandardScaler

# ================= Streamlit UI =================
st.set_page_config(page_title="Student Grade Predictor", layout="centered")
st.title("🎓 Student Final Grade Predictor")

st.write("This app predicts whether a student will **Pass/Fail** and also estimates their **Final Grade Percentage**.")

# ================= Input Sliders =================
study = st.slider("📖 Study Hours Per Day", 0, 10, 5)
attendance = st.slider("📅 Attendance (%)", 0, 100, 75)
sleep = st.slider("😴 Sleep Hours", 0, 12, 7)
midterm = st.slider("📝 Midterm Grade (%)", 0, 100, 50)

# ================= Prediction =================
if st.button("🔮 Predict Result"):
    # Prepare input
    input_data = np.array([[study, attendance, sleep, midterm]])
    input_scaled = scaler.transform(input_data)  # scale input

    # Logistic Regression Prediction (Pass/Fail)
    prediction_class = log_model.predict(input_scaled)[0]
    result = "✅ Pass 😎" if prediction_class == 1 else "❌ Fail 😓"

    # Linear Regression Prediction (Final Percentage)
    prediction_percentage = lin_model.predict(input_scaled)[0]
    prediction_percentage = max(0, min(100, prediction_percentage))  # keep within 0–100

    # ================= Show Results =================
    st.subheader("📊 Prediction Results")
    st.write(f"**Pass/Fail Prediction:** {result}")
    st.write(f"**Predicted Final Grade Percentage:** {prediction_percentage:.2f}%")

    # Progress Bar Visualization
    st.progress(int(prediction_percentage))
    st.write("📈 Predicted grade visualized as progress bar above ⬆️")

    # ================= Friendly Messages =================
    if prediction_class == 1:
        if prediction_percentage >= 80:
            st.balloons()
            st.success("🔥 Excellent! You're on track to score very high. Keep it up! 🎉")
        elif 60 <= prediction_percentage < 80:
            st.success("👍 Good! You're predicted to pass comfortably. Aim higher! 🚀")
        else:
            st.info("🙂 You will likely pass, but there’s room for improvement. Stay consistent 💪")
    else:
        st.error("⚠️ Risk of failing. Try to increase study hours, improve attendance, and revise regularly 📘")

