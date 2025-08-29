import streamlit as st

# page config
st.set_page_config(page_title="Home | Prediction Hub", page_icon="📊", layout="wide")

# app title
st.title("🚀 Group 4 | Machine Learning Operations Prediction Hub")
st.markdown(
    """
    Welcome to **Group 4's Machine Learning Operations Prediction Hub!**  
    Explore our machine learning models built with **PyCaret**, 
    designed to predict outcomes in real-world scenarios across multiple domains.
    """
)

st.divider()

# available Tools Section
st.subheader("🛠️ Available Prediction Tools")

# wheat kernel classifier
st.markdown("🌾 **Wheat Kernel Classification** (Tze Hsuen): Identify wheat varieties (Kama, Rosa, Canadian) based on kernel features.")
if st.button("Go to Wheat Kernel Classification Page"):
    st.switch_page("pages/1_Wheat.py")

st.markdown("---")

# melbourne housing prices
st.markdown("🏠 **Melbourne Housing Prices** (Ian): Estimate residential property prices in Melbourne.")
if st.button("Go to Melbourne Housing Prices Page"):
    st.switch_page("pages/2_Melbourne.py")

st.markdown("---")

# used car prices
st.markdown("🚗 **Used Car Prices** (Aniq): Predict the resale value of used vehicles.")
if st.button("Go to Used Car Prices Page"):
    st.switch_page("pages/3_Used_Car_Prices.py")

st.divider()

# navigation info
st.subheader("📌 How to Navigate between Pages")
st.write("Navigate using the **buttons above** or the **sidebar** on the left.")

st.info("👉 Get started by selecting a topic to begin exploring the predictions!")
