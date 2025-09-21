#!/usr/bin/env python3
"""
Test script to verify button styling works correctly
"""
import streamlit as st

# Page config
st.set_page_config(
    page_title="Button Styling Test",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS for button styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Button styling - Enhanced selectors */
    .stButton > button,
    .stButton > button:focus,
    .stButton > button:active,
    .stButton > button:hover,
    button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 25px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover,
    button[data-testid="baseButton-primary"]:hover,
    button[data-testid="baseButton-secondary"]:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        color: white !important;
    }
    
    /* Additional button styling for better coverage */
    div[data-testid="stButton"] > button,
    div[data-testid="stButton"] > button:focus,
    div[data-testid="stButton"] > button:active,
    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 25px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Force button styling with higher specificity */
    .stApp .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 25px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stApp .stButton > button:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🎨 Button Styling Test")
    st.markdown("This page tests the button styling to ensure they display with the beautiful gradient colors.")
    
    st.markdown("### Test Buttons:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Primary Button", type="primary"):
            st.success("Primary button clicked!")
    
    with col2:
        if st.button("📚 Secondary Button"):
            st.info("Secondary button clicked!")
    
    with col3:
        if st.button("🎯 Test Button"):
            st.warning("Test button clicked!")
    
    st.markdown("### Expected Results:")
    st.markdown("""
    - ✅ Buttons should have a **purple-blue gradient background**
    - ✅ Text should be **white**
    - ✅ Buttons should have **rounded corners** (25px radius)
    - ✅ Buttons should have a **shadow effect**
    - ✅ On hover, buttons should **lift up** and change color slightly
    """)
    
    st.markdown("### If buttons are still white:")
    st.markdown("""
    1. **Refresh the page** (Ctrl+F5 or Cmd+Shift+R)
    2. **Clear browser cache**
    3. **Check browser developer tools** for CSS conflicts
    4. **Try a different browser**
    """)

if __name__ == "__main__":
    main()
