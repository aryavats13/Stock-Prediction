import streamlit as st
from stock_page import stock_page  # Import the function
from home_page import home_page

# Initialize session state
st.session_state.setdefault('page', 'home')
st.session_state.setdefault('stock', None)

# Add custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Page routing
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'analysis':
    stock_page()