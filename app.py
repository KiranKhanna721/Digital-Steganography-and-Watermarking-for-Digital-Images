import streamlit as st
import app1
import app2
PAGES = {
    
    "Steganography" :app1 ,
    "Watermarking" :app2
}
st.sidebar.title('Steganography and Watermarking for Digital Images')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()