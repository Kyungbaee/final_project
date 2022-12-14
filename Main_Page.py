import streamlit as st
# from tkinter.tix import COLUMN
from pyparsing import empty

st.set_page_config(
    page_title="Seoul_Landmark",
    page_icon="π’",
    layout="wide",
)

empty1,con1 = st.columns([0.3,1.0])
empty1,con2,con3 = st.columns([0.3,0.5,0.5])

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2018/07/05/16/59/students-3518726_960_720.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

def main() :

    with empty1 :
        empty() # μ¬λ°±λΆλΆ1
        st.sidebar.markdown("# Seoul_Landmark Dataset ")
        st.sidebar.markdown('# Final Project')
        
    with con1 :
        st.markdown("### λλλ§ν¬ κ±΄μ€! π’")
        st.markdown("**μ΄λ±νμ**μ λμμΌλ‘ ν λλλ§ν¬ **λ₯λ¬λ** μ΄λ―Έμ§ λΆλ₯")
    
    with con2 :
        st.info(
        '### μλ³Έ μΆμ²: [Landmark Images Dataset link](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=56)')
    # with con3 :
      
main()







