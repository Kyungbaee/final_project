import streamlit as st

st.set_page_config(
    page_title="Seoul_Landmark",
    page_icon="🏢",
    layout="wide",
)


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




st.sidebar.markdown("# Seoul Landmark Dataset ")

st.sidebar.markdown('# Final Project')
st.markdown("### 랜드마크 건설! 🏢")
st.markdown("**초등학생**을 대상으로 한 랜드마크 **딥러닝** 이미지 분류")
st.info(
    '### 원본 출처: [Landmark Images Dataset link](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=56)')
