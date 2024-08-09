import streamlit as st

x = st.slider('Select a value', 8, 90)
st.write(x, 'squared is', x * x)

st.button("Reset", type="primary")
if st.button("Say hello"):
   st.write("Why hello there")
else:
      st.write("Goodbye")
