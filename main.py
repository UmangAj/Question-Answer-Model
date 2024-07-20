from transformers import T5ForConditionalGeneration, T5Tokenizer
import streamlit as st
from pdfReader import read_pdf

# Load pre-trained model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)

st.sidebar.write("### Select one Option")

navigation = st.sidebar.radio("", ["Upload_File", "Write_Paragraph"], label_visibility="collapsed")

st.markdown("<h2 style='text-align: center'>Question - Answer Model</h2>", unsafe_allow_html=True)

if navigation == "Upload_File":

    st.write("##### Upload your PDF file :-")

    pdf = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    pdf_text = ""
    if pdf is not None:
        pdf_text = read_pdf(pdf)

        st.write("###### Uploaded PDF Content")
        with st.expander("Click here"):
            st.write(pdf_text)
    st.write("##### Enter a Question :-")
    question1 = st.text_input("", label_visibility="collapsed", key="question1")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Get Answer"):
        if pdf is None:
            st.error("Please upload a PDF file.")
        elif not question1:
            st.error("Please enter a question.")
        else:
            input_text = f"question: {question1} context: {pdf_text}"
            inputs = tokenizer.encode(input_text, return_tensors="pt")

            # Generate the answer
            outputs = model.generate(inputs, max_new_tokens=50)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write("##### Your answer is :-")
            st.text_area("", answer, height=100, key="answer_output_1", label_visibility="collapsed")

if navigation == "Write_Paragraph":

    st.write("##### Enter a Paragraph :-")
    write_context = st.text_area("", label_visibility="collapsed", height=250)
    st.write("##### Enter a Question :-")
    question2 = st.text_input("", label_visibility="collapsed", key="question2")
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Get Answer"):
        if not write_context:
            st.error("Please enter a paragraph.")

        elif not question2:
            st.error("Please enter a question.")

        else:
            input_text = f"question: {question2} context: {write_context}"
            inputs = tokenizer.encode(input_text, return_tensors="pt")

            # Generate the answer
            outputs = model.generate(inputs, max_new_tokens=50)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write("##### Your answer is :-")
            st.text_area("", answer, height=100, key="answer_output_2", label_visibility="collapsed")












