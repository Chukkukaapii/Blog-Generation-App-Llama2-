import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

# llama2 response function
def getllamaresponse(input_text, no_words, blog_style):
    st.write("Creating CTransformers instance...")
    try:
        llm = CTransformers(model='MODELS/llama-2-7b-chat.ggmlv3.q8_0.bin',
                            model_type='llama',
                            config={'max_new_tokens': 256, 'temperature': 0.01})
        st.write("CTransformers instance created.")
    except Exception as e:
        st.write(f"Error creating CTransformers instance: {e}")
        return None

    # prompt template
    template = """
    write a blog for {style} profile for a topic {text} with {n_words} words.
    """
    prompts = [template.format(style=blog_style, text=input_text, n_words=no_words)]

    st.write("Generating response...")
    try:
        # generate responses
        response = llm.generate(prompts)
        st.write("Response generated.")
        return response
    except Exception as e:
        st.write(f"Error generating response: {e}")
        return None

st.set_page_config(page_title="generate the blog",
                   page_icon='<3',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.title("Generate the Blog")

input_text = st.text_input("Enter the topic")

col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('Number of words')

with col2:
    blog_style = st.selectbox('Writing the blog for',
                              ('researchers', "data scientist",
                               "common people"), index=0)

submit = st.button("Generate")

# final response
if submit:
    st.write("Generating blog...")
    response = getllamaresponse(input_text, no_words, blog_style)
    if response:
        st.write("Blog generated.")
        st.write(response)

