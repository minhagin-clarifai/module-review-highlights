#### imports
import streamlit as st

import pandas as pd

from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai_grpc.grpc.api import resources_pb2, service_pb2

from langchain.llms import Clarifai
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.summarize import load_summarize_chain


#### helper functions
def get_langchain_summary(list_of_text, pat, user_id, app_id, model_id):
  llm = Clarifai(pat=pat, user_id=user_id, app_id=app_id, model_id=model_id)
  summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
  summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
  text_summary = summarize_document_chain.run(list_of_text)
  return text_summary


#### streamlit config
st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)


#### clarifai credentials/config
auth = ClarifaiAuthHelper.from_streamlit(st)
pat = auth._pat
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()


#### llama2-13b-chat credentials
user_id = 'meta'
app_id = 'Llama-2'
model_id = 'llama2-13b-chat'
version_id = '79a1af31aa8249a99602fc05687e8f40'


#### title / description
st.image('resources/clarifai-logo-full.png', width=200)
st.title(":sparkles: Highlights from Product Reviews :sparkles:")
st.write("Using reviews from this 'Gorilla Ladders - 3-Step pro-grade Steel Step Stool', we'll show how you can leverage LLMs to produce a list of noteable characteristics pulled from real user reviews.")
st.write("In this demo, we're using one of the Llama2 models! :llama::llama: https://clarifai.com/meta/Llama-2/models/llama2-13b-chat")
st.divider()


#### present product information
st.write("So here is a our lovely example product with some real customer reviews. [Home Depot Product Link](https://www.homedepot.com/p/Gorilla-Ladders-3-Step-Pro-Grade-Steel-Step-Stool-300-lbs-Load-Capacity-Type-IA-Duty-Rating-9ft-Reach-Height-GLHD-3/302159307)")
col1, col2 = st.columns([1,2])

with col1:
  st.subheader('3-Step Pro-Grade Steel Step Stool')
  st.caption('Gorilla Ladders')
  st.image('resources/gorilla-ladders-step-stools-glhd-3-64_1000.png')

with col2:
  st.subheader('Product Reviews')
  st.caption('Sourced from Home Depot on August 29, 2023')
  df = pd.read_csv('resources/3-step-pro-grade-step-stool-reviews.csv')
  st.dataframe(df, hide_index=True, column_config={'time': None})

st.divider()


#### do llm - summarization
st.subheader('Summarization using LLMs / Clarifai / Langchain')
st.write("Here we'll use langchain's summzarization functionality to pass all of the reviews to the underlying LLM, llama2-13b-chat in this case, to provide a summarization of the different reviews.")
st.write(":parrot::link: By the way, we're integrated! https://python.langchain.com/docs/integrations/providers/clarifai")
st.write('')

list_of_text = '\n'.join(df.text.to_list())

if 'clicked1' not in st.session_state:
  st.session_state.clicked1 = {1:False,2:False}

def clicked1(button):
  st.session_state.clicked1[button] = True

if st.button('Summarize reviews', on_click=clicked1, args=[1]):
  if 'text_summary' not in st.session_state:
  
    with st.spinner('Getting summarization...'):
      text_summary = get_langchain_summary(list_of_text, pat, user_id, app_id, model_id)
      st.session_state.text_summary = text_summary

if st.session_state.clicked1[1]:
  with st.expander('Text Summary', expanded=True):
    st.write(st.session_state.text_summary)

st.divider()


#### do llm - highlights / lowlights
st.subheader('Use LLMs to create highlights and lowlights')
st.write('Using Clarifai and some clever prompt engineering to produce a list of noteable noteables.')

text_prompt = """
<s>[INST] <<SYS>>
From the product reviews provided, please return a "highlight" list and a "lowlight" list of tags, one describing the highlights of the product and one for the lowlights. Do not provide explanations.
<</SYS>>
{}
[/INST]</s>
"""

n = 20

if 'clicked2' not in st.session_state:
  st.session_state.clicked2 = {1:False,2:False}

def clicked2(button):
  st.session_state.clicked2[button] = True

if st.button('Get tags', on_click=clicked2, args=[1]):
  if 'highlights' not in st.session_state:
  
    with st.spinner('Reticulating splines...'):
      res_pmo = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
          user_app_id = resources_pb2.UserAppIDSet(
            user_id = user_id,
            app_id = app_id
          ),
          model_id = model_id,
          version_id = version_id,
          inputs = [
            resources_pb2.Input(
              data = resources_pb2.Data(
                text = resources_pb2.Text(
                  raw = text_prompt.format(
                    '\n'.join(df.text.to_list()[:n])
                  )
                )
              )
            )
          ]
        )
      )

      st.session_state.highlights = res_pmo.outputs[0].data.text.raw

if st.session_state.clicked2[1]:
  with st.expander('Highlights and Lowlights', expanded=True):
    st.write(st.session_state.highlights)

st.divider()

#### final call to action
st.write('Psst..  Go here to schedule a demo! https://www.clarifai.com/company/schedule-demo')
