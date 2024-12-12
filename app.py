import os
from dotenv import load_dotenv
import streamlit as st
from fpdf import FPDF
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.llms.base import LLM
from pydantic import BaseModel
import requests
from typing import Optional, List


# Load environment variables
load_dotenv()

# Normalize text for PDF compatibility
def normalize_text(text):
    import unicodedata
    return unicodedata.normalize("NFKD", text).encode("latin-1", "ignore").decode("latin-1")

# PDF Generation Function
def generate_pdf(message, recipient_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, txt=f"Christmas Card for {recipient_name}", ln=True, align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 10, txt=normalize_text(message))
    return pdf.output(dest='S').encode('latin-1')


# Custom LLM Wrapper for Azure OpenAI
class AzureCustomLLM(LLM, BaseModel):
    endpoint: str
    api_key: str
    model: str
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 300

    @property
    def _llm_type(self) -> str:
        return "azure_custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Você é um assistente útil."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens
        }

        response = requests.post(self.endpoint, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Erro na API Azure: {response.status_code}, {response.text}")

    @property
    def identifying_params(self) -> dict:
        """
        Retorna os parâmetros que identificam a configuração do LLM.
        """
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }


# Azure Custom LLM Configuration
azure_custom_llm = AzureCustomLLM(
    endpoint="https://ai-ferrerallanhubprd164442737391.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview",
    api_key="4sWNd6hxPcqyx1ajv0n0VVsL5N3eWCHpdhvW0DkWckrYV4zIqnnAJQQJ99ALACYeBjFXJ3w3AAAAACOGNgn9",
    model="Ministral-3B",
    temperature=0.7,
    top_p=0.95
)

# Streamlit UI
st.title("🎅 Christmas Card Assistant with LangChain")

# Collect user inputs
sender_name = st.text_input("Your Name (Sender):")
name = st.text_input("Recipient's Name:")
relation = st.text_input("Your Relationship with the Recipient:")
hobbies = st.text_input("Recipient's Hobbies or Interests:")
tone = st.selectbox("Tone of the Message:", ["Warm", "Funny", "Formal", "Heartfelt"])
region = st.text_input("Recipient's Region or Country:")

# Define Prompts
# Step 1: Generate Base Message Prompt
base_message_prompt = PromptTemplate(
    input_variables=["name", "relation", "hobbies", "tone"],
    template="""
    Create a personalized Christmas message:
    - Recipient's name: {name}
    - Relationship with sender: {relation}
    - Recipient's hobbies or preferences: {hobbies}
    - Desired tone: {tone}

    Example:
    Dear {name},
    As we approach the most wonderful time of the year, I wanted to take a moment to wish you a Merry Christmas!
    Since you are my {relation}, I know how much you enjoy {hobbies}, and I hope this holiday season brings you joy and happiness.
    """
)

base_message_chain = LLMChain(llm=azure_custom_llm, prompt=base_message_prompt, output_key="base_message")

# Step 2: Enrich Message Prompt
enrich_message_prompt = PromptTemplate(
    input_variables=["base_message", "region"],
    template="""
    Based on the following Christmas message:
    "{base_message}"
    
    Enrich it with cultural or Christmas traditions related to {region}.
    Mention specific holiday activities or traditions unique to this region to make the message more personal and heartfelt.
    Return the final enriched message.
    """
)

enrich_message_chain = LLMChain(llm=azure_custom_llm, prompt=enrich_message_prompt, output_key="final_message")

# Combine into a SequentialChain
overall_chain = SequentialChain(
    chains=[base_message_chain, enrich_message_chain],
    input_variables=["name", "relation", "hobbies", "tone", "region"],
    output_variables=["final_message"],
    verbose=True
)

# Generate and display results
if st.button("Generate Christmas Card"):
    if not name.strip():
        st.error("Please provide the recipient's name.")
    elif not relation.strip():
        st.error("Please describe your relationship with the recipient.")
    elif not region.strip():
        st.error("Please specify the recipient's region or country.")
    elif not sender_name.strip():
        st.error("Please provide your name (sender).")
    else:
        # Inputs estão validados
        inputs = {
            "name": name,
            "relation": relation,
            "hobbies": hobbies or "their favorite activities",
            "tone": tone,
            "region": region,
        }

        with st.spinner("Processing..."):
            try:
                # Run the chain
                final_message = overall_chain.run(inputs)

                # Substituir o placeholder [Your Name]
                final_message = final_message.replace("[Your Name]", sender_name)

                # Display the final message
                st.subheader("Final Enriched Message")
                st.write(final_message)

                # Generate the PDF
                pdf_file = generate_pdf(final_message, name)

                st.success("Christmas card created successfully!")

                # Download the PDF
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_file,
                    file_name=f"christmas_card_{name}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")
