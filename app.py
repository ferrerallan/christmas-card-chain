import os
from dotenv import load_dotenv
import streamlit as st
from fpdf import FPDF
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import LLM
from pydantic import BaseModel
import requests
from typing import Optional, List, ClassVar

load_dotenv(dotenv_path=".env")

class AzureLLMService(LLM, BaseModel):
    endpoint: str
    api_key: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int

    @property
    def _llm_type(self) -> str:
        return "azure_custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Content-Type": "application/json", "api-key": self.api_key}
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Você é um assistente útil."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        response = requests.post(self.endpoint, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        raise ValueError(f"API Error: {response.status_code}, {response.text}")

class PDFService:
    @staticmethod
    def normalize_text(text):
        import unicodedata
        return unicodedata.normalize("NFKD", text).encode("latin-1", "ignore").decode("latin-1")

    @staticmethod
    def generate_pdf(message: str, recipient_name: str):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(200, 10, txt=f"Christmas Card for {recipient_name}", ln=True, align="C")
        pdf.ln(10)
        pdf.multi_cell(0, 10, txt=PDFService.normalize_text(message))
        return pdf.output(dest="S").encode("latin-1")

class PDFChain(Chain):
    input_keys: ClassVar[List[str]] = ["final_message", "name"]
    output_keys: ClassVar[List[str]] = ["pdf_file"]

    def _call(self, inputs: dict) -> dict:
        pdf_file = PDFService.generate_pdf(inputs["final_message"], inputs["name"])
        return {"pdf_file": pdf_file}

    @property
    def _chain_type(self) -> str:
        return "pdf_chain"

class LangChainService:
    def __init__(self, base_llm: LLM, enricher_llm: LLM):
        self.pipeline = self._create_pipeline(base_llm, enricher_llm)

    def _create_pipeline(self, base_llm: LLM, enricher_llm: LLM):
        base_message_prompt = PromptTemplate(
            input_variables=["name", "relation", "hobbies", "tone","sender_name"],
            template="""
                Create a personalized Christmas message:
                - Recipient's name: {name}
                - Relationship with sender: {relation}
                - Recipient's hobbies or preferences: {hobbies}
                - Desired tone: {tone}
                - Sender's name: {sender_name}

                Example:
                Dear {name},
                As we approach the most wonderful time of the year, I wanted to take a moment to wish you a Merry Christmas!
                Since you are my {relation}, I know how much you enjoy {hobbies}, and I hope this holiday season brings you joy and happiness.
                            """
        )
        enrich_message_prompt = PromptTemplate(
            input_variables=["base_message", "region"],
            template="""
            Based on the following Christmas message:
            "{base_message}"
                        
            Enrich it with cultural or Christmas traditions related to {region} maintaining the same tone and structure:
            Mention specific holiday activities or traditions unique to this region to make the message more personal and heartfelt.
            Return only the text of final enriched message.
                        """
        )
        base_message_chain = LLMChain(llm=base_llm, prompt=base_message_prompt, output_key="base_message")
        enrich_message_chain = LLMChain(llm=enricher_llm, prompt=enrich_message_prompt, output_key="final_message")
        pdf_chain = PDFChain()
        
        return SequentialChain(
            chains=[base_message_chain, enrich_message_chain, pdf_chain],
            input_variables=["name", "relation", "hobbies", "tone", "region", "sender_name"],
            output_variables=["pdf_file"],
            verbose=True,
        )

    def generate_message(self, inputs: dict) -> bytes:
        return self.pipeline.run(inputs)

def main():
    base_llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", 0.7)),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", 300)),
    )

    enricher_llm = AzureLLMService(
        endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
        model=os.getenv("AZURE_MODEL"),
        temperature=float(os.getenv("AZURE_TEMPERATURE", 0.7)),
        top_p=float(os.getenv("AZURE_TOP_P", 0.95)),
        max_tokens=int(os.getenv("AZURE_ENRICHER_MAX_TOKENS", 300)),
    )

    langchain_service = LangChainService(base_llm=base_llm, enricher_llm=enricher_llm)

    st.title("🎅 Christmas Card Assistant with LangChain")

    sender_name = st.text_input("Your Name (Sender):")
    name = st.text_input("Recipient's Name:")
    relation = st.text_input("Your Relationship with the Recipient:")
    hobbies = st.text_input("Recipient's Hobbies or Interests:")
    tone = st.selectbox("Tone of the Message:", ["Warm", "Funny", "Formal", "Heartfelt"])
    region = st.text_input("Recipient's Region or Country:")

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
            inputs = {
                "name": name,
                "relation": relation,
                "hobbies": hobbies or "their favorite activities",
                "tone": tone,
                "region": region,
                "sender_name": sender_name,
            }
            with st.spinner("Processing..."):
                try:
                    pdf_file = langchain_service.generate_message(inputs)
                    st.success("Christmas card created successfully!")
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_file,
                        file_name=f"christmas_card_{name}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
