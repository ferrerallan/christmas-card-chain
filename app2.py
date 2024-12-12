import os
from dotenv import load_dotenv
import streamlit as st
from fpdf import FPDF
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.title("🎅 Christmas Card Assistant with LangChain")

# Collect user inputs
name = st.text_input("Recipient's Name:")
relation = st.text_input("Your Relationship with the Recipient:")
hobbies = st.text_input("Recipient's Hobbies or Interests:")
tone = st.selectbox("Tone of the Message:", ["Warm", "Funny", "Formal", "Heartfelt"])
region = st.text_input("Recipient's Region or Country:")

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

# Define LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Step 1: Generate Base Message Prompt
base_message_prompt = PromptTemplate(
    input_variables=["name", "relation", "hobbies", "tone"],
    template="""
    Create a personalized Christmas message:
    - Recipient's name: {name}
    - Relationship with sender: {relation}
    - Recipient's hobbies or preferences: {hobbies}
    - Desired tone: {tone}
    """
)
base_message_chain = LLMChain(llm=llm, prompt=base_message_prompt, output_key="base_message")

# Step 2: Enrich Message Prompt
enrich_message_prompt = PromptTemplate(
    input_variables=["base_message", "region"],
    template="""
    Based on the following Christmas message:
    "{base_message}"
    
    Enrich it with cultural or Christmas traditions related to {region}.
    Add interesting or heartfelt details about how Christmas is celebrated in this region.

    Return the final enriched message, combining the base message and enriched details.
    """
)
enrich_message_chain = LLMChain(llm=llm, prompt=enrich_message_prompt, output_key="final_message")

# Combine into a SequentialChain
overall_chain = SequentialChain(
    chains=[base_message_chain, enrich_message_chain],
    input_variables=["name", "relation", "hobbies", "tone", "region"],
    output_variables=["final_message"],
    verbose=True
)

# Generate and display results
if st.button("Generate Christmas Card"):
    if not name or not relation or not region:
        st.error("Please fill in all required fields.")
    else:
        # Prepare input
        inputs = {
            "name": name,
            "relation": relation,
            "hobbies": hobbies,
            "tone": tone,
            "region": region,
        }

        with st.spinner("Processing..."):
            try:
                # Run the chain
                final_message = overall_chain.run(inputs)  # Retorna diretamente uma string

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
