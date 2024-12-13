
# Christmas Card Assistant with LangChain

## Description

The **Christmas Card Assistant** is a Streamlit-based web application designed to create personalized and culturally enriched Christmas messages. Using advanced AI models powered by OpenAI and Azure, this tool generates heartfelt messages tailored to the recipient's details, such as their name, relationship with the sender, hobbies, and regional traditions. Additionally, the application can generate downloadable PDF Christmas cards with the customized message.

This project showcases a seamless integration of LangChain's chaining capabilities, custom LLM services, and PDF generation, making it a great example of AI-driven personalized content creation.

## Requirements

To run this application, you will need:

- Python 3.8 or higher
- Environment variables configured for:
  - OpenAI API key (`OPENAI_API_KEY`)
  - Azure API endpoint (`AZURE_ENDPOINT`)
  - Azure API key (`AZURE_API_KEY`)
- Required Python packages:
  - `streamlit`
  - `fpdf`
  - `langchain`
  - `python-dotenv`
  - `requests`

Install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```

## Mode of Use

1. Clone the repository:
   ```bash
   git clone https://github.com/ferrerallan/christmas-card-chain.git
   ```
2. Navigate to the project directory:
   ```bash
   cd christmas-card-chain
   ```
3. Create a `.env` file in the root directory and define the following environment variables:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_MODEL=gpt-3.5-turbo
   OPENAI_TEMPERATURE=0.7
   OPENAI_MAX_TOKENS=300

   AZURE_API_KEY=your_azure_api_key
   AZURE_ENDPOINT=your_azure_endpoint
   AZURE_MODEL=your_azure_model_name
   AZURE_TEMPERATURE=0.7
   AZURE_TOP_P=0.95
   AZURE_ENRICHER_MAX_TOKENS=300
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```
5. Open the provided URL in your browser to access the web interface.

## Features

- **Personalized Messages**: Generate Christmas messages tailored to the recipient's name, hobbies, relationship, and desired tone.
- **Cultural Enrichment**: Incorporate regional traditions and customs into the message for a personalized touch.
- **PDF Generation**: Convert the message into a downloadable PDF formatted as a Christmas card.
- **User-Friendly Interface**: Intuitive web interface built with Streamlit.

## Example Usage

1. Open the application in your browser.
2. Fill in the required fields:
   - Your name (sender)
   - Recipient's name
   - Relationship with the recipient
   - Hobbies or interests
   - Tone of the message
   - Recipient's region or country
3. Click "Generate Christmas Card" to create a message.
4. Review the message, download the PDF, and share it with your loved one.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
