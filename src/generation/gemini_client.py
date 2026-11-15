import os
from google import genai
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


def create_gemini_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")

    client = genai.Client(api_key=api_key)

    return client


def generate_answer(client, question, retrieved_chunks):

    # Combine retrieved chunks into context
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    You are a helpful assistant answering questions
    using the provided context.

    Rules:
    1. Answer only using the provided context.
    2. Do not make up information.
    3. If the context does not contain the answer,
       clearly say that the information is not available.
    4. Give clear and concise answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text