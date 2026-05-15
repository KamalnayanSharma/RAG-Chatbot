import google.generativeai as genai

genai.configure(api_key="")

model = genai.GenerativeModel("models/gemini-2.5-flash")

def get_answer(question, context):

    prompt = f"""
    You are a helpful AI assistant.

    Answer the user's question using the provided context.

    If the exact answer is not available,
    give the closest possible answer from the context.

    Context:
    {context}

    Question:
    {question}
    """

    response = model.generate_content(prompt)

    return response.text
