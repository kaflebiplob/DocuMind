import os

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "groq")  

OLLAMA_MODEL = "llama3.2"
GEMINI_MODEL = "gemini-2.0-flash"
GROQ_MODEL = "llama-3.1-8b-instant"


def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    return f"""Answer the question using only the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}"""


def _ollama_answer(prompt: str) -> str:
    import ollama

    response = ollama.chat(
        model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


def _gemini_answer(prompt: str) -> str:
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text


def _groq_answer(prompt: str) -> str:
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def generate_answer(question: str, chunks: list[str]) -> str:
    prompt = build_prompt(question, chunks)

    if PROVIDER == "ollama":
        return _ollama_answer(prompt)
    if PROVIDER == "gemini":
        return _gemini_answer(prompt)
    if PROVIDER == "groq":
        return _groq_answer(prompt)

    raise ValueError(f"Unknown LLM_PROVIDER: {PROVIDER}")
