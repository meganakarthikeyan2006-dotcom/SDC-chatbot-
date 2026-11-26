# The chatbot. Owners: Engineer 2 (retrieve + answer), Engineer 3 (Gradio + deploy)
# Step 5: Retrieve the 5 closest passages
# Step 6: Send question + passages to Gemini, return answer with sources
# Step 7: Gradio chat window
#
# Try it from the terminal:
#   python app.py "Can I get accessible housing?"
# Shows the passages found. Also asks Gemini if GEMINI_API_KEY is set: on a laptop in
# .env (copy .env.example), in Colab Secrets, or in Hugging Face Space Secrets. Never in code.

import os
import sys

from dotenv import load_dotenv

load_dotenv()  # reads GEMINI_API_KEY from .env if the file exists; does nothing otherwise

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = "chroma_db"                                    # built by ingest.py
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # MUST match ingest.py
TOP_K = 5
GEMINI_MODEL = "gemini-3.5-flash-lite"                     # change here if the team picks anothe
SDC_CONTACT = "the Student Disability Center (sdc.ucdavis.edu)"

_db = None


def get_db():
    """Open the search library once and reuse it for every question."""
    global _db
    if _db is None:
        if not os.path.isdir(DB_DIR):
            raise SystemExit(f"No {DB_DIR}/ found. Run ingest.py first.")
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        _db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return _db


# Step 5: Retrieve the 5 closest passages
def retrieve(question, k=TOP_K):
    """Return the k passages closest in meaning to the question.
    Each item is (passage, distance); a smaller distance means a closer match."""
    return get_db().similarity_search_with_score(question, k=k)


def format_passages(passages):
    """Number each passage and label it with its title and link, for the prompt."""
    blocks = []
    for i, (doc, _distance) in enumerate(passages, start=1):
        title = doc.metadata.get("title", "Untitled")
        source = doc.metadata.get("source", "unknown source")
        blocks.append(f"[{i}] {title} ({source})\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)


def build_prompt(question, passages):
    """The instructions Gemini follows. The rules are what keep answers trustworthy."""
    return f"""You answer UC Davis students' questions about disability accommodations.

Rules:
- Use ONLY the passages below. Do not use outside knowledge.
- After each fact, cite the passage number it came from, like [1] or [2].
- If the passages do not contain the answer, say "I don't know" and suggest contacting {SDC_CONTACT}.
- Do not give personal advice or decide anything about a specific student's case.
- Keep the answer short and plain.

Passages:
{format_passages(passages)}

Question: {question}
Answer:"""


def list_sources(passages):
    """Unique title + link for each passage used, in order, for showing under the answer."""
    seen, sources = set(), []
    for doc, _distance in passages:
        key = (doc.metadata.get("title", "Untitled"), doc.metadata.get("source", ""))
        if key not in seen:
            seen.add(key)
            sources.append(key)
    return sources


# Step 6: Send question + passages to Gemini, return answer with sources
def answer(question):
    """Return {"answer": text, "sources": [(title, link), ...]}. Step 7 calls this."""
    passages = retrieve(question)
    if not passages:
        return {"answer": f"I don't know. Please contact {SDC_CONTACT}.", "sources": []}

    from google import genai  # imported here so retrieval works without the Gemini library set up
    client = genai.Client()   # reads GEMINI_API_KEY from the environment
    response = client.models.generate_content(model=GEMINI_MODEL,
                                              contents=build_prompt(question, passages))
    return {"answer": response.text.strip(), "sources": list_sources(passages)}






# Step 7: Gradio chat window
import gradio as gr

def chat_response(message, history):
    result = answer(message)
    response = result["answer"]

    if result["sources"]:
        response += "\n\n**Retrieved Sources:**\n"
        for title, link in result["sources"]:
            response += f"\n- [{title}]({link})"

    return response

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.amber,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate
).set(
    body_background_fill="#071D36",
    body_background_fill_dark="#071D36",
    body_text_color="#FFFFFF",
    body_text_color_dark="#FFFFFF",
    block_background_fill="#102F50",
    block_background_fill_dark="#102F50",
    block_border_color="#365675",
    block_border_color_dark="#365675",
    block_label_text_color="#FFDF80",
    block_label_text_color_dark="#FFDF80",
    button_secondary_background_fill="#1A4065",
    button_secondary_background_fill_dark="#1A4065",
    button_secondary_text_color="#FFFFFF",
    button_secondary_text_color_dark="#FFFFFF"
)

custom_css = """
body, .gradio-container {
    background: #071D36 !important;
    color: #FFFFFF !important;
}

.gradio-container {
    width: 100% !important;
    max-width: 800px !important;
    margin: 0 auto !important;
    padding: 20px !important;
    box-sizing: border-box !important;
    gap: 8px !important;
}

.gradio-container .row,
.gradio-container .column {
    min-width: 0 !important;
    max-width: 100% !important;
}

#aggie-header {
    background: linear-gradient(135deg, #022851, #164A75);
    padding: 20px;
    border-radius: 14px;
    border-bottom: 4px solid #FFBF00;
}

#aggie-header h1 {
    color: #FFFFFF !important;
    font-size: 34px;
    margin: 0;
}

#aggie-header p {
    color: #FFDF80 !important;
    font-size: 15px;
    margin: 5px 0;
}

#aggie-header svg {
    width: 55px;
    height: 55px;
}

img#campus-banner {
    display: block !important;
    width: 100% !important;
    height: 150px !important;
    object-fit: cover !important;
    object-position: center 65%;
    border-radius: 12px;
}

#welcome {
    text-align: center;
    padding: 4px !important;
}

#welcome h2 {
    color: #FFFFFF !important;
    font-size: 28px;
    margin: 5px 0;
}

#welcome p {
    color: #C4D5E8 !important;
    margin: 5px 0;
}

#aggie-chat {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
    background: #102F50 !important;
    border: 1px solid #365675 !important;
    border-radius: 16px !important;
}

#aggie-chat button {
    background: #1A4065 !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
}

#aggie-chat button:hover {
    background: #285780 !important;
}

#aggie-chat .message,
#aggie-chat .prose,
#aggie-chat p,
#aggie-chat a {
    min-width: 0 !important;
    max-width: 100% !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
}

#aggie-chat .examples {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 12px !important;
}

textarea {
    background: #102F50 !important;
    color: #FFFFFF !important;
    font-size: 16px !important;
}

textarea::placeholder {
    color: #B5C8DE !important;
    opacity: 1;
}

#privacy {
    background: #102F50;
    border: 1px solid #365675;
    border-radius: 12px;
    text-align: center;
    padding: 10px;
    font-size: 13px;
    color: #C4D5E8 !important;
}

#privacy strong {
    color: #FFDF80 !important;
}

button:focus-visible,
textarea:focus-visible {
    outline: 3px solid #FFBF00 !important;
    outline-offset: 3px;
}

footer {
    display: none !important;
}

@media (max-width: 500px) {
    #aggie-chat .examples {
        grid-template-columns: 1fr !important;
    }
}
"""

with gr.Blocks(title="AggieAccess", fill_width=True) as demo:

    gr.HTML("""
    <div id="aggie-header">
        <div style="display:flex;align-items:center;gap:18px;">
            <svg width="70" height="70" viewBox="0 0 100 100"
                 aria-hidden="true" focusable="false">
                <rect width="100" height="100" rx="20" fill="#FFBF00"/>
                <circle cx="50" cy="50" r="37"
                        fill="none" stroke="#022851" stroke-width="3"/>
                <circle cx="50" cy="27" r="7" fill="#022851"/>
                <path d="M26 40 L50 46 L74 40 M50 46 L50 63
                         M50 61 L35 80 M50 61 L65 80"
                      stroke="#022851" stroke-width="6"
                      stroke-linecap="round" fill="none"/>
            </svg>
            <h1>AggieAccess</h1>
        </div>
        <p>UC Davis | Student Accessibility Assistant</p>
    </div>
    """)

    gr.HTML("""
    <img id="campus-banner"
         src="/gradio_api/file=campus.jpg"
         alt="Tree-lined walkways and green lawns at UC Davis">

    <div id="welcome">
        <h2>Welcome, Aggies!</h2>
        <p>Your guide to a more accessible campus.</p>
        <p>Explore accommodations, student services, and accessibility resources at UC Davis.</p>
    </div>
    """)

    chat_interface = chat_interface = chat_interface = gr.ChatInterface(
        fn=chat_response,
        chatbot=gr.Chatbot(
            height=260,
            autoscroll=False,
            elem_id="aggie-chat",
            label="AggieAccess Assistant"
        ),
        textbox=gr.Textbox(
            placeholder="Ask AggieAccess a question...",
            container=False,
            submit_btn="Send"
        ),
        examples=[
            "How do I register with the SDC?",
            "How can I request exam accommodations?",
            "Can I get accessible housing?",
            "What accessibility resources does UC Davis offer?"
        ],
        cache_examples=False
    )

    chat_interface.chatbot.clear(
        fn=lambda: "",
        inputs=None,
        outputs=chat_interface.textbox
    )

    chat_interface.chatbot.clear(
        fn=lambda: "",
        inputs=None,
        outputs=chat_interface.textbox
    )

    chat_interface.chatbot.clear(
        fn=lambda: "",
        inputs=None,
        outputs=chat_interface.textbox
    )

    gr.HTML("""
    <div id="privacy">
        <strong>Privacy Notice:</strong>
        Please do not enter personal or medical information.
        <br>
        Student-developed project | Not an official SDC service.
    </div>
    """)

# allows me to test from terminal
if __name__ == "__main__" and len(sys.argv) == 1:
    demo.launch(theme=theme, css=custom_css, allowed_paths=["campus.jpg"])

elif __name__ == "__main__" and len(sys.argv) == 1:
    demo.launch(theme=theme, css=custom_css, allowed_paths=["campus.jpg"])

elif __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Can I get accessible housing?"
    print(f"Question: {question}\n")
    print("Passages found (smaller distance = closer match):")
    for doc, distance in retrieve(question):
        snippet = " ".join(doc.page_content.split())[:120]
        print(f"  {distance:.3f}  {doc.metadata.get('title', 'Untitled')}: {snippet}...")

    key = os.environ.get("GEMINI_API_KEY", "")
    if key and key != "paste-your-key-here":
        result = answer(question)
        print("\nAnswer:\n" + result["answer"])
        print("\nSources:")
        for title, link in result["sources"]:
            print(f"  - {title}: {link}")
    else:
        print("\n(Put your key in .env to also get Gemini's answer.)")
