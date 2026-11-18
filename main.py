import os
import io
import csv
import base64
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from docx import Document

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper

load_dotenv()

DEFAULT_INSTRUCTION = """
You are a sarcastic calorie tracker.
Adhere to scientifically correct information when the user asks for advice, but
roast the users to your heart's content. Be frank, evil, ruthless but helpful.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    timeout=20,
    max_retries=2,
)
serpapi = SerpAPIWrapper()


def extract_text(msg) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# File attachment handling (ported from agent.py's file_to_content_block)
# ---------------------------------------------------------------------------

IMAGE_MIME_BY_EXT = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}


def extract_pdf_text(raw_bytes: bytes, max_chars: int = 8000) -> str:
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages).strip()
        return text[:max_chars] if text else "[PDF appears to be scanned/image-based — no extractable text]"
    except Exception as e:
        return f"[Failed to read PDF: {e}]"


def extract_docx_text(raw_bytes: bytes, max_chars: int = 8000) -> str:
    try:
        doc = Document(io.BytesIO(raw_bytes))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return text[:max_chars]
    except Exception as e:
        return f"[Failed to read Word document: {e}]"


def extract_csv_text(raw_bytes: bytes, max_rows: int = 200) -> str:
    try:
        decoded = raw_bytes.decode("utf-8", errors="ignore")
        reader = csv.reader(io.StringIO(decoded))
        rows = list(reader)[:max_rows]
        return "\n".join(", ".join(row) for row in rows)
    except Exception as e:
        return f"[Failed to read CSV: {e}]"


def extract_txt_text(raw_bytes: bytes, max_chars: int = 8000) -> str:
    try:
        return raw_bytes.decode("utf-8", errors="ignore")[:max_chars]
    except Exception as e:
        return f"[Failed to read text file: {e}]"


async def upload_to_content_block(file: UploadFile) -> dict:
    raw_bytes = await file.read()
    filename = file.filename or "attachment"
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext in IMAGE_MIME_BY_EXT:
        b64 = base64.b64encode(raw_bytes).decode("utf-8")
        return {"type": "image_url", "image_url": f"data:{IMAGE_MIME_BY_EXT[ext]};base64,{b64}"}
    elif ext == "pdf":
        return {"type": "text", "text": f"[Attached PDF: {filename}]\n{extract_pdf_text(raw_bytes)}"}
    elif ext == "docx":
        return {"type": "text", "text": f"[Attached Word doc: {filename}]\n{extract_docx_text(raw_bytes)}"}
    elif ext == "csv":
        return {"type": "text", "text": f"[Attached CSV: {filename}]\n{extract_csv_text(raw_bytes)}"}
    elif ext in ("txt", "md"):
        return {"type": "text", "text": f"[Attached text file: {filename}]\n{extract_txt_text(raw_bytes)}"}
    else:
        return {"type": "text", "text": f"[Attached file: {filename} — unsupported type, skipped]"}


class AgentState(TypedDict):
    user_context: str
    search_required: str
    model_instruction: str
    search_results: str
    messages: Annotated[list[AnyMessage], add_messages]


def search_necessity(state: AgentState):
    last_msg = state["messages"][-1]
    query_text = extract_text(last_msg) if isinstance(last_msg, HumanMessage) else ""
    prompt = (
        f"The user asked: {query_text}\n"
        "Determine if this requires additional knowledge from the internet. "
        "If you can answer confidently without it, say no. "
        "Answer 'yes' if a search is required, 'no' otherwise."
    )
    response = llm.invoke(prompt)
    response_text = extract_text(response).lower()
    return {"search_required": "yes" if "yes" in response_text else "no"}


def route_search(state: AgentState):
    return "yes" if state["search_required"] == "yes" else "no"


def search_context(state: AgentState):
    last_msg = state["messages"][-1]
    query = extract_text(last_msg).strip() if isinstance(last_msg, HumanMessage) else ""
    if not query:
        return {"search_results": ""}
    try:
        raw_results = serpapi.run(query)
    except Exception:
        raw_results = ""
    return {"search_results": raw_results}


def agent_chat(state: AgentState):
    search_results = state.get("search_results", "")
    grounding_note = ""
    if state["search_required"] == "yes" and search_results:
        grounding_note = (
            "\n\nUse the following current web search results to verify facts "
            "before answering. If they contradict your prior knowledge, prioritize them. "
            "Only cite them if actually relevant to the question:\n"
            f"{search_results[:1500]}"
        )

    context_suffix = f"\n\n[Context: {state['user_context']}]"
    system_message = SystemMessage(content=state["model_instruction"] + grounding_note + context_suffix)
    payload = [system_message] + state["messages"]

    try:
        agent_res = llm.invoke(payload)
    except Exception as e:
        agent_res = AIMessage(content=f"The model is currently unavailable: {e}")

    return {"messages": [agent_res]}


builder = StateGraph(AgentState)
builder.add_node("search_necessity", search_necessity)
builder.add_node("search_context", search_context)
builder.add_node("agent_chat", agent_chat)
builder.add_edge(START, "search_necessity")
builder.add_conditional_edges(
    "search_necessity", route_search, {"yes": "search_context", "no": "agent_chat"}
)
builder.add_edge("search_context", "agent_chat")
builder.add_edge("agent_chat", END)

memory = MemorySaver()
graph_app = builder.compile(checkpointer=memory)

# ---------------------------------------------------------------------------
# Web API
# ---------------------------------------------------------------------------

app = FastAPI()

# TODO: once deployed, replace "*" with your actual Vercel URL for security,
# e.g. ["https://your-gym-app.vercel.app", "http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "https://nyxzlair.vercel.app",
    "http://localhost:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(
    message: str = Form(...),
    session_id: str = Form("default"),
    calorie_context: str = Form(""),
    files: List[UploadFile] = File(default=[]),
):
    content_blocks = [{"type": "text", "text": message}]
    for f in files:
        content_blocks.append(await upload_to_content_block(f))

    # Keep plain string content when there are no attachments (simpler for the LLM),
    # otherwise send the multimodal content block list.
    human_content = message if len(content_blocks) == 1 else content_blocks

    result = graph_app.invoke(
        {
            "user_context": calorie_context or "No calorie log provided yet.",
            "model_instruction": DEFAULT_INSTRUCTION,
            "messages": [HumanMessage(content=human_content)],
        },
        {"configurable": {"thread_id": session_id}},
    )
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    reply = extract_text(ai_messages[-1]) if ai_messages else "No response generated."
    return {"reply": reply}
