import streamlit as st
import requests
import json
from typing import Any, Dict, List

# ---------------------------------------------
# Streamlit Setup
# ---------------------------------------------
st.set_page_config(page_title="GraphRAG Chatbot", layout="wide")

with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("FastAPI /ask URL", value="http://localhost:8000/ask")

    if st.button("Clear Chat"):
        st.session_state.pop("chat", None)
        st.rerun()

    st.markdown("---")
    st.caption("This chatbot streams newline-delimited JSON events from your FastAPI agent.")

st.title("GraphRAG Chatbot")

# ---------------------------------------------
# Initialize session state
# ---------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []


# ---------------------------------------------
# Helper: Safe JSON loads
# ---------------------------------------------
def safe_json_loads(val: str) -> Any:
    try:
        return json.loads(val)
    except Exception:
        return val


# ---------------------------------------------
# Rendering chat history
# ---------------------------------------------
def render_message(msg: Dict[str, Any]):
    role = msg.get("role", "assistant")

    # Pick avatar
    avatar = "🙂" if role == "user" else "💬"

    with st.chat_message(role, avatar=avatar):
        st.write(msg.get("text", ""))

        # --- Tool Calls ---
        if msg.get("tool_calls"):
            with st.expander("🔧 Tool Calls"):
                for i, tc in enumerate(msg["tool_calls"], start=1):
                    name = tc.get("tool_name") or tc.get("name") or f"tool #{i}"
                    st.markdown(f"**{name}**")

                    if tc.get("message"):
                        st.write(tc["message"])

                    if tc.get("args"):
                        st.json(tc["args"])

                    if tc.get("content"):
                        st.json(safe_json_loads(tc["content"]))

                    st.markdown("---")

        # --- Reasoning Steps ---
        if msg.get("reasoning"):
            with st.expander("🧠 Reasoning Trace"):
                for step in msg["reasoning"]:
                    st.markdown(f"- {step}")

        # --- Raw JSON ---
        if msg.get("raw"):
            with st.expander("📦 Raw JSON"):
                st.json(msg["raw"])


def render_history():
    for m in st.session_state.chat:
        render_message(m)


# Render existing history
render_history()


# ---------------------------------------------
# Streaming Logic
# ---------------------------------------------
def stream_response(question: str, api_url: str):
    """
    Stream newline-delimited JSON from FastAPI and progressively update UI.
    """
    # Display user message immediately
    st.session_state.chat.append({"role": "user", "text": question})
    st.chat_message("user", avatar="🙂").write(question)

    # Placeholder for streaming assistant output
    assistant_container = st.chat_message("assistant", avatar="💬")
    placeholder = assistant_container.empty()

    final_text = ""
    reasoning: List[str] = []
    tool_calls: List[Dict] = []
    raw_final = None

    try:
        with requests.post(api_url, json={"question": question}, stream=True, timeout=120) as resp:
            resp.raise_for_status()

            for raw in resp.iter_lines():
                if not raw:
                    continue

                try:
                    event = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue

                etype = event.get("type", "").lower()

                # --- Thinking ---
                if etype == "thinking":
                    placeholder.write("Thinking… 🤔")
                    continue

                # --- Tool Start ---
                if etype == "tool_start":
                    tool_calls.append(event)
                    placeholder.write(final_text or "")
                    with st.expander(f"🔧 Tool Start: {event.get('tool_name', '')}"):
                        st.write(event.get("message"))
                        if event.get("args"):
                            st.json(event.get("args"))
                    continue

                # --- Tool End ---
                if etype == "tool_end":
                    tool_calls.append(event)
                    placeholder.write(final_text or "")
                    with st.expander(f"🧰 Tool Result: {event.get('tool_name', '')}"):
                        st.write(event.get("message"))
                        if event.get("content"):
                            st.json(safe_json_loads(event["content"]))
                    continue

                # --- Intermediate Answer ---
                if etype == "answer":
                    final_text = event.get("message", final_text)
                    placeholder.write(final_text)
                    continue

                # --- Final Result ---
                if etype == "final":
                    raw_final = event.get("answer") or event

                    if isinstance(raw_final, dict):
                        final_text = (
                            raw_final.get("answer")
                            or raw_final.get("text")
                            or final_text
                        )
                        reasoning = raw_final.get("reasoning_steps", [])
                        extra_tools = raw_final.get("tool_calls") or raw_final.get("tools") or []
                        if isinstance(extra_tools, list):
                            tool_calls.extend(extra_tools)
                    else:
                        final_text = str(raw_final)

                    placeholder.write(final_text)
                    continue

    except Exception as e:
        placeholder.error(f"❌ Backend error: {e}")
        raw_final = {"error": str(e)}
        final_text = "Error contacting backend."

    # Append assistant message
    st.session_state.chat.append({
        "role": "assistant",
        "text": final_text,
        "reasoning": reasoning,
        "tool_calls": tool_calls,
        "raw": raw_final
    })

    st.rerun()


# ---------------------------------------------
# Trigger on user input
# ---------------------------------------------
user_input = st.chat_input("Ask something about your graph...")

if user_input:
    stream_response(user_input, api_url)