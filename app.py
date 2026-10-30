"""
AI Writing Coach
Improve Your English, One Paragraph at a Time!

Run with:  streamlit run app.py
"""

import html
import json
import os

import streamlit as st
import streamlit.components.v1 as components
from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_NAME = "llama-3.3-70b-versatile"
MAX_CHARS = 5000

STYLE_OPTIONS = {
    "Simple English": (
        "Rewrite the paragraph using easy, everyday vocabulary and short, simple "
        "sentences that a beginner can understand."
    ),
    "Formal English": (
        "Rewrite the paragraph in a professional, formal tone suitable for work, "
        "school, or official communication. Avoid slang and contractions."
    ),
    "Advanced English": (
        "Rewrite the paragraph using more sophisticated vocabulary and varied, "
        "well-structured sentences, like a skilled and confident writer."
    ),
}

SYSTEM_PROMPT = """You are a friendly, patient English writing teacher helping learners improve their writing.

Rules:
1. Correct grammar, spelling, punctuation, sentence structure, and incorrect word usage.
2. Preserve the user's original meaning. Do not add new ideas or remove important ones.
3. NEVER invent mistakes. If the paragraph is already correct, return it unchanged, set "has_mistakes" to false, and return an empty "mistakes" list.
4. Explain each mistake in very simple English for beginners. Avoid complicated grammar terminology; if you must use a grammar word, explain it in plain words.
5. For every mistake, give the original incorrect sentence (exactly as written by the user), the corrected sentence, and a short, clear explanation of why the change was needed. If one sentence has several mistakes, explain them together in that one entry.
6. The rewritten paragraph must be natural, fluent English in the requested style and must keep the original meaning.
7. Respond with VALID JSON ONLY, with no extra text and no markdown, using exactly this structure:
{
  "has_mistakes": true,
  "corrected_text": "the full corrected paragraph",
  "mistakes": [
    {
      "original_sentence": "the incorrect sentence from the user's text",
      "corrected_sentence": "the corrected version of that sentence",
      "explanation": "a simple explanation of why it was wrong"
    }
  ],
  "rewritten_text": "the paragraph rewritten in the requested style"
}"""


# --------------------------------------------------------------------------- #
# Page setup and styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="AI Writing Coach",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 900px; }

    .hero {
        text-align: center;
        padding: 2rem 1rem 1.6rem 1rem;
        margin-bottom: 1.5rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #0ea5e9 100%);
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    }
    .hero h1 { color: #ffffff; margin: 0; font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px; }
    .hero p  { color: rgba(255,255,255,0.92); margin: 0.5rem 0 0 0; font-size: 1.1rem; }

    .section-title {
        font-size: 1.35rem; font-weight: 700; margin: 1.8rem 0 0.7rem 0;
        padding-left: 0.7rem; border-left: 5px solid #6366f1;
    }

    .card {
        padding: 1.1rem 1.3rem; border-radius: 14px; line-height: 1.7;
        border: 1px solid rgba(99, 102, 241, 0.25);
        background: rgba(99, 102, 241, 0.07);
        white-space: pre-wrap; word-wrap: break-word;
    }
    .card.green { border-color: rgba(16,185,129,0.35); background: rgba(16,185,129,0.08); }
    .card.blue  { border-color: rgba(14,165,233,0.35); background: rgba(14,165,233,0.08); }

    .mistake-card {
        padding: 1rem 1.2rem; margin-bottom: 0.9rem; border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.06);
    }
    .mistake-num { font-weight: 700; color: #6366f1; margin-bottom: 0.5rem; }
    .tag { display: inline-block; font-size: 0.78rem; font-weight: 700; padding: 2px 9px;
           border-radius: 999px; margin-right: 6px; }
    .tag.wrong { background: rgba(239,68,68,0.15); color: #ef4444; }
    .tag.right { background: rgba(16,185,129,0.15); color: #10b981; }
    .tag.why   { background: rgba(245,158,11,0.18); color: #d97706; }
    .line { margin: 0.35rem 0; line-height: 1.6; }

    .stButton > button {
        border-radius: 10px; font-weight: 600; padding: 0.55rem 1rem;
        transition: all 0.15s ease-in-out;
    }
    .stButton > button:hover { transform: translateY(-1px); }

    .footer { text-align: center; opacity: 0.6; font-size: 0.85rem; margin-top: 3rem; }
    #MainMenu, footer { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def get_api_key(sidebar_key: str = "") -> str:
    """Find the Groq API key: sidebar input > Streamlit secrets > environment."""
    if sidebar_key.strip():
        return sidebar_key.strip()
    try:
        secret = st.secrets.get("GROQ_API_KEY", "")
        if secret:
            return str(secret).strip()
    except Exception:
        pass  # No secrets file available
    return os.getenv("GROQ_API_KEY", "").strip()


def clear_text() -> None:
    """Callback for the Clear Text button."""
    st.session_state["input_text"] = ""
    st.session_state["results"] = None
    st.session_state["empty_warning"] = False


def copy_button(text: str, key: str, label: str = "📋 Copy Text") -> None:
    """Render a button that copies `text` to the clipboard (client-side JS)."""
    safe_text = json.dumps(text).replace("</", "<\\/")
    components.html(
        f"""
        <button id="btn-{key}" style="
            font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
            background: #6366f1; color: white; border: none; border-radius: 10px;
            padding: 9px 18px; font-size: 14px; font-weight: 600; cursor: pointer;">
            {label}
        </button>
        <script>
            const text = {safe_text};
            const btn = document.getElementById("btn-{key}");
            const original = btn.innerHTML;
            function done(ok) {{
                btn.innerHTML = ok ? "✅ Copied!" : "❌ Copy failed";
                setTimeout(() => btn.innerHTML = original, 1800);
            }}
            function fallback() {{
                const ta = document.createElement("textarea");
                ta.value = text;
                ta.style.position = "fixed";
                ta.style.opacity = "0";
                document.body.appendChild(ta);
                ta.select();
                let ok = false;
                try {{ ok = document.execCommand("copy"); }} catch (e) {{}}
                document.body.removeChild(ta);
                done(ok);
            }}
            btn.addEventListener("click", () => {{
                if (navigator.clipboard && window.isSecureContext) {{
                    navigator.clipboard.writeText(text).then(() => done(true)).catch(fallback);
                }} else {{
                    fallback();
                }}
            }});
        </script>
        """,
        height=48,
    )


def analyze_writing(paragraph: str, style: str, api_key: str) -> dict:
    """Send the paragraph to Groq and return the parsed JSON result."""
    client = Groq(api_key=api_key, timeout=60.0)

    user_prompt = (
        f"Rewriting style requested: {style}\n"
        f"Style instructions: {STYLE_OPTIONS[style]}\n\n"
        "Here is the student's paragraph:\n"
        '"""\n'
        f"{paragraph}\n"
        '"""\n\n'
        "Return the JSON now."
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=3000,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or ""
    data = json.loads(raw)

    # Validate and normalise the structure
    corrected = str(data.get("corrected_text", "")).strip()
    rewritten = str(data.get("rewritten_text", "")).strip()
    if not corrected or not rewritten:
        raise ValueError("The AI response was incomplete.")

    mistakes = []
    for item in data.get("mistakes", []) or []:
        if not isinstance(item, dict):
            continue
        original = str(item.get("original_sentence", "")).strip()
        fixed = str(item.get("corrected_sentence", "")).strip()
        why = str(item.get("explanation", "")).strip()
        if original and fixed and original != fixed:
            mistakes.append(
                {"original_sentence": original, "corrected_sentence": fixed, "explanation": why}
            )

    return {
        "has_mistakes": bool(mistakes) and bool(data.get("has_mistakes", True)),
        "corrected_text": corrected,
        "mistakes": mistakes,
        "rewritten_text": rewritten,
        "style": style,
    }


def render_card(text: str, css_class: str = "") -> None:
    st.markdown(f'<div class="card {css_class}">{html.escape(text)}</div>', unsafe_allow_html=True)


def render_results(results: dict) -> None:
    # 1. Corrected paragraph
    st.markdown('<div class="section-title">✅ Corrected Paragraph</div>', unsafe_allow_html=True)
    render_card(results["corrected_text"], "green")
    copy_button(results["corrected_text"], key="corrected")

    # 2. Mistakes & explanations
    st.markdown('<div class="section-title">🔍 Mistakes &amp; Explanations</div>', unsafe_allow_html=True)
    if not results["mistakes"]:
        st.success("🎉 Great job! I didn't find any mistakes in your paragraph.")
    else:
        count = len(results["mistakes"])
        st.caption(f"I found {count} thing{'s' if count != 1 else ''} to improve. Let's go through them:")
        for i, m in enumerate(results["mistakes"], start=1):
            st.markdown(
                f"""
                <div class="mistake-card">
                    <div class="mistake-num">Mistake {i}</div>
                    <div class="line"><span class="tag wrong">❌ Original</span>{html.escape(m['original_sentence'])}</div>
                    <div class="line"><span class="tag right">✔ Corrected</span>{html.escape(m['corrected_sentence'])}</div>
                    <div class="line"><span class="tag why">💡 Why?</span>{html.escape(m['explanation'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 3. Rewritten paragraph
    st.markdown(
        f'<div class="section-title">✨ Rewritten Paragraph ({html.escape(results["style"])})</div>',
        unsafe_allow_html=True,
    )
    render_card(results["rewritten_text"], "blue")
    copy_button(results["rewritten_text"], key="rewritten")


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
st.session_state.setdefault("input_text", "")
st.session_state.setdefault("results", None)
st.session_state.setdefault("empty_warning", False)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    sidebar_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Optional if GROQ_API_KEY is set in .streamlit/secrets.toml or as an environment variable.",
    )
    st.caption("Get a free key at console.groq.com")
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Paste your paragraph\n"
        "2. Pick a rewriting style\n"
        "3. Click **Analyze My Writing**"
    )

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
        <h1>✍️ AI Writing Coach</h1>
        <p>Improve Your English, One Paragraph at a Time!</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Input section
# --------------------------------------------------------------------------- #
st.markdown('<div class="section-title">📝 Your Paragraph</div>', unsafe_allow_html=True)

st.text_area(
    "Enter or paste your paragraph",
    key="input_text",
    height=220,
    max_chars=MAX_CHARS,
    placeholder="Example: Yesterday me and my friend goes to the market for buy some vegetable...",
    label_visibility="collapsed",
)

selected_style = st.radio(
    "Choose a rewriting style",
    options=list(STYLE_OPTIONS.keys()),
    horizontal=True,
    help="The corrected paragraph keeps your style. The rewritten paragraph uses the style you pick here.",
)

col_analyze, col_clear = st.columns([3, 1])
with col_analyze:
    analyze_clicked = st.button("🚀 Analyze My Writing", type="primary", use_container_width=True)
with col_clear:
    st.button("🗑️ Clear Text", on_click=clear_text, use_container_width=True)

# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
if analyze_clicked:
    paragraph = st.session_state["input_text"].strip()
    st.session_state["empty_warning"] = False

    if not paragraph:
        st.session_state["results"] = None
        st.session_state["empty_warning"] = True
    else:
        api_key = get_api_key(sidebar_key)
        if not api_key:
            st.session_state["results"] = None
            st.error(
                "🔑 No Groq API key found. Enter it in the sidebar, set the `GROQ_API_KEY` "
                "environment variable, or add it to `.streamlit/secrets.toml`."
            )
        else:
            try:
                with st.spinner("Your writing coach is reading your paragraph..."):
                    st.session_state["results"] = analyze_writing(paragraph, selected_style, api_key)
            except AuthenticationError:
                st.session_state["results"] = None
                st.error("🔑 Your API key was rejected by Groq. Please check that it is correct.")
            except RateLimitError:
                st.session_state["results"] = None
                st.warning("⏳ Too many requests right now. Please wait a moment and try again.")
            except APIConnectionError:
                st.session_state["results"] = None
                st.error("🌐 Could not connect to Groq. Please check your internet connection.")
            except APIStatusError as e:
                st.session_state["results"] = None
                st.error(f"⚠️ Groq returned an error (status {e.status_code}). Please try again.")
            except (json.JSONDecodeError, ValueError):
                st.session_state["results"] = None
                st.error("😕 The AI gave an unexpected answer. Please click the button again.")
            except Exception as e:  # last-resort safety net
                st.session_state["results"] = None
                st.error(f"Something went wrong: {e}")

if st.session_state["empty_warning"]:
    st.warning("✋ Please write or paste a paragraph first, then click **Analyze My Writing**.")

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
if st.session_state["results"]:
    render_results(st.session_state["results"])

st.markdown(
    '<div class="footer">Made with ❤️ using Streamlit &amp; Groq</div>',
    unsafe_allow_html=True,
)
