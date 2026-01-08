import gradio as gr
import base64
from retrieve import answer_question
from pathlib import Path

# Load welcome.png and convert to base64
welcome_image_path = Path(__file__).parent.parent / "welcome.png"
welcome_image_base64 = None
if welcome_image_path.exists():
    with open(welcome_image_path, 'rb') as f:
        image_data = f.read()
        welcome_image_base64 = base64.b64encode(image_data).decode('utf-8')

custom_css = """
.gradio-container { max-width: 100% !important; }
.welcome-container {
    text-align: center;
    padding: 2rem;
    margin-bottom: 2rem;
}
.welcome-icon {
    width: 80px;
    height: 80px;
    margin: 0 auto 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.welcome-icon img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    border-radius: 12px;
}
.welcome-text {
    font-size: 2.5rem;
    font-weight: bold;
    color: #000000;
    margin: 0;
}
"""

welcome_html = f"""
<div class="welcome-container">
    <div class="welcome-icon">
        <img src="data:image/png;base64,{welcome_image_base64}" alt="Welcome Icon" />
    </div>
    <h1 class="welcome-text">Form EP1001 documentation</h1>
</div>
""" if welcome_image_base64 else """
<div class="welcome-container">
    <h1 class="welcome-text">Welcome</h1>
</div>
"""

with gr.Blocks(css=custom_css) as demo:
    gr.HTML(welcome_html)
    gr.ChatInterface(answer_question)

demo.launch(
    auth=[("epo", "epo12345")]
)
