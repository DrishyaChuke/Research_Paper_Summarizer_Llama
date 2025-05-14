import gradio as gr
from main import extract_content_and_summarize_text
import tempfile
import os
import re

def process_pdf(file):
    if file is None:
        yield "⚠️ Please upload a PDF file to begin."
        return

    # save upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file)
        tmp_path = tmp.name

    summary_md = "# 📄 Research Paper Summary\n\n"
    yield summary_md + "⏳ Extracting and summarizing, please wait...\n\n"

    # stream chunks
    for chunk in extract_content_and_summarize_text(tmp_path):
        if chunk and chunk.strip():
            print("🔹 Chunk received:", repr(chunk))
            formatted = re.sub(
                r'(?:^|\n)(?:\d+\.\s*)?([A-Z][^:]+):',
                r'\n## \1',
                chunk
            )
            summary_md += formatted + "\n"
            yield summary_md

    # cleanup
    os.unlink(tmp_path)

# build interface
demo = gr.Interface(
    fn=process_pdf,
    inputs=gr.File(label="Upload Research Paper (PDF)", file_types=[".pdf"], type="binary"),
    outputs=gr.Markdown(),
    title="🧠 Research Paper Summarizer (Together AI)",
    description="Upload a PDF to get an AI-powered summary using Together AI's LLaMA API.",
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
