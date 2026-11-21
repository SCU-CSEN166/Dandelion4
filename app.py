
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Dandelion Africa Activity Processor",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 Dandelion Africa Activity Processor")
st.write("Process community activity messages and audio recordings.")

mode = st.radio(
    "Choose input type",
    ["Text message", "Upload files"],
    horizontal=True,
)

text = ""
uploads = []

if mode == "Text message":
    text = st.text_area(
        "Enter a WhatsApp-style message",
        placeholder="We screened 28 women in Nakuru today.",
        height=140,
    )
else:
    uploads = st.file_uploader(
        "Upload audio or message files",
        type=["wav", "mp3", "m4a", "txt", "json", "csv", "zip"],
        accept_multiple_files=True,
    )

if st.button("Process", type="primary"):
    if mode == "Text message" and not text.strip():
        st.warning("Please enter a message first.")
    elif mode == "Upload files" and not uploads:
        st.warning("Please upload a file first.")
    else:
        try:
            from src.pipeline.pipeline import run_on_input

            with tempfile.TemporaryDirectory() as tmp:
                workdir = Path(tmp)
                input_dir = workdir / "inputs"
                output_dir = workdir / "outputs"
                input_dir.mkdir()
                output_dir.mkdir()

                if mode == "Text message":
                    (input_dir / "message.txt").write_text(
                        text, encoding="utf-8"
                    )
                else:
                    for uploaded in uploads:
                        destination = input_dir / Path(uploaded.name).name
                        destination.write_bytes(uploaded.getvalue())

                with st.spinner("Processing with the AI pipeline..."):
                    result = run_on_input(input_dir, out_dir=output_dir)

                csv_path = output_dir / "outputs.csv"
                json_path = output_dir / "records.json"

                if csv_path.exists():
                    csv_data = csv_path.read_bytes()
                    st.subheader("Processed activity records")
                    st.dataframe(pd.read_csv(csv_path), use_container_width=True)
                    st.download_button(
                        "Download CSV",
                        csv_data,
                        "dandelion_results.csv",
                        "text/csv",
                    )

                if json_path.exists():
                    json_data = json_path.read_bytes()
                    st.subheader("Structured JSON output")
                    st.json(json.loads(json_data.decode("utf-8")))
                    st.download_button(
                        "Download JSON",
                        json_data,
                        "dandelion_results.json",
                        "application/json",
                    )

                if not csv_path.exists() and not json_path.exists():
                    st.warning(
                        f"No expected output files were found. Pipeline returned: {result}"
                    )

        except Exception as exc:
            st.error(f"Processing failed: {exc}")
