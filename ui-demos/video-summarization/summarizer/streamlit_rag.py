import streamlit as st
import ast
from rag import search_in_milvus
import re

def play_video(video_path, offset):
    video_file = open(video_path, 'rb') 
    video_bytes = video_file.read() 
    st.video(video_bytes, start_time=int(offset))

st.set_page_config(page_title="RAG Seach UI", layout="wide")

st.title("Video RAG Search interface")

query = st.text_input("Enter your query:", "")

video_file = 'computex_video.mp4'
#video_url= f'http://localhost:8005/{video_file}'

col1, col2 = st.columns([1, 1])

if st.button("Search"):
    with st.spinner("Searching..."):
        response = search_in_milvus(query)

    if "error" in response:
        st.error(f"Search failed: {response['error']}")

    else:
        try:
            results = ast.literal_eval(response["result"])
            st.success("Results recieved!")
            chunk_summary = results["results"][0]["chunk_summary"]
            chunk_path = results["results"][0]["chunk_path"]
            start_time = results["results"][0]["start_time"]

            with col2:
                st.markdown("## Textual Summary")
                safe_summary = re.sub(
                    r"(Start time:\s*\d+\s*End time:\s*\d+)",
                    r"<strong>\1</strong>",
                    chunk_summary
                )
                safe_summary = safe_summary.replace('\n', '<br>')
                summary_placeholder = st.empty()
                summary_placeholder.markdown(
                    f"""
                    <div id="scrollable" style='height:500px; overflow-y:auto;'>
                       <div style="white-space: pre-wrap;">{safe_summary}</div>
                    </div>
                    <script>
                        var container = document.getElementById('scrollable');
                        container.scrollTop = container.scrollHeight;
                    </script>
                    """,
                    unsafe_allow_html=True
                )
                #st.text_area(f"Summary of this segment:\n{chunk_summary}", height=300)
            with col1:
                if video_file:
                    st.markdown("Video playback")
                    play_video(video_file, start_time)

        except Exception as e:
            st.error(f"Failed to parse response: {e}")
