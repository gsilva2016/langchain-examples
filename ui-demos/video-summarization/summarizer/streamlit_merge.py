import threading
import time
import queue
import streamlit as st
import streamlit.components.v1 as components
import argparse
from queue import Queue
import re
import os
import base64

stream_result_queue = Queue()
merge_result_queue = Queue()


from ov_lvm_wrapper import stream_queue

from streamlit_summarizer import summarizer_main, merge_queue, vertex_queue, alert_queue, stop_signal


# Initialize session state
if 'streamed_text' not in st.session_state:
    st.session_state['streamed_text'] = ''

if 'merged_summary' not in st.session_state:
    st.session_state['merged_summary'] = ''

if 'vertex_summary' not in st.session_state:
    st.session_state['vertex_summary'] = ''
    
if 'summarization_started' not in st.session_state:
    st.session_state['summarization_started'] = False
    
def vlm_video_player(url_or_path, **kwargs):
  st.video(url_or_path, **kwargs)

def run_summarization(args):
    summarizer_main(args)

def poll_streaming_results(stop_signal):
    streamed_text = ""
    while not stop_signal.is_set() or not stream_queue.empty():
        try:
            token = stream_queue.get(timeout=0.1)
            streamed_text += token
            stream_result_queue.put(streamed_text)
        except queue.Empty:
            continue

def poll_merge_results(stop_signal):
    merge_summary = ""
    while not stop_signal.is_set() or not merge_queue.empty():
        try:
            summary = merge_queue.get(timeout=0.1)
            print(f"\n\nRecieved merged summary in poll results: {summary}\n\n")
            merge_summary += summary
            merge_result_queue.put(merge_summary)
        except queue.Empty:
            continue
            
def toggle_start_button():
    if st.session_state['summarization_started']:
      stop_signal.set()
      st.session_state['summarization_started'] = False
      st.toast('Summarization process already started', icon=":material/info:")
      return
      
    else:
      st.session_state['summarization_started'] = True
      
def clear_queue(q):
  # Clear the queue
  while not q.empty():
    q.get()           

def toggle_relaunch_button():  
    clear_queue(stream_result_queue)   
    clear_queue(merge_result_queue)  
    clear_queue(stream_queue)  
    clear_queue(merge_queue)   
    clear_queue(vertex_queue)   
    clear_queue(alert_queue)    # Reset session state   
    st.session_state['streamed_text'] = '' 
    st.session_state['merged_summary'] = ''  
    st.session_state['vertex_summary'] = ''
    st.toast('Summarization process stopped', icon=":material/info:") # Main loopwhile st.session_state['summarization_started'] and not stop_signal.is_set():    # Your existing logic for polling queues and updating UI    # Ensure each thread checks stop_signal and exits if set

    stop_signal.set()  
    st.session_state['summarization_started'] = False # Wait for all threads to finish   
    if summarize_thread is not None:       
        summarize_thread.join()
 
def ShowMessage(severity_level):
    icon=""
    msg=""
    #low or no suspicious activity
    if severity_level < 0.3: 
        return 
    elif severity_level < 0.7:
        icon = ":material/warning:"
        msg = "Suspicious activity detected - Medium Level"
    else:
        icon = ":material/dangerous:"
        msg = "suspicious activity detected - Critical Level"
        
    message = st.toast(msg, icon=icon)            

st.set_page_config(initial_sidebar_state='collapsed', layout="wide")

st.title("🎥 Loss Prevention Video Summarization")
    
with st.sidebar:
  relaunch = st.button('stop', on_click=toggle_relaunch_button, disabled=not st.session_state['summarization_started'] )

# Split the page into two columns
spacer_col, left_col, right_col = st.columns([0.05, 0.55, 0.4])  # Adjust ratio as needed

video_path = 'one-by-one-person-detection.mp4'

with left_col:
    video_placeholder=st.empty()    
    with video_placeholder:
      if os.path.exists(video_path):
          st.video(video_path, muted=True)
      else:
          st.warning("The video file cannot be found")
    start_button_pressed = st.button("Start Summarization", on_click=toggle_start_button,) 

with right_col:
    alert_placeholder = st.empty()
    st.markdown("### 🧠 Merged Summaries")
    merge_placeholder = st.empty()
    merge_placeholder.markdown(
        f"""
        <div id="merge_scrollable" style='height:500px; overflow-y:auto;'>
            <pre>{st.session_state['merged_summary']}</pre>
        </div>
        <script>
            var container = document.getElementById('merge_scrollable');
            if (container) {{
                container.scrollTop = container.scrollHeight;
            }}
        </script>
        """,
        unsafe_allow_html=True
    )
    right_col.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    st.markdown("### ☁️ Cloud Generated Anomalous Summaries")
    vertex_placeholder = st.empty()
    vertex_placeholder.markdown(
        f"""
        <div id="vertex_scrollable" style='height:500px; overflow-y:auto;'>
            <pre>{st.session_state['vertex_summary']}</pre>
        </div>
        <script>
            var container = document.getElementById('vertex_scrollable');
            if (container) {{
                container.scrollTop = container.scrollHeight;
            }}
        </script>
        """,
        unsafe_allow_html=True
    )

with left_col:
    st.markdown("### 📄 Chunk Summaries")
    chunk_placeholder = st.empty()
    safe_text = (st.session_state['streamed_text'].replace('\n', '<br>').replace('[CHUNK ', '<br><strong>[CHUNK ').replace('sec]', 'sec]</strong>'))
    chunk_placeholder.markdown(
        f"""
        <div id="scrollable" style='height:500px; overflow-y:auto;'>
            <pre>{st.session_state['streamed_text']}</pre>
        </div>
        <script>
            var container = document.getElementById('scrollable');
            if (container) {{
                container.scrollTop = container.scrollHeight;
            }}
        </script>
        """,
        unsafe_allow_html=True
    )


summarize_thread = None

if start_button_pressed: 
    clear_queue(stream_result_queue)
    clear_queue(merge_result_queue)
    clear_queue(stream_queue)
    clear_queue(merge_queue)
    clear_queue(vertex_queue)
    clear_queue(alert_queue)

    st.session_state['streamed_text'] = ''
    st.session_state['merged_summary'] = ''
    st.session_state['vertex_summary'] = ''
  
    stop_signal.clear()
      
    st.session_state['summarization_started'] = True
    
    args = argparse.Namespace(
        video_file=video_path,
        model_dir='MiniCPM_INT8/',
        prompt="""
        You are an expert investigator, please analyze this video and identify the behaviors and actions of the people seen in the video.
        Summarize the video - noting actions of all individuals in the scene - generating an Overall Summary and Highlights of the key events.
        It should be formatted as such:

        Overall Summary
        Here is a detailed description of the video.

        Highlights
        1) Here is a bullet point list of behavior/actions that are significant enough to highlight.
        """,
        device='GPU.1',
        max_new_tokens=220,
        max_num_frames=48,
        chunk_duration=15,
        chunk_overlap=2,
        merge_cadence=30,
        resolution=[480, 270],
        outfile='',
        extend_to_vertex=False,
        anomaly_thresh=0.5,
        cloud_model="gemini-2.5-pro-preview-05-06"
    )

    #stop_signal = threading.Event()
    summarize_thread = threading.Thread(target=run_summarization, args=(args,))
    summarize_thread.start()
    
    # Define these before the loop
    chunk_duration = args.chunk_duration
    chunk_overlap = args.chunk_overlap
    chunk_index = 0
    current_time = 0
    chunk_summaries = []
    
    #reload video player
    with video_placeholder:
      if os.path.exists(video_path):
        st.write("Loading model...")
        time.sleep(11)
        st.video(video_path, muted=True, autoplay=True)
      else:
          st.warning("The video file cannot be found")    

while st.session_state['summarization_started'] and not stop_signal.is_set(): # and ( summarize_thread.is_alive() or not stream_queue.empty() or not merge_queue.empty() or not vertex_queue.empty() or not alert_queue.empty()):
      try:
          if not alert_queue.empty():
              level = alert_queue.get(timeout=0.1)
              with alert_placeholder:
                ShowMessage(level)
      except queue.Empty:
        pass


      try:
          if not stream_queue.empty():
              token = stream_queue.get(timeout=0.1)
              st.session_state['streamed_text'] += token
              safe_text = (st.session_state['streamed_text'].replace('\n', '<br>').replace('[CHUNK ', '<br><strong>[CHUNK ').replace('sec]', 'sec]</strong>'))
              chunk_placeholder.markdown(
                  f"""
                  <div id="scrollable" style='height:500px; overflow-y:auto; position:relative'>
                      <div style="white-space: pre-wrap;flex-direction: column-reverse;" id="streamed_text">{safe_text}</div>
                      <div id="bottom-anchor"></div>
                  <script>
                  """,
                  unsafe_allow_html=True
              )
      except queue.Empty:
          pass

      try:
          if not merge_queue.empty():
              summary = merge_queue.get(timeout=0.1)
              st.session_state['merged_summary'] += summary
              safe_merged_text = (st.session_state['merged_summary'].replace('\n', '<br>').replace('[MERGED SUMMARY ', '<br><strong>[MERGED SUMMARY ').replace('sec]', 'sec]</strong>'))
              merge_placeholder.markdown(
                  f"""
                  <div id="merge_scrollable" style='height:400px; overflow-y:auto;'>
                      <div style="white-space: pre-wrap;flex-direction: column-reverse;">{safe_merged_text}</div>
                  </div>
                  """,
                  unsafe_allow_html=True
              )
      except queue.Empty:
          pass
          
      try:
          if not vertex_queue.empty():
              cloud_summary = vertex_queue.get(timeout=0.1)
              st.session_state['vertex_summary'] += cloud_summary
              safe_vertex_text = (st.session_state['vertex_summary'].replace('\n', '<br>').replace('[CLOUD SUMMARY ', '<br><strong>[CLOUD SUMMARY ').replace('sec]', 'sec]</strong>'))
              vertex_placeholder.markdown(
                  f"""
                  <div id="merge_scrollable" style='height:400px; overflow-y:auto;'>
                      <div style="white-space: pre-wrap;">{safe_vertex_text}</div>
                  </div>
                  <script>
                      var container = document.getElementById('merge_scrollable');
                      container.scrollTop = -container.scrollHeight;
                  </script>
                  """,
                  unsafe_allow_html=True
              )
      except queue.Empty:
          pass
          
if summarize_thread is not None:
  print(f"streamlit main, summarize_thread.join()")
  summarize_thread.join()
  st.success("Summarization complete!")
  st.session_state['summarization_started'] = False
  print(f"end streamlit main, streamlit_merge")

