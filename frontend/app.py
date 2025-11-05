import streamlit as st
import requests
import pandas as pd
import json
from PIL import Image
import io
import uuid
import base64
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Configuration - Dynamic API URL for Cloud Run deployment
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8080")

st.set_page_config(
    page_title="Botlytics - Data Analytics with AI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Botlytics")
st.markdown("*Explore and visualize your data with natural language*")

# Initialize session state
if 'dataset_id' not in st.session_state:
    st.session_state.dataset_id = None
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None
if 'conversation_id' not in st.session_state:
    st.session_state.conversation_id = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Upload Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your CSV data to start analyzing"
    )
    
    if uploaded_file is not None:
        if st.button("Upload & Process"):
            with st.spinner("Uploading file..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    response = requests.post(f"{API_BASE_URL}/api/v1/upload", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.dataset_id = result['dataset_id']
                        st.session_state.dataset_info = result
                        st.success("✅ File uploaded successfully!")
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Main content area
if st.session_state.dataset_id:
    # Display dataset info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Dataset Overview")
        info = st.session_state.dataset_info
        
        st.write(f"**Dataset ID:** `{st.session_state.dataset_id}`")
        st.write(f"**Rows:** {info['rows']:,}")
        st.write(f"**Columns:** {', '.join(info['columns'])}")
        
        # Show preview
        if 'preview' in info:
            st.write("**Preview:**")
            df_preview = pd.DataFrame(info['preview'])
            st.dataframe(df_preview, use_container_width=True)
    
    with col2:
        st.subheader("🔄 Actions")
        if st.button("🔍 Refresh Dataset Info"):
            try:
                response = requests.get(f"{API_BASE_URL}/api/v1/datasets/{st.session_state.dataset_id}")
                if response.status_code == 200:
                    st.session_state.dataset_info = response.json()
                    st.success("Dataset info refreshed!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error refreshing: {str(e)}")
        
        if st.button("🗑️ Clear Dataset"):
            st.session_state.dataset_id = None
            st.session_state.dataset_info = None
            st.rerun()
    
    # Advanced Agent Interface
    st.subheader("🤖 Advanced AI Agent")
    
    # Accessibility settings in sidebar
    with st.sidebar:
        st.header("♿ Accessibility")
        
        # Text-to-speech settings
        enable_tts = st.checkbox("🔊 Enable Text-to-Speech", value=False)
        if enable_tts:
            tts_language = st.selectbox("TTS Language", ["en-US", "es-ES", "fr-FR", "de-DE"], index=0)
            tts_rate = st.slider("Speaking Rate", 0.5, 2.0, 1.0, 0.1)
            tts_pitch = st.slider("Voice Pitch", -10.0, 10.0, 0.0, 1.0)
        
        # Voice input settings
        enable_voice = st.checkbox("🎤 Enable Voice Input", value=False)
        if enable_voice:
            voice_language = st.selectbox("Voice Language", ["en-US", "es-ES", "fr-FR", "de-DE"], index=0)
        
        # Visual accessibility
        high_contrast = st.checkbox("🔲 High Contrast Mode", value=False)
        large_text = st.checkbox("🔍 Large Text", value=False)
        
        # Audio descriptions
        enable_audio_desc = st.checkbox("📢 Audio Descriptions for Charts", value=False)
    
    # Apply accessibility styles
    if high_contrast:
        st.markdown("""
        <style>
        .stApp { background-color: #000000; color: #FFFFFF; }
        .stButton > button { background-color: #FFFFFF; color: #000000; border: 2px solid #FFFFFF; }
        </style>
        """, unsafe_allow_html=True)
    
    if large_text:
        st.markdown("""
        <style>
        .stApp { font-size: 18px; }
        .stMarkdown { font-size: 18px; }
        </style>
        """, unsafe_allow_html=True)
    
    # Conversation interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "🧠 Reasoning", "💻 Code", "📊 Analysis", "♿ Accessibility"])
    
    with tab1:
        st.write("**Multi-turn Conversation with Memory**")
        
        # Start conversation if not exists
        if not st.session_state.conversation_id:
            if st.button("🚀 Start New Conversation"):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/conversation/start",
                        params={"user_id": st.session_state.user_id}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.conversation_id = result["session_id"]
                        st.session_state.conversation_history = [
                            {"role": "assistant", "content": result["message"]}
                        ]
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to start conversation: {e}")
        
        # Display conversation history
        if st.session_state.conversation_history:
            for msg in st.session_state.conversation_history:
                if msg["role"] == "user":
                    st.chat_message("user").write(msg["content"])
                else:
                    st.chat_message("assistant").write(msg["content"])
        
        # Voice input option
        if enable_voice and st.session_state.conversation_id:
            st.write("🎤 **Voice Input**")
            
            # Simple voice input button (placeholder - would need WebRTC implementation)
            if st.button("🎤 Start Voice Input"):
                st.info("Voice input feature would be implemented with WebRTC for real-time audio capture")
                # Placeholder for voice input implementation
                # In a real implementation, this would:
                # 1. Capture audio using WebRTC
                # 2. Convert to base64
                # 3. Send to speech-to-text API
                # 4. Process the transcribed text
        
        # Chat input
        if st.session_state.conversation_id:
            user_input = st.chat_input("Ask me anything about your data...")
            
            if user_input:
                # Add user message to history
                st.session_state.conversation_history.append({"role": "user", "content": user_input})
                
                with st.spinner("🤖 Agent is thinking..."):
                    try:
                        payload = {
                            "session_id": st.session_state.conversation_id,
                            "message": user_input,
                            "dataset_id": st.session_state.dataset_id
                        }
                        
                        response = requests.post(
                            f"{API_BASE_URL}/api/v1/conversation/continue",
                            json=payload
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            assistant_response = result["response"]
                            
                            # Add assistant response
                            st.session_state.conversation_history.append({
                                "role": "assistant", 
                                "content": assistant_response
                            })
                            
                            # Text-to-speech for assistant response
                            if enable_tts:
                                try:
                                    tts_payload = {
                                        "text": assistant_response,
                                        "language_code": tts_language,
                                        "speaking_rate": tts_rate,
                                        "pitch": tts_pitch
                                    }
                                    
                                    tts_response = requests.post(
                                        f"{API_BASE_URL}/api/v1/accessibility/text-to-speech",
                                        json=tts_payload
                                    )
                                    
                                    if tts_response.status_code == 200:
                                        tts_result = tts_response.json()
                                        if tts_result.get("success") and tts_result.get("audio_base64"):
                                            # Display audio player
                                            audio_bytes = base64.b64decode(tts_result["audio_base64"])
                                            st.audio(audio_bytes, format="audio/mp3")
                                            st.caption("🔊 Audio response generated")
                                except Exception as e:
                                    st.warning(f"TTS failed: {e}")
                            
                            # Show tools used
                            if result.get("tools_used"):
                                st.info(f"🛠️ Tools used: {', '.join(result['tools_used'])}")
                            
                            st.rerun()
                        else:
                            st.error(f"Conversation failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    with tab2:
        st.write("**Advanced Reasoning Chains**")
        
        problem = st.text_area(
            "Describe a complex problem to solve:",
            placeholder="e.g., Analyze sales performance, identify trends, and recommend strategies for improvement",
            height=100
        )
        
        if st.button("🧠 Execute Reasoning Chain"):
            if problem.strip():
                with st.spinner("🔗 Executing reasoning chain..."):
                    try:
                        payload = {
                            "problem": problem,
                            "dataset_id": st.session_state.dataset_id
                        }
                        
                        response = requests.post(
                            f"{API_BASE_URL}/api/v1/reasoning-chain",
                            json=payload
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            st.success("✅ Reasoning chain completed!")
                            
                            # Show decomposition
                            st.write("**Problem Decomposition:**")
                            if result.get("decomposition", {}).get("steps"):
                                for step in result["decomposition"]["steps"]:
                                    st.write(f"**Step {step['step']}:** {step['description']}")
                            
                            # Show synthesis
                            st.write("**Final Analysis:**")
                            synthesis = result.get("synthesis", {})
                            if isinstance(synthesis, dict) and "text" in synthesis:
                                st.write(synthesis["text"])
                            else:
                                st.write(synthesis)
                        else:
                            st.error(f"Reasoning failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    with tab3:
        st.write("**Code Interpreter**")
        
        code = st.text_area(
            "Enter Python code to execute:",
            placeholder="# Example: Analyze data with pandas\nresult = df.groupby('category')['sales'].sum()\nprint(result)",
            height=150
        )
        
        if st.button("💻 Execute Code"):
            if code.strip():
                with st.spinner("⚡ Executing code..."):
                    try:
                        payload = {
                            "code": code,
                            "dataset_id": st.session_state.dataset_id
                        }
                        
                        response = requests.post(
                            f"{API_BASE_URL}/api/v1/code-interpreter",
                            json=payload
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            if result["success"]:
                                st.success("✅ Code executed successfully!")
                                
                                if result.get("output"):
                                    st.write("**Output:**")
                                    st.code(result["output"])
                                
                                if result.get("variables"):
                                    st.write("**Variables:**")
                                    st.json(result["variables"])
                            else:
                                st.error(f"Execution failed: {result.get('error')}")
                        else:
                            st.error(f"Code execution failed: {response.text}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    with tab4:
        st.write("**Advanced Data Analysis**")
        
        analysis_type = st.selectbox(
            "Select analysis type:",
            ["comprehensive", "statistical", "trend", "correlation"]
        )
        
        if st.button("📊 Run Advanced Analysis"):
            with st.spinner("🔍 Performing advanced analysis..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/data-analysis/advanced",
                        params={
                            "dataset_id": st.session_state.dataset_id,
                            "analysis_type": analysis_type
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success("✅ Analysis completed!")
                        
                        # Show basic analysis
                        if result.get("basic_analysis"):
                            st.write("**Dataset Overview:**")
                            basic = result["basic_analysis"]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Rows", basic["shape"][0])
                                st.metric("Columns", basic["shape"][1])
                            with col2:
                                missing = sum(basic.get("missing_values", {}).values())
                                st.metric("Missing Values", missing)
                                st.metric("Memory Usage", f"{basic.get('memory_usage', 0):,} bytes")
                        
                        # Show recommendations
                        if result.get("recommendations"):
                            st.write("**Recommendations:**")
                            for rec in result["recommendations"]:
                                st.write(f"• {rec}")
                    else:
                        st.error(f"Analysis failed: {response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Conversation summary
        if st.session_state.conversation_id:
            if st.button("📋 Get Conversation Summary"):
                try:
                    response = requests.get(
                        f"{API_BASE_URL}/api/v1/conversation/{st.session_state.conversation_id}/summary"
                    )
                    if response.status_code == 200:
                        summary = response.json()
                        st.write("**Conversation Summary:**")
                        st.json(summary)
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab5:
        st.write("**♿ Accessibility Features**")
        
        # Text-to-Speech Testing
        st.subheader("🔊 Text-to-Speech")
        
        tts_text = st.text_area(
            "Enter text to convert to speech:",
            placeholder="Type any text here to test text-to-speech functionality...",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            tts_lang = st.selectbox("Language", ["en-US", "es-ES", "fr-FR", "de-DE", "ja-JP"], key="tts_lang")
            tts_speed = st.slider("Speaking Rate", 0.5, 2.0, 1.0, 0.1, key="tts_speed")
        with col2:
            tts_voice_pitch = st.slider("Voice Pitch", -10.0, 10.0, 0.0, 1.0, key="tts_pitch")
        
        if st.button("🔊 Generate Speech") and tts_text.strip():
            with st.spinner("Generating speech..."):
                try:
                    payload = {
                        "text": tts_text,
                        "language_code": tts_lang,
                        "speaking_rate": tts_speed,
                        "pitch": tts_voice_pitch
                    }
                    
                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/accessibility/text-to-speech",
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            audio_bytes = base64.b64decode(result["audio_base64"])
                            st.audio(audio_bytes, format="audio/mp3")
                            
                            st.success("✅ Speech generated successfully!")
                            st.write(f"**Duration:** ~{result.get('duration_estimate', 0):.1f} seconds")
                            st.write(f"**Format:** {result.get('audio_format', 'mp3')}")
                        else:
                            st.error(f"TTS failed: {result.get('error')}")
                    else:
                        st.error(f"Request failed: {response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Audio Description Testing
        st.subheader("📢 Audio Descriptions")
        
        st.write("Test audio descriptions for data visualizations:")
        
        # Sample chart data for testing
        sample_chart_data = {
            "title": "Sales by Product",
            "x_label": "Product",
            "y_label": "Sales ($)",
            "data_points": [
                {"x": "Product A", "y": 15000},
                {"x": "Product B", "y": 23000},
                {"x": "Product C", "y": 18000}
            ]
        }
        
        chart_type_desc = st.selectbox(
            "Chart Type", 
            ["bar", "line", "scatter", "pie"],
            key="chart_type_desc"
        )
        
        if st.button("📢 Generate Audio Description"):
            with st.spinner("Generating audio description..."):
                try:
                    payload = {
                        "chart_data": sample_chart_data,
                        "chart_type": chart_type_desc,
                        "detail_level": "detailed"
                    }
                    
                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/accessibility/audio-description",
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success("✅ Audio description generated!")
                        
                        # Show text description
                        st.write("**Text Description:**")
                        st.write(result["description"])
                        
                        # Show different summary formats
                        if result.get("summaries"):
                            st.write("**Accessible Summaries:**")
                            
                            summaries = result["summaries"]
                            
                            with st.expander("Screen Reader Optimized"):
                                st.write(summaries.get("screen_reader", ""))
                            
                            with st.expander("Simple Language"):
                                st.write(summaries.get("simple_language", ""))
                            
                            with st.expander("Technical Summary"):
                                st.write(summaries.get("technical", ""))
                        
                        # Play audio description
                        if result.get("audio_description") and result["audio_description"].get("success"):
                            st.write("**Audio Description:**")
                            audio_data = result["audio_description"]["audio_base64"]
                            audio_bytes = base64.b64decode(audio_data)
                            st.audio(audio_bytes, format="audio/mp3")
                        
                    else:
                        st.error(f"Request failed: {response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        # Accessibility Guidelines
        st.subheader("ℹ️ Accessibility Information")
        
        st.info("""
        **Available Accessibility Features:**
        
        🔊 **Text-to-Speech**: Convert any text response to natural-sounding speech
        
        🎤 **Voice Input**: Use voice commands to interact with the agent (coming soon)
        
        📢 **Audio Descriptions**: Detailed audio descriptions of charts and visualizations
        
        🔲 **High Contrast Mode**: Enhanced visual contrast for better readability
        
        🔍 **Large Text**: Increased font sizes for better visibility
        
        📱 **Screen Reader Support**: Optimized content structure for screen readers
        
        🌐 **Multi-language Support**: TTS and STT in multiple languages
        """)
        
        # Keyboard shortcuts
        with st.expander("⌨️ Keyboard Shortcuts"):
            st.write("""
            - **Tab**: Navigate between elements
            - **Enter**: Activate buttons and submit forms
            - **Space**: Toggle checkboxes and buttons
            - **Arrow Keys**: Navigate within components
            - **Escape**: Close modals and dropdowns
            """)
        
        # Contact for accessibility support
        st.write("**Need accessibility support?** Contact our team for additional assistance.")

else:
    # Welcome screen
    st.subheader("👋 Welcome to Botlytics!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 🚀 Get Started
        
        1. **Upload** your CSV data using the sidebar
        2. **Ask questions** in natural language
        3. **Get insights** with AI-powered analysis
        4. **View charts** and visualizations
        
        ### ✨ Features
        - Natural language queries
        - Automatic chart generation
        - Secure data processing
        - Real-time analysis
        """)
    
    with col2:
        st.markdown("""
        ### 📝 Example Use Cases
        
        - **Sales Analysis**: "Show me monthly sales trends"
        - **Performance Metrics**: "Which products perform best?"
        - **Regional Insights**: "Compare sales by region"
        - **Time Series**: "Plot revenue over time"
        
        ### 🔒 Privacy & Security
        - Your data is processed securely
        - Files are stored temporarily
        - No data is shared with third parties
        """)
    
    # Sample data option
    st.subheader("🎯 Try with Sample Data")
    if st.button("Load Sample Sales Data"):
        st.info("In a real deployment, this would load sample data automatically.")

# Footer
st.markdown("---")
st.markdown("*Powered by Google Cloud Vertex AI and FastAPI*")