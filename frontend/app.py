import streamlit as st
import requests
import pandas as pd
import json
from PIL import Image
import io

# Configuration
API_BASE_URL = "http://localhost:8080"  # Update for production

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
    
    # Query interface
    st.subheader("💬 Ask Questions About Your Data")
    
    # Example queries
    with st.expander("💡 Example Queries"):
        st.write("""
        - "What are the total sales by product?"
        - "Show me sales trends over time"
        - "Which region has the highest sales?"
        - "Create a chart showing sales by category"
        - "What's the average sales per day?"
        """)
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="e.g., What are the total sales by product?",
        height=100
    )
    
    if st.button("🚀 Analyze", type="primary"):
        if query.strip():
            with st.spinner("🤖 AI is analyzing your data..."):
                try:
                    payload = {
                        "dataset_id": st.session_state.dataset_id,
                        "query": query
                    }
                    
                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/query",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        st.subheader("📊 Analysis Results")
                        
                        # Summary
                        st.write("**Summary:**")
                        st.write(result['summary'])
                        
                        # Results preview
                        if result.get('preview'):
                            st.write("**Data Results:**")
                            if isinstance(result['preview'], list):
                                df_result = pd.DataFrame(result['preview'])
                                st.dataframe(df_result, use_container_width=True)
                            else:
                                st.write(result['preview'])
                        
                        # Chart
                        if result.get('chart_url'):
                            st.write("**Visualization:**")
                            st.info("Chart generated and saved to cloud storage")
                            st.write(f"Chart URL: {result['chart_url']}")
                        
                        # Code executed (for transparency)
                        if result.get('code_executed'):
                            with st.expander("🔍 Code Executed"):
                                st.code(result['code_executed'], language='python')
                        
                    else:
                        st.error(f"Query failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question about your data.")

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