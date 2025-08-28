import streamlit as st
import asyncio
import sys
import atexit
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
import os
from dotenv import load_dotenv
import threading
from datetime import datetime
import time

load_dotenv()

# Configure page
st.set_page_config(
    page_title="🤖 MagenticOne AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .user-message {
        background: linear-gradient(135deg, #007bff, #0056b3);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        margin-left: 3rem;
        box-shadow: 0 3px 10px rgba(0,123,255,0.3);
        animation: slideInRight 0.3s ease;
    }
    
    .assistant-message {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        color: #333;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        margin-right: 3rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #28a745;
        animation: slideInLeft 0.3s ease;
    }
    
    .status-success {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 25px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(40,167,69,0.3);
    }
    
    .status-processing {
        background: linear-gradient(135deg, #ffc107, #fd7e14);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 25px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
        animation: pulse 1.5s infinite;
    }
    
    .status-error {
        background: linear-gradient(135deg, #dc3545, #c82333);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 25px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 2px 8px rgba(220,53,69,0.3);
    }
    
    .output-container {
        background: linear-gradient(145deg, #1e1e1e, #2d2d2d);
        color: #00ff00;
        padding: 1rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #444;
        white-space: pre-wrap;
    }
    
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f1f3f4);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .welcome-container {
        text-align: center; 
        padding: 2rem; 
        color: #666;
        background: linear-gradient(145deg, #f8f9fa, #ffffff);
        border-radius: 15px;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "processing_logs" not in st.session_state:
    st.session_state.processing_logs = []

async def process_with_magnetic_one(user_input, status_placeholder, output_placeholder):
    """Process user input with MagenticOne"""
    surfer = None
    try:
        # Update status
        status_placeholder.markdown("""
        <div class="status-processing">
            🔄 Initializing Azure OpenAI client...
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize model client
        model_client = AzureOpenAIChatCompletionClient(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_API_VERSION")
        )
        
        # Update status
        status_placeholder.markdown("""
        <div class="status-processing">
            🌐 Starting MultimodalWebSurfer...
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize web surfer
        surfer = MultimodalWebSurfer(
            "MultimodalWebSurfer",
            model_client=model_client,
            downloads_folder="./downs",
            debug_dir="./debug",
            headless=True,  # Headless for Streamlit
            to_resize_viewport=True,
            description="A web surfing assistant that can browse and interact with web pages.",
            start_page="https://www.bing.com",
            animate_actions=False,
            browser_data_dir="./browser_data",
        )
        
        # Update status
        status_placeholder.markdown("""
        <div class="status-processing">
            🤖 Creating MagenticOne team...
        </div>
        """, unsafe_allow_html=True)
        
        # Create team
        team = MagenticOneGroupChat([surfer], model_client=model_client)
        
        # Update status
        status_placeholder.markdown("""
        <div class="status-processing">
            ✨ Processing your request...
        </div>
        """, unsafe_allow_html=True)
        
        # Process the request
        result_parts = []
        output_text = ""
        
        async for message in team.run_stream(task=user_input):
            # Capture output
            message_str = str(message)
            result_parts.append(message_str)
            
            # Create a formatted output display
            timestamp = datetime.now().strftime("%H:%M:%S")
            output_text += f"[{timestamp}] {message_str}\n"
            
            # Update output display (keep last 2000 characters to prevent overflow)
            display_text = output_text[-2000:] if len(output_text) > 2000 else output_text
            output_placeholder.markdown(f"""
            <div class="output-container">{display_text}</div>
            """, unsafe_allow_html=True)
        
        # Final result
        final_result = "\n".join(result_parts) if result_parts else "Task completed successfully!"
        
        # Update status to success
        status_placeholder.markdown("""
        <div class="status-success">
            ✅ Task completed successfully!
        </div>
        """, unsafe_allow_html=True)
        
        return final_result
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        status_placeholder.markdown(f"""
        <div class="status-error">
            ❌ {error_msg}
        </div>
        """, unsafe_allow_html=True)
        return error_msg
        
    finally:
        # Cleanup
        try:
            if surfer:
                if hasattr(surfer, 'close'):
                    await surfer.close()
                elif hasattr(surfer, '_browser') and surfer._browser:
                    await surfer._browser.close()
        except Exception as cleanup_error:
            st.warning(f"Cleanup warning: {cleanup_error}")
        
        # Cancel remaining tasks
        try:
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0.1)
        except:
            pass

def run_magnetic_one_async(user_input, status_placeholder, output_placeholder):
    """Run MagenticOne in a separate thread"""
    try:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the async function
        result = loop.run_until_complete(
            process_with_magnetic_one(user_input, status_placeholder, output_placeholder)
        )
        
        return result
        
    except Exception as e:
        return f"Thread error: {str(e)}"
    finally:
        try:
            loop.close()
        except:
            pass

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 MagenticOne AI Assistant</h1>
        <p>Powered by Azure OpenAI • Web Browsing • Multi-Agent Collaboration</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Environment status
        st.subheader("🔐 Environment Status")
        env_vars = {
            "AZURE_OPENAI_DEPLOYMENT": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "AZURE_OPENAI_KEY": "••••••••" if os.getenv("AZURE_OPENAI_KEY") else None,
            "AZURE_API_VERSION": os.getenv("AZURE_API_VERSION")
        }
        
        for var, value in env_vars.items():
            if value:
                st.markdown(f"""
                <div class="metric-card">
                    <strong>✅ {var}</strong><br>
                    <small>{value}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"❌ {var} not set")
        
        st.divider()
        
        # Stats
        st.subheader("📊 Session Stats")
        st.metric("Messages", len(st.session_state.messages))
        st.metric("Processing", "Yes" if st.session_state.is_processing else "No")
        
        st.divider()
        
        # Controls
        if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.processing_logs = []
            st.rerun()
        
        if st.button("🔄 Reset Session", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Main layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Chat container with custom styling
        chat_container = st.container(height=600)
        
        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                <div class="welcome-container">
                    <h3>👋 Welcome to MagenticOne!</h3>
                    <p>Ask me anything and I'll help you with web research, analysis, and more.</p>
                    <p><strong>Try asking:</strong></p>
                    <ul style="text-align: left; max-width: 400px; margin: 0 auto;">
                        <li>"Summarize the latest AI research papers"</li>
                        <li>"What's trending in technology today?"</li>
                        <li>"Find information about quantum computing"</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.markdown(f"""
                        <div class="user-message">
                            <strong>🧑‍💻 You:</strong><br>
                            {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="assistant-message">
                            <strong>🤖 MagenticOne:</strong><br>
                            {message["content"]}
                        </div>
                        """, unsafe_allow_html=True)
    
    with col2:
        st.header("📊 Processing Status")
        
        # Status display
        status_container = st.empty()
        
        st.subheader("🔍 Output Monitor")
        output_container = st.empty()
        
        if not st.session_state.is_processing:
            status_container.markdown("""
            <div class="status-success">
                💭 Ready for your next request...
            </div>
            """, unsafe_allow_html=True)
            
            output_container.markdown("""
            <div class="output-container">Waiting for processing to begin...

This monitor will show real-time output from the AI agents
as they work on your request.

You'll see:
• Agent initialization
• Web browsing activities  
• AI reasoning processes
• Task completion status</div>
            """, unsafe_allow_html=True)

    # Chat input at the bottom
    if user_input := st.chat_input(
        "💭 Ask me anything... (e.g., 'Find the latest news about AI')", 
        disabled=st.session_state.is_processing
    ):
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input
        })
        
        # Set processing state
        st.session_state.is_processing = True
        st.rerun()

    # Process the request if we're in processing state
    if st.session_state.is_processing and st.session_state.messages:
        latest_message = st.session_state.messages[-1]
        
        if latest_message["role"] == "user":
            try:
                # Process in background thread
                result = run_magnetic_one_async(
                    latest_message["content"],
                    status_container,
                    output_container
                )
                
                # Add response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result
                })
                
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I encountered an error: {str(e)}"
                })
            
            finally:
                st.session_state.is_processing = False
                st.rerun()

if __name__ == "__main__":
    main()
