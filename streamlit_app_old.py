import streamlit as st
import asyncio
import sys
import signal
import atexit
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
import os
from dotenv import load_dotenv
import threading
from io import StringIO
import contextlib

load_dotenv()

# Configure page
st.set_page_config(
    page_title="MagenticOne AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        background-color: #f8f9fa;
        max-height: 600px;
        overflow-y: auto;
    }
    
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        margin-left: 2rem;
        display: inline-block;
        max-width: 80%;
        float: right;
        clear: both;
    }
    
    .assistant-message {
        background-color: #e9ecef;
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        margin-right: 2rem;
        display: inline-block;
        max-width: 80%;
        float: left;
        clear: both;
    }
    
    .status-message {
        background-color: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 10px;
        margin: 0.3rem 0;
        font-style: italic;
        text-align: center;
    }
    
    .error-message {
        background-color: #dc3545;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 10px;
        margin: 0.3rem 0;
        text-align: center;
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

class StreamlitConsole:
    """Custom console class that captures output and displays it in Streamlit"""
    
    def __init__(self, container):
        self.container = container
        self.output_buffer = []
        
    async def __call__(self, stream):
        """Process the stream and capture output"""
        try:
            async for message in stream:
                # Capture the message content
                content = str(message)
                self.output_buffer.append(content)
                
                # Update the display in real-time
                with self.container:
                    st.write("🔄 Processing...")
                    for output in self.output_buffer[-5:]:  # Show last 5 outputs
                        st.text(output[:200] + "..." if len(output) > 200 else output)
                        
        except Exception as e:
            self.output_buffer.append(f"Error in stream processing: {str(e)}")

async def process_with_magnetic_one(task, output_container, status_container):
    """Process the task with MagenticOne"""
    surfer = None
    try:
        with status_container:
            st.info("🔄 Initializing AI models...")
        
        # Initialize the model client
        model_client = AzureOpenAIChatCompletionClient(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_API_VERSION")
        )

        with status_container:
            st.info("🌐 Starting web surfer...")

        # Initialize MultimodalWebSurfer
        surfer = MultimodalWebSurfer(
            "MultimodalWebSurfer",
            model_client=model_client,
            downloads_folder="./downs",
            debug_dir="./debug",
            headless=True,  # Set to headless for Streamlit
            to_resize_viewport=True,
            description="A web surfing assistant that can browse and interact with web pages.",
            start_page="https://www.bing.com",
            animate_actions=False,  # Disable animations for better performance
            browser_data_dir="./browser_data",
        )

        with status_container:
            st.info("🤖 Creating AI team...")

        # Create the team
        team = MagenticOneGroupChat([surfer], model_client=model_client)
        
        with status_container:
            st.success("✅ Processing your request...")
        
        # Create custom console for capturing output
        console = StreamlitConsole(output_container)
        
        # Process the task
        result_stream = team.run_stream(task=task)
        await console(result_stream)
        
        with status_container:
            st.success("✅ Task completed successfully!")
        
        return "Task completed successfully!"
        
    except Exception as e:
        error_msg = f"Error processing task: {str(e)}"
        with status_container:
            st.error(f"❌ {error_msg}")
        return error_msg
        
    finally:
        # Cleanup
        try:
            if surfer and hasattr(surfer, 'close'):
                await surfer.close()
            elif surfer and hasattr(surfer, '_browser') and surfer._browser:
                await surfer._browser.close()
        except Exception as cleanup_error:
            st.warning(f"Cleanup warning: {cleanup_error}")
        
        # Cancel remaining tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(0.1)

def run_async_task(task, output_container, status_container):
    """Run async task in a separate thread"""
    try:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the task
        result = loop.run_until_complete(
            process_with_magnetic_one(task, output_container, status_container)
        )
        
        return result
    except Exception as e:
        return f"Thread error: {str(e)}"
    finally:
        try:
            loop.close()
        except:
            pass

# Main UI
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 MagenticOne AI Assistant</h1>
        <p>Powered by Azure OpenAI and Web Browsing Capabilities</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar with configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Environment variables status
        st.subheader("Environment Status")
        env_vars = [
            "AZURE_OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_ENDPOINT", 
            "AZURE_OPENAI_KEY",
            "AZURE_API_VERSION"
        ]
        
        for var in env_vars:
            if os.getenv(var):
                st.success(f"✅ {var}")
            else:
                st.error(f"❌ {var}")
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Chat container
        chat_container = st.container(height=600)
        
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="assistant-message">
                        <strong>Assistant:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask me anything... (e.g., 'Summarize the latest AI research papers')", disabled=st.session_state.is_processing):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.is_processing = True
            st.rerun()
    
    with col2:
        st.header("📊 Status & Output")
        
        # Status container
        status_container = st.container()
        
        # Output container
        st.subheader("🔍 Processing Details")
        output_container = st.container(height=400)
        
        if not st.session_state.is_processing:
            with status_container:
                st.info("💭 Ready to process your request...")
            with output_container:
                st.write("Processing details will appear here...")

    # Process the latest message if we're in processing state
    if st.session_state.is_processing and st.session_state.messages:
        latest_message = st.session_state.messages[-1]
        if latest_message["role"] == "user":
            
            # Run the async task in a thread
            with st.spinner("🤖 AI is thinking..."):
                try:
                    result = run_async_task(
                        latest_message["content"], 
                        output_container, 
                        status_container
                    )
                    
                    # Add assistant response
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": result
                    })
                    
                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"Sorry, I encountered an error: {str(e)}"
                    })
                
                finally:
                    st.session_state.is_processing = False
                    st.rerun()

if __name__ == "__main__":
    main()
