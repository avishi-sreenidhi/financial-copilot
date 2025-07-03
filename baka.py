import streamlit as st
import pandas as pd
import sys
import os
import time
import uuid
import re
import base64
from pathlib import Path
from typing import Dict, Any, List
import io

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from langgraph_engine.graph_builder import build_agent_graph, clear_memory_agent
from agents.pandas_agent import df_manager
from langchain_core.messages import HumanMessage, AIMessage

# Page configuration
st.set_page_config(
    page_title="Financial Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    /* Main chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* User message styling */
    .user-message {
        background-color: #f7f7f8;
        border-radius: 18px;
        padding: 12px 16px;
        margin: 10px 0;
        margin-left: 50px;
        position: relative;
    }
    
    /* Assistant message styling */
    .assistant-message {
        background-color: #ffffff;
        border: 1px solid #e5e5e7;
        border-radius: 18px;
        padding: 12px 16px;
        margin: 10px 0;
        margin-right: 50px;
        position: relative;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Message metadata */
    .message-meta {
        font-size: 0.8em;
        color: #666;
        margin-top: 5px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Avatar styling */
    .avatar {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        color: white;
        margin-right: 10px;
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .assistant-avatar {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    /* Input area styling */
    .input-area {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        border-top: 1px solid #e5e5e7;
        padding: 20px;
        z-index: 1000;
    }
    
    /* File upload area */
    .upload-area {
        border: 2px dashed #d1d5db;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
        background: #f9fafb;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 500;
    }
    
    .status-success {
        background: #d1fae5;
        color: #065f46;
    }
    
    .status-warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    .status-info {
        background: #dbeafe;
        color: #1e40af;
    }
    
    /* Hide Streamlit default elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Custom scrollbar */
    .main::-webkit-scrollbar {
        width: 6px;
    }
    
    .main::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    .main::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 3px;
    }
    
    .main::-webkit-scrollbar-thumb:hover {
        background: #a8a8a8;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitChatInterface:
    """Streamlit-based ChatGPT-style interface for the data analytics agent"""
    
    def __init__(self):
        self.initialize_session_state()
        self.workflow = self.get_workflow()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'uploaded_data' not in st.session_state:
            st.session_state.uploaded_data = None
        if 'data_info' not in st.session_state:
            st.session_state.data_info = {}
    
    @st.cache_resource
    def get_workflow(_self):
        """Get the workflow with caching"""
        return build_agent_graph()
    
    def upload_file_section(self):
        """File upload section in sidebar"""
        st.sidebar.markdown("""
        <div class="sidebar-header">
            <h2>🤖 Financial Copilot</h2>
            <p>Crunch up your numbers and chat with them!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown("### 📁 Upload Data")
        
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your data file to start analysis"
        )
        
        if uploaded_file is not None:
            try:
                # Read the file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Store in dataframe manager
                df_manager.store_dataframe("uploaded_data", df)
                st.session_state.uploaded_data = df
                
                # Store data info
                st.session_state.data_info = {
                    "filename": uploaded_file.name,
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.to_dict(),
                    "memory_usage": df.memory_usage(deep=True).sum(),
                    "upload_time": time.time()
                }
                
                st.sidebar.success(f"✅ File uploaded successfully!")
                
                # Show data info
                with st.sidebar.expander("📊 Data Overview", expanded=True):
                    st.write(f"**File:** {uploaded_file.name}")
                    st.write(f"**Rows:** {df.shape[0]:,}")
                    st.write(f"**Columns:** {df.shape[1]}")
                    st.write(f"**Size:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
                    
                    st.write("**Column Types:**")
                    for dtype, count in df.dtypes.value_counts().items():
                        st.write(f"  • {dtype}: {count} columns")
                
                # Show data preview
                with st.sidebar.expander("👁️ Data Preview"):
                    st.dataframe(df.head(3), use_container_width=True)
                
            except Exception as e:
                st.sidebar.error(f"❌ Error reading file: {str(e)}")
        
        # Data status
        if st.session_state.uploaded_data is not None:
            st.sidebar.markdown("""
            <div class="status-indicator status-success">
                ✅ Data Ready for Analysis
            </div>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.markdown("""
            <div class="status-indicator status-warning">
                ⚠️ No Data Uploaded
            </div>
            """, unsafe_allow_html=True)
    
    def display_conversation_stats(self):
        """Display conversation statistics in sidebar"""
        if st.session_state.conversation_history:
            st.sidebar.markdown("### 📈 Conversation Stats")
            
            total_messages = len(st.session_state.conversation_history)
            context_aware = sum(1 for msg in st.session_state.conversation_history 
                              if msg.get('context_aware', False))
            
            agents_used = set(msg.get('agent', 'unknown') 
                            for msg in st.session_state.conversation_history)
            
            avg_time = sum(msg.get('processing_time', 0) 
                         for msg in st.session_state.conversation_history) / total_messages
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Messages", total_messages)
                st.metric("Avg Time", f"{avg_time:.1f}s")
            
            with col2:
                st.metric("Context Aware", f"{context_aware}/{total_messages}")
                st.metric("Agents Used", len(agents_used))
            
            # Show agents used
            st.sidebar.write("**Agents:**")
            for agent in sorted(agents_used):
                if agent != 'unknown':
                    agent_emoji = {'pandas': '🐼', 'python': '🐍', 'chart': '📊', 'search': '🔍'}.get(agent, '🤖')
                    st.sidebar.write(f"  {agent_emoji} {agent.title()}")
    
    def display_message(self, message: Dict[str, Any], is_user: bool = False):
        """Display a single message in ChatGPT style with support for charts and dataframes"""
        if is_user:
            # User message with simple HTML escaping
            import html
            escaped_content = html.escape(message['content'])
            
            st.markdown(f"""
            <div class="user-message">
                <div style="display: flex; align-items: flex-start;">
                    <div class="avatar user-avatar">👤</div>
                    <div style="flex: 1;">
                        {escaped_content}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Assistant message with rich content rendering
            agent = message.get('agent', 'assistant')
            agent_emoji = {'pandas': '🐼', 'python': '🐍', 'chart': '📊', 'search': '🔍', 'router': '🎯'}.get(agent, '🤖')
            
            content = message['content']
            
            # Clean up HTML and timing indicators
            import html
            import re
            
            # Remove HTML tags and timing patterns
            html_tag_pattern = r'<[^>]*>'
            content = re.sub(html_tag_pattern, '', content)
            content = content.replace('&lt;', '').replace('&gt;', '').replace('&amp;', '&')
            
            timing_text_patterns = [
                r'⏱️\s*\d+\.?\d*s',
                r'⏱️[^0-9]*\d+\.?\d*s',
            ]
            for pattern in timing_text_patterns:
                content = re.sub(pattern, '', content, flags=re.DOTALL)
            
            content = re.sub(r'\s+', ' ', content).strip()
            
            # Check for base64 charts
            chart_data = None
            if "data:image/png;base64," in content:
                pattern = r'data:image/png;base64,([A-Za-z0-9+/=]+)'
                match = re.search(pattern, content)
                if match:
                    chart_data = match.group(1)
                    content = re.sub(pattern, '', content)
            
            # Check for dataframes
            parsed_df = None
            if self._detect_dataframe_output(content):
                parsed_df = self._parse_dataframe_from_text(content)
                if parsed_df is not None:
                    content = self._replace_dataframe_text_with_placeholder(content)
            
            # Display avatar and agent info
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown(f"""
                <div class="avatar assistant-avatar" style="margin: 10px 0;">
                    {agent_emoji}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Render content as rich markdown
                if content.strip():
                    # Process content to improve markdown rendering
                    processed_content = self._process_content_for_display(content)
                    st.markdown(processed_content)
                
                # Display dataframe if found
                if parsed_df is not None:
                    st.subheader("📊 Data Table")
                    st.dataframe(parsed_df, use_container_width=True)
                
                # Display chart if found
                if chart_data:
                    try:
                        img_data = base64.b64decode(chart_data)
                        st.image(img_data, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying chart: {e}")
    
    def _process_content_for_display(self, content: str) -> str:
        """Process content to improve markdown display"""
        # Handle bullet points and formatting
        lines = content.split('\n')
        processed_lines = []
        in_numbered_list = False
        
        for line in lines:
            line = line.strip()
            if not line:
                processed_lines.append('')
                continue
            
            # Handle numbered organization entries (1. **OrgName**: ...)
            if re.match(r'^\d+\.\s*\*\*[^*]+\*\*:', line):
                # Extract organization number and name
                match = re.match(r'^(\d+)\.\s*\*\*([^*]+)\*\*:', line)
                if match:
                    num, org_name = match.groups()
                    processed_lines.append(f"## {num}. {org_name}")
                    in_numbered_list = True
                    continue
            
            # Convert **text**: pattern to proper markdown headers for organization info
            if line.startswith('**') and '**:' in line and not in_numbered_list:
                # Extract organization name and make it a header
                if line.count('**') >= 2:
                    org_name = line.split('**')[1]
                    processed_lines.append(f"### {org_name}")
                    continue
            
            # Convert - **field**: value to nicer format with better styling
            if line.startswith('- **') and '**:' in line:
                # Extract field and value
                parts = line.split('**:', 1)
                if len(parts) >= 2:
                    field = parts[0].replace('- **', '').strip()
                    value = parts[1].strip()
                    # Format specific fields differently
                    if field.lower() in ['description']:
                        processed_lines.append(f"📝 **{field}:** {value}")
                    elif field.lower() in ['founded date', 'last funding date']:
                        processed_lines.append(f"📅 **{field}:** {value}")
                    elif field.lower() in ['last funding amount', 'total funding amount']:
                        processed_lines.append(f"💰 **{field}:** {value}")
                    elif field.lower() in ['number of employees']:
                        processed_lines.append(f"👥 **{field}:** {value}")
                    elif field.lower() in ['headquarters location']:
                        processed_lines.append(f"📍 **{field}:** {value}")
                    elif field.lower() in ['industries']:
                        processed_lines.append(f"🏢 **{field}:** {value}")
                    elif field.lower() in ['operating status']:
                        status_emoji = "✅" if "active" in value.lower() else "❌" if "closed" in value.lower() else "⚠️"
                        processed_lines.append(f"{status_emoji} **{field}:** {value}")
                    else:
                        processed_lines.append(f"**{field}:** {value}")
                    continue
            
            # Regular lines
            processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def _detect_dataframe_output(self, text: str) -> bool:
        """Detect if the text contains tabular dataframe output"""
        indicators = [
            "DATAFRAME_START" in text and "DATAFRAME_END" in text,  # New structured format
            "First " in text and " rows:" in text,
            "Last " in text and " rows:" in text,  
            "Dataset shape:" in text,
            "Correlation matrix:" in text,
            "Value counts for" in text,
            "Missing values:" in text,
            "Describing" in text and "columns" in text,
            # Look for table-like patterns
            "\n   " in text and len([line for line in text.split('\n') if '   ' in line]) > 3,
            # Look for index patterns
            len([line for line in text.split('\n') if re.match(r'^\d+\s+', line.strip())]) > 2
        ]
        return any(indicators)
    
    def _parse_dataframe_from_text(self, text: str) -> pd.DataFrame:
        """Try to parse a dataframe from tabular text output"""
        try:
            # First, try to parse the new structured format
            if "DATAFRAME_START" in text and "DATAFRAME_END" in text:
                start_idx = text.find("DATAFRAME_START") + len("DATAFRAME_START")
                end_idx = text.find("DATAFRAME_END")
                
                if start_idx < end_idx:
                    table_text = text[start_idx:end_idx].strip()
                    
                    # Try to parse using pandas read_csv with StringIO
                    lines = table_text.split('\n')
                    if len(lines) > 1:
                        # Clean up the lines
                        clean_lines = [line.strip() for line in lines if line.strip()]
                        
                        # Try to detect if there's an index column
                        header_line = clean_lines[0]
                        data_lines = clean_lines[1:]
                        
                        # Split by multiple spaces
                        header_parts = re.split(r'\s{2,}', header_line)
                        
                        if len(data_lines) > 0:
                            data_rows = []
                            for line in data_lines[:20]:  # Limit to 20 rows
                                parts = re.split(r'\s{2,}', line)
                                if len(parts) >= 2:
                                    data_rows.append(parts)
                            
                            if data_rows:
                                # Check if first column looks like an index
                                first_col_is_index = all(
                                    row[0].isdigit() or row[0] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
                                    for row in data_rows[:5] if len(row) > 0
                                )
                                
                                if first_col_is_index and len(header_parts) == len(data_rows[0]) - 1:
                                    # First column is index, skip it
                                    columns = header_parts
                                    data = [row[1:len(columns)+1] for row in data_rows]
                                else:
                                    # No index column or header matches data
                                    columns = header_parts[:len(data_rows[0])]
                                    data = [row[:len(columns)] for row in data_rows]
                                
                                # Create dataframe
                                df = pd.DataFrame(data, columns=columns)
                                return df
            
            # Fallback to old parsing method
            lines = text.split('\n')
            
            # Find lines that look like dataframe output
            table_lines = []
            start_collecting = False
            
            for line in lines:
                # Look for typical dataframe patterns
                if (('  ' in line and any(char.isdigit() for char in line)) or 
                    (line.strip().startswith('Date') or line.strip().startswith('0') or 
                     line.strip().startswith('1') or 'dtype:' in line)):
                    start_collecting = True
                    
                if start_collecting:
                    if line.strip() == '' and len(table_lines) > 0:
                        break
                    if 'dtype:' in line:
                        break
                    if line.strip():
                        table_lines.append(line)
            
            if len(table_lines) < 2:
                return None
                
            # Try to parse as space-separated values
            # First, try to identify the header
            potential_header = table_lines[0].split()
            
            if len(potential_header) > 1:
                # Clean up the table lines
                clean_lines = []
                for line in table_lines[1:]:
                    if line.strip() and not line.startswith('dtype:'):
                        clean_lines.append(line)
                
                if clean_lines:
                    # Try to parse the data
                    data_rows = []
                    for line in clean_lines[:20]:  # Limit to first 20 rows
                        # Split by multiple spaces to handle formatted output
                        parts = re.split(r'\s{2,}', line.strip())
                        if len(parts) >= 2:
                            data_rows.append(parts)
                    
                    if data_rows:
                        # Create dataframe
                        max_cols = max(len(row) for row in data_rows)
                        padded_rows = [row + [''] * (max_cols - len(row)) for row in data_rows]
                        
                        # Generate column names
                        if len(potential_header) == max_cols:
                            columns = potential_header
                        else:
                            columns = [f'Column_{i}' for i in range(max_cols)]
                        
                        df = pd.DataFrame(padded_rows, columns=columns[:max_cols])
                        return df
                        
        except Exception as e:
            print(f"Error parsing dataframe from text: {e}")
            return None
        
        return None
    
    def _replace_dataframe_text_with_placeholder(self, text: str) -> str:
        """Replace dataframe text sections with a placeholder"""
        # Handle new structured format
        if "DATAFRAME_START" in text and "DATAFRAME_END" in text:
            start_idx = text.find("DATAFRAME_START")
            end_idx = text.find("DATAFRAME_END") + len("DATAFRAME_END")
            
            before_table = text[:start_idx]
            after_table = text[end_idx:]
            
            return before_table + "[Interactive table displayed below]" + after_table
        
        # Fallback to old method
        lines = text.split('\n')
        result_lines = []
        in_table = False
        
        for line in lines:
            # Detect start of table
            if (('  ' in line and any(char.isdigit() for char in line)) or 
                (line.strip().startswith('Date') or line.strip().startswith('0') or 
                 line.strip().startswith('1'))):
                if not in_table:
                    result_lines.append("[Interactive table displayed below]")
                    in_table = True
                continue
            
            # Detect end of table
            if in_table and (line.strip() == '' or 'dtype:' in line):
                in_table = False
                continue
                
            if not in_table:
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def send_message(self, user_message: str) -> Dict[str, Any]:
        """Send a message to the agent and get response"""
        # Add user message to conversation
        user_msg = HumanMessage(content=user_message)
        st.session_state.messages.append(user_msg)
        
        # Create state for this interaction
        state = {
            "query": user_message,
            "messages": st.session_state.messages.copy(),
            "next_agent": "",
            "current_agent": "",
            "agent_outputs": {},
            "dataframe_info": st.session_state.data_info,
            "has_data": st.session_state.uploaded_data is not None,
            "final_result": "",
            "metadata": {},
            "iteration_count": 0,
            "chat_response": {},
            "session_id": st.session_state.session_id,
            "conversation_summary": ""
        }
        
        try:
            # Show processing indicator
            with st.spinner('🤖 Thinking...'):
                start_time = time.time()
                result = self.workflow.invoke(state)
                end_time = time.time()
            
            # Extract response
            chat_response = result.get("chat_response", {})
            agent_response = chat_response.get("message", result.get("final_result", "Sorry, I couldn't process your request."))
            
            # Add agent response to conversation
            agent_msg = AIMessage(content=agent_response)
            st.session_state.messages.append(agent_msg)
            
            # Store conversation entry
            conversation_entry = {
                "content": agent_response,
                "agent": chat_response.get("agent", "assistant"),
                "context_aware": chat_response.get("context_aware", False),
                "processing_time": end_time - start_time,
                "timestamp": time.time(),
                "conversation_summary": chat_response.get("conversation_summary", "")
            }
            
            st.session_state.conversation_history.append(conversation_entry)
            
            return conversation_entry
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            error_response = {
                "content": f"Sorry, I encountered an error: {str(e)}",
                "agent": "system",
                "context_aware": False,
                "processing_time": 0,
                "timestamp": time.time(),
                "error": True
            }
            
            # Add error response to conversation
            error_msg = AIMessage(content=error_response["content"])
            st.session_state.messages.append(error_msg)
            
            return error_response
    
    def clear_conversation(self):
        """Clear the conversation history and memory"""
        old_session_id = st.session_state.session_id
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.session_id = str(uuid.uuid4())
        
        # Clear memory from the memory agent
        try:
            clear_memory_agent(old_session_id)
            print(f"Cleared conversation and memory for session: {old_session_id}")
        except Exception as e:
            print(f"Error clearing memory: {e}")
        
        st.rerun()
    
    def clear_all_memory(self):
        """Clear all memory across all sessions"""
        try:
            clear_memory_agent()  # Clear all sessions
            st.success("✅ All memory cleared successfully!")
        except Exception as e:
            st.error(f"❌ Error clearing memory: {e}")
        st.rerun()
    
    def export_conversation(self):
        """Export conversation as text"""
        if not st.session_state.conversation_history:
            return ""
        
        export_text = f"# Data Analytics AI Conversation\n"
        export_text += f"Session ID: {st.session_state.session_id}\n"
        export_text += f"Exported: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if st.session_state.data_info:
            export_text += f"## Data Information\n"
            export_text += f"File: {st.session_state.data_info.get('filename', 'N/A')}\n"
            export_text += f"Shape: {st.session_state.data_info.get('shape', 'N/A')}\n\n"
        
        export_text += f"## Conversation\n\n"
        
        # Reconstruct conversation from messages
        for i in range(0, len(st.session_state.messages), 2):
            if i < len(st.session_state.messages):
                user_msg = st.session_state.messages[i]
                export_text += f"**User:** {user_msg.content}\n\n"
            
            if i + 1 < len(st.session_state.messages):
                assistant_msg = st.session_state.messages[i + 1]
                # Find corresponding conversation entry
                conv_idx = i // 2
                if conv_idx < len(st.session_state.conversation_history):
                    conv_entry = st.session_state.conversation_history[conv_idx]
                    agent = conv_entry.get('agent', 'assistant')
                    context = " (Context Aware)" if conv_entry.get('context_aware') else ""
                    time_taken = conv_entry.get('processing_time', 0)
                    export_text += f"**Assistant ({agent}{context}) [{time_taken:.1f}s]:** {assistant_msg.content}\n\n"
                else:
                    export_text += f"**Assistant:** {assistant_msg.content}\n\n"
            
            export_text += "---\n\n"
        
        return export_text
    
    def run(self):
        """Main application loop"""
        # Title
        st.title("🤖 Financial Copilot")
        st.markdown("*Your ChatGPT-style companion for Financial data analysis and exploration*")
        
        # Sidebar
        self.upload_file_section()
        
        # Conversation controls
        st.sidebar.markdown("### 💬 Conversation")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                self.clear_conversation()
        
        with col2:
            if st.session_state.conversation_history:
                export_text = self.export_conversation()
                st.download_button(
                    "📥 Export",
                    data=export_text,
                    file_name=f"chat_export_{st.session_state.session_id[:8]}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        
        # Memory management
        st.sidebar.markdown("### 🧠 Memory Management")
        col3, col4 = st.sidebar.columns(2)
        
        with col3:
            if st.button("🧹 Clear Memory", use_container_width=True, help="Clear conversation memory but keep chat history"):
                self.clear_all_memory()
        
        with col4:
            if st.button("🔄 Fresh Start", use_container_width=True, help="Clear everything and start completely fresh"):
                self.clear_conversation()
                st.session_state.uploaded_data = None
                st.session_state.data_info = {}
                st.rerun()
        
        # Display conversation stats
        self.display_conversation_stats()
        
        # Main chat area
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Welcome message if no conversation
        if not st.session_state.messages:
            st.markdown("""
            <div class="assistant-message">
                <div style="display: flex; align-items: flex-start;">
                    <div class="avatar assistant-avatar">💸</div>
                    <div style="flex: 1;">
                        Welcome! I am your Financial CoPilot! I specialize in analyzing financial datasets, uncovering trends, and generating powerful insights to guide business decisions.
                        <br><br>
                        <strong>To get started:</strong>
                        <ul>
                            <li>📁 <strong>Upload</strong> your financial data (CSV or Excel) from the sidebar</li>
                            <li>📈 <strong>Ask questions</strong> like “What’s the ROI trend?” or “Compare Q1 vs Q2 revenue”</li>
                            <li>💹 <strong>Request charts</strong> for cash flow, burn rate, profit margins, and more</li>
                            <li>🐍 <strong>Generate or debug Python code</strong> for advanced financial analysis</li>
                        </ul>
                        Whether you're analyzing income statements, tracking expenses, or forecasting revenue — I'm here to help you make sense of the numbers.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display conversation history
        for i in range(0, len(st.session_state.messages), 2):
            # User message
            if i < len(st.session_state.messages):
                user_msg = st.session_state.messages[i]
                self.display_message({"content": user_msg.content}, is_user=True)
            
            # Assistant message
            if i + 1 < len(st.session_state.messages):
                assistant_msg = st.session_state.messages[i + 1]
                # Find corresponding conversation entry for metadata
                conv_idx = i // 2
                if conv_idx < len(st.session_state.conversation_history):
                    conv_entry = st.session_state.conversation_history[conv_idx]
                    message_data = {
                        "content": assistant_msg.content,
                        "agent": conv_entry.get("agent", "assistant"),
                        "context_aware": conv_entry.get("context_aware", False),
                        "processing_time": conv_entry.get("processing_time", 0)
                    }
                else:
                    message_data = {"content": assistant_msg.content}
                
                self.display_message(message_data)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        st.markdown("---")
        
        # Input area
        user_input = st.chat_input(
            "Type your message here...",
            key="chat_input"
        )
        
        if user_input:
            # Display user message immediately
            self.display_message({"content": user_input}, is_user=True)
            
            # Process and get response
            response = self.send_message(user_input)
            
            # Display assistant response
            self.display_message(response)
            
            # Rerun to update the interface
            st.rerun()

# Create and run the interface
if __name__ == "__main__":
    app = StreamlitChatInterface()
    app.run()
else:
    # When imported by Streamlit
    app = StreamlitChatInterface()
    app.run()
