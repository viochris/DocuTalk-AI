import os
import json
import tempfile
import streamlit as st
from google import genai
from function import init_state, change_on_api_key, reset_state, reset_agent, process_file
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationSummaryMemory
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

st.set_page_config(
    page_title="DocuTalk-AI | Smart Document Assistant", 
    page_icon="📚", 
    layout="wide"
)

st.title("🤖 DocuTalk-AI: Interactive Document Intelligence")
st.markdown(
    """
    **Unlock insights from your documents.** Simply upload your files (PDF, CSV, MD, TXT) via the sidebar to create a personal knowledge base. 
    Once processed, you can chat, query, and analyze your content instantly using AI.
    """
)

# Initialize session state variables
init_state()

with st.sidebar:
    st.header("⚙️ System Configuration")

    # API Key Input with clearer label and security tooltip
    st.text_input(
        "🔑 Google Gemini API Key", 
        type="password",
        key="google_api_key",
        on_change=change_on_api_key,
        help="Paste your Google AI Studio API key here. It is required to power the AI models."
    )

    # Language selection with descriptive label and help text
    chosen_language = st.selectbox(
        "🌐 Response Language",
        options=["English", "Indonesian"],
        index=0,
        on_change=reset_agent,
        help="Select the language you want the AI to use when answering your questions."
    )

    st.button(
        "🔄 Reset Conversation", 
        help="Resets the conversation history and clears the current session memory.", 
        on_click=reset_state
    )

    st.divider()

    st.header("🧠 Model Settings")
    creativity = st.slider(
        "🌡️ Creativity Level (Temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        on_change= reset_agent,
        help="Lower values (0.0) make the AI more factual and strict. Higher values (0.7+) make it more creative."    
    )

    st.divider()

    uploaded_files = st.file_uploader(
        label="📂 Upload Reference Documents",
        type=["pdf", "csv", "md", "txt"],
        accept_multiple_files=True,
        help="Upload PDF, CSV, Markdown, or Text files to build your knowledge base. You can select multiple files at once."
    )

    # Feedback on file upload status
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} document(s) uploaded. Ready to process.")
    else:
        st.info("ℹ️ Awaiting documents. Upload PDFs to begin analysis.")
    
    process_btn = st.button("🚀 Process & Embed Documents")

if not st.session_state["google_api_key"]:
    st.warning("⚠️ Access Denied: Please provide a valid Google API Key in the sidebar.")

if process_btn and uploaded_files:
    try:
        new_docs = []
        new_filenames = []

        # Identify which files are actually new (not yet processed)
        files_to_process = []
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                files_to_process.append(uploaded_file)

        # If no new files are found, inform the user
        if not files_to_process:
            st.toast("All uploaded documents are already in the knowledge base.", icon="ℹ️")
        else:
            # Display a spinner while processing
            with st.spinner(f"Processing {len(files_to_process)} new document(s)..."):
                
                # Iterate through files to create temporary paths and load data
                for uploaded_file in files_to_process:
                    # Load the content
                    docs = process_file(uploaded_file)

                    if docs is None:
                        # Inform the user why this file is skipped
                        st.warning(f"⚠️ Skipped '{uploaded_file.name}': Format not supported or file is empty.")
                        continue

                    # Filter and add metadata to valid pages
                    valid_docs = []
                    for doc in docs:
                        # Check if the page content is not empty or just whitespace
                        if doc.page_content and doc.page_content.strip():
                            doc.metadata["source"] = uploaded_file.name
                            valid_docs.append(doc)

                    # Handle case where the file contains no extractable text
                    if not valid_docs:
                        st.warning(f"⚠️ The file '{uploaded_file.name}' appears empty or unreadable (possibly a scanned PDF). Skipping this file.")
                        continue

                    # Add only the validated documents to the processing list
                    new_docs.extend(valid_docs)
                    new_filenames.append(uploaded_file.name)

                # Only proceed if there is text content extracted
                if new_docs:
                    # Split text into manageable chunks for the AI
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    text_chunks = text_splitter.split_documents(new_docs)

                    # Initialize the Embedding Model
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/gemini-embedding-001", 
                        google_api_key=st.session_state["google_api_key"]
                    )

                    # Check if the Vector Database exists
                    if st.session_state.vectorstore is None:
                        # If not, create a new one
                        st.session_state.vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)
                    else:
                        # If yes, add the new documents to the existing database (Append)
                        st.session_state.vectorstore.add_documents(text_chunks)

                    # Mark these files as processed in the session state
                    st.session_state.processed_files.extend(new_filenames)

                    # Reset the agent executor to ensure it uses the updated vector store
                    st.session_state.pop("agent_executor", None)
                    
                    # Notify success
                    st.toast(f"Successfully added: {', '.join(new_filenames)}", icon="✅")
                else:
                    # If new_docs is empty (all uploaded files were empty/unreadable)
                    st.info("ℹ️ No valid text could be extracted from the uploaded files.")

    # --- ADVANCED ERROR HANDLING ---
    except Exception as e:
        error_message = str(e)
        
        # 1. Check for Quota/Rate Limit issues
        if "429" in error_message or "Quota exceeded" in error_message:
            st.error("🚨 **Oops! API Quota Exceeded**")
            st.warning(
                """
                **Explanation:** You have reached the free usage limit for the Google Gemini API.
                
                **Solutions:**
                1. Wait for a few minutes.
                2. Try again tomorrow or use a different API Key.
                """
            )
        
        # 2. Check for Invalid API Key
        elif "API key not valid" in error_message or "403" in error_message:
            st.error("🔑 **Invalid API Key**")
            st.info("Please ensure you have entered the correct Google API Key in the sidebar.")

        # 3. Handle other errors (Network, parsing, etc.)
        else:
            st.error("⚠️ **System Error Occurred**")
            st.write(f"**Error Details:** `{error_message}`")
        
        # Stop execution so the app doesn't look broken below
        st.stop()

# Check if the agent is not yet initialized AND if the vector database exists
if "agent_executor" not in st.session_state \
    and st.session_state.vectorstore is not None:

    try:
        # 1. Configure the Retriever (Fetch top 3 most relevant chunks)
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={'k': 3})

        # 2. Create the Retriever Tool for the Agent
        retriever_tool = create_retriever_tool(
            retriever,
            name="search_info_in_document",
            # Description is set to "Authoritative" so the LLM feels compelled to check here first
            description="The PRIMARY SOURCE OF TRUTH. Contains the full content of user's uploaded documents (PDF, CSV, MD, TXT). You MUST query this tool first before answering any question."
        )

        # 3. Initialize the LLM (Google Gemini)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=st.session_state["google_api_key"],
            temperature=creativity
        )

        # 4. Define the Toolkit (Web Search + Document Search)
        tools = [DuckDuckGoSearchRun(name="Web Search"), retriever_tool]

        # 5. Pull and Customize the Prompt Template
        prompt_agent = hub.pull("hwchase17/react-chat")

        # Enforce priority on document search AND inject Language settings AND FORMATTING
        # 👇 INFO: Use f-string (f""") so Python can read the {chosen_language} variable
        # Enforce priority on document search AND inject Language settings AND FORMATTING
        prefix_prompt = f"""
        You are a highly capable research assistant. 
        
        *** CRITICAL FORMATTING RULES (YOU MUST FOLLOW THIS) ***
        You are using the ReAct (Reasoning + Acting) framework. 
        Every time you want to use a tool, you MUST strictly use this exact structure (no extra text):

        Thought: [Your reasoning process]
        Action: [The tool name, e.g., search_info_in_document]
        Action Input: [The input query]
        Observation: [The result from the tool]

        If you have enough information to answer the user, or if you don't need a tool:
        
        Thought: I have the answer.
        Final Answer: [Your final response in {chosen_language}]

        ----------------------------------------------------------------
        
        *** WORKFLOW PRIORITIES ***
        1. **PRIORITY 1 - CHECK DOCUMENTS:** ALWAYS start by using the 'search_info_in_document' tool.
        2. **PRIORITY 2 - INTERNAL KNOWLEDGE:** Use internal knowledge only if documents fail.
        3. **PRIORITY 3 - WEB SEARCH:** Use 'Web Search' as a last resort.

        *** CRITICAL LANGUAGE RULE ***
        The user has explicitly selected to receive answers in: **{chosen_language}**.
        
        You MUST strictly follow this translation logic for EVERY response:
        1. **FINAL OUTPUT:** Your thought process and final answer MUST be in **{chosen_language}**.
        2. **INPUT INDEPENDENCE:** It does NOT matter what language the uploaded documents or the user's question are in.
           - IGNORE the language of the provided documents.
           - IGNORE the language of the user's question.
        3. **TRANSLATION EXAMPLES:**
           - If documents/question are in **English** and target is **Indonesian** -> YOU MUST ANSWER IN **INDONESIAN**.
           - If documents/question are in **Indonesian** and target is **English** -> YOU MUST ANSWER IN **ENGLISH**.

        NEVER skip Step 1. Your primary goal is to answer based on the documents.
        """
        
        # Merge the new instruction with the default React Agent template
        prompt_agent.template = prefix_prompt + "\n\n" + prompt_agent.template

        # 6. Create the Agent (Brain)
        agent_brain = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt_agent
        )

        # 7. Initialize the Agent Executor (Runtime with Memory)
        st.session_state.agent_executor = AgentExecutor(
            agent=agent_brain,
            tools=tools,
            memory=ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True),
            handle_parsing_errors=True
        )

    # --- ADVANCED ERROR HANDLING FOR AGENT INIT ---
    except Exception as e:
        error_message = str(e)
        
        # 1. Check for Quota/Rate Limit issues (Common with Gemini)
        if "429" in error_message or "Quota exceeded" in error_message:
            st.error("🚨 **Agent Initialization Failed: API Quota Exceeded**")
            st.warning(
                """
                **Explanation:** The Google Gemini API is currently unavailable due to usage limits.
                
                **Solution:** Please wait a moment before trying to chat, or try again tomorrow.
                """
            )

        # 2. Check for Invalid API Key
        elif "API key not valid" in error_message or "403" in error_message:
            st.error("🔑 **Agent Initialization Failed: Invalid API Key**")
            st.info("Please check the API Key entered in the sidebar.")

        # 3. Check for Network/Hub Issues (hub.pull often fails on restricted networks)
        elif "Connection error" in error_message or "Failed to establish" in error_message:
             st.error("🌐 **Network Error**")
             st.write("Failed to connect to LangChain Hub or Google API. Please check your internet connection.")

        # 4. Handle generic errors
        else:
            st.error("⚠️ **Failed to Initialize AI Agent**")
            st.write(f"**Error Details:** `{error_message}`")
        
        # Critical failure: Stop the app because we can't chat without an agent
        st.stop()

if "agent_executor" in st.session_state \
    and st.session_state.agent_executor is not None \
    and st.session_state.vectorstore is not None:

    with st.expander("📝 Recommended Questions (Click to Copy)", expanded=False):
        
        # Split the layout into 2 columns for better readability
        col_help1, col_help2 = st.columns(2)
        
        with col_help1:
            st.markdown("### 📑 Summarization & Key Points")
            st.code("Summarize the entire document in 3 concise paragraphs.", language="text")
            st.code("What are the 5 main takeaways from this document?", language="text")
            st.code("Explain the main objective and conclusion of this file.", language="text")

            st.markdown("### 🔍 Specific Details")
            st.code("What does the document say about [Specific Topic]?", language="text")
            st.code("Extract all dates, deadlines, and timeline events mentioned.", language="text")
            
        with col_help2:
            st.markdown("### 📊 Analysis & Methodology")
            st.code("What methodology or approach was used in this study?", language="text")
            st.code("What are the limitations or risks mentioned in the text?", language="text")
            st.code("Compare the pros and cons discussed in the document.", language="text")

            st.markdown("### 🎨 Tone & Structure")
            st.code("Analyze the tone of the writer (e.g., formal, persuasive).", language="text")
            st.code("Who is the intended audience for this document?", language="text")
            
        st.caption("💡 **Tip:** Click the copy icon on the right of any box above, then paste it into the chat!")
        
# Initialize chat history list if it doesn't exist yet
if "message" not in st.session_state:
    st.session_state["message"] = []

# # Display previous chat history (renders messages from session state)y
for msg in st.session_state["message"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Capture user input from the chat bar
if prompt_text := st.chat_input(placeholder="Ask specific questions about your documents..."):

    # Validation 1: Check if documents are processed (Vector Database)
    if st.session_state.vectorstore is None:
        st.error("⚠️ Knowledge base is empty. Please upload a file (PDF, CSV, MD, TXT) and click 'Process & Embed Documents' first.")
    
    # Validation 2: Check if the AI Agent is initialized
    elif not st.session_state.agent_executor:
        st.error("⚠️ AI Agent is not ready. Please try clicking 'Reset Conversation' to reset the connection.")

    else: 
        # 1. Add User's message to history and display it
        st.session_state["message"].append({"role": "human", "content": prompt_text})
        with st.chat_message("human"):
            st.markdown(prompt_text)

        # 2. Generate AI Response
        with st.chat_message("ai"):
            try:
                # 1. Initialize the Streamlit Callback Handler
                # This creates a dynamic container to visualize the Agent's internal "thought process" 
                # (e.g., "Thinking...", "Searching Document...", "Found X") in real-time.
                st_callback = StreamlitCallbackHandler(st.container())

                # 2. Execute the Agent
                # We pass the 'st_callback' here so the Agent can write its live thoughts to the screen
                response = st.session_state.agent_executor.invoke(
                    {"input": prompt_text},
                    {"callbacks": [st_callback]} # <--- The visualizer hook
                )

                # 3. Output Validation
                # Ensure the Agent actually returned a valid string response
                if "output" in response and len(response["output"]) > 0:
                    answer = response["output"]
                else:
                    # Fallback message if the Agent returns an empty or malformed result
                    answer = "I'm sorry, I couldn't find a relevant answer in the documents."

            except Exception as e:
                error_msg = str(e)
                
                # 1. Handle API Quota Limits (Most Common)
                if "429" in error_msg or "Quota exceeded" in error_msg:
                    answer = "🚨 **API Quota Exceeded**\n\nI cannot answer right now because the Google Gemini API limit has been reached. Please wait a minute before trying again."
                
                # 2. Handle Safety/Content Filters
                elif "finish_reason" in error_msg and "SAFETY" in error_msg:
                     answer = "🛡️ **Safety Restriction**\n\nI cannot answer this question because it triggered Google's safety filters. Please rephrase your question."

                # 3. Handle Invalid API Key
                elif "API key not valid" in error_msg:
                     answer = "🔑 **Invalid API Key**\n\nPlease check the Google API Key settings in the sidebar."

                # 4. Handle Web Search Tool Errors (DuckDuckGo Rate Limit)
                elif "ratelimit" in error_msg.lower() or "auth" in error_msg.lower():
                     answer = "🌐 **Search Tool Issue**\n\nThe web search tool is currently unavailable. I will try to answer based ONLY on the documents."

                # 5. Handle General/Unknown Errors
                else:
                    # Clean up the error message to be less scary if possible
                    answer = f"❌ **An error occurred:**\n\nI encountered an issue while processing your request. \n\n**Details:** `{error_msg}`"

            # 3. Display AI's answer and save it to history
            st.markdown(answer)
            st.session_state.message.append({"role": "ai", "content": answer})

with st.sidebar:
    st.divider()

    # Check if there is any conversation history to download
    # (Empty lists evaluate to False in Python)
    if st.session_state.message:
        # Prepare the data: Convert the message list to a formatted JSON string
        chat_str = json.dumps(st.session_state.message, indent=2)

        # Render the active Download Button and capture the click event
        is_downloaded = st.download_button(
            label="Download Chat History",
            data=chat_str,
            file_name="docuchat_history.json",
            mime="application/json",
            icon="📥",
            key="dl_btn_bottom", # SAYA TAMBAH KEY UNIK AGAR TIDAK ERROR DUPLICATE ID
            help="Save your conversation and research insights as a JSON file."
        )
        # Feedback: Show a toast notification if the user just clicked the button
        if is_downloaded:
            st.toast("✅ Chat history downloaded successfully!", icon="🎉")

    else:
        # Render a Disabled Button to indicate the feature exists but is unavailable
        # This provides better UX than hiding the button completely
        st.button(
            label="Download Chat History",
            icon="📥",
            disabled=True,
            key="dl_btn_disabled_bottom", # SAYA TAMBAH KEY UNIK AGAR TIDAK ERROR DUPLICATE ID
            help="Chat history is empty. Start a conversation first to enable export."
        )

    st.divider()

    with st.expander("📖 How To Use"):
        st.markdown("""
        1. **Setup:** Enter your valid **Google Gemini API Key**.
        2. **Language:** Select your preferred **Response Language** (English/Indonesian). The AI will translate answers for you automatically.
        3. **Creativity:** Adjust the slider. Low (0.1) for strict facts, High (0.7+) for fluid explanation.
        4. **Upload:** Click **'Browse files'** to select your PDF, CSV, MD, or TXT files.
        5. **Process:** Click **'Process & Embed Documents'**.
        6. **Chat:** Type your question. The AI answers based strictly on your documents.
        7. **Export:** Click **'Download Chat History'** to save the conversation as JSON.
        8. **Add More:** Need to add another file? Simply upload it and click **'Process'** again.
        """)

    with st.expander("❓ Frequently Asked Questions"):
        st.markdown("""
        **Q: My document is in English, can the AI answer in Indonesian?** A: **Yes!** Just select "Indonesian" in the sidebar. The AI acts as a translator and researcher simultaneously.

        **Q: What does the "Creativity Level" (Temperature) do?** A: It controls the AI's imagination. 
        - **Low (0.0 - 0.3):** Best for strict fact-checking and precise data extraction.
        - **High (0.7 +):** Better for creative summaries or brainstorming.

        **Q: Can I save my conversation?** A: **Yes!** Use the "Download Chat History" button to get a JSON file of your session.

        **Q: Can I add more documents in the middle of a conversation?** A: **Yes!** Upload new files and click Process. The AI updates its knowledge base instantly.

        **Q: What happens if I refresh the page?** A: All data (documents & chat) will be lost because this session is temporary.

        **Q: Why does the AI say "I don't know"?** A: It is restricted to answer **only** based on your uploaded files to prevent hallucinations.

        **Q: Is my API Key safe?** A: Yes. Your API Key is used only for this session to communicate with Google's servers and is not stored permanently.
        """)

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; font-size: 0.85rem; color: #888;">
            © 2026 <b>Silvio Christian, Joe</b><br>
            Powered by <b>Google Gemini</b> 🚀<br><br>
            <a href="https://www.linkedin.com/in/silvio-christian-joe/" target="_blank" style="text-decoration: none; margin-right: 10px;">🔗 LinkedIn</a>
            <a href="mailto:viochristian12@gmail.com" style="text-decoration: none;">📧 Email</a>
        </div>
        """, 
        unsafe_allow_html=True
    )