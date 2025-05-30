import streamlit as st
from youtube_agent_team import youtube_team

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    st.title("YouTube Analysis Assistant")
    st.markdown("""
    ### Your AI-powered YouTube Analysis Assistant
    
    Ask me anything about YouTube channels, videos, or content analysis. I can help you with:
    - Channel analysis and statistics
    - Video performance metrics
    - Content analysis and sentiment
    - Brand safety and risk assessment
    - And much more!
    """)

    initialize_session_state()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know about YouTube?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    response = youtube_team.run(prompt)
                    st.markdown(response.content)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response.content})

                    # Show tool calls in expander if available
                    if hasattr(response, 'tool_calls'):
                        with st.expander("Show API calls made"):
                            for call in response.tool_calls:
                                st.json(call)

                    # Show agent interactions in expander if available
                    if hasattr(response, 'members_responses'):
                        with st.expander("Show agent interactions"):
                            for agent, resp in response.members_responses.items():
                                st.markdown(f"*{agent}*")
                                st.markdown(resp.content)
                                st.divider()
                except Exception as e:
                    error_message = f"Analysis failed: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()