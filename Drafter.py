from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

load_dotenv()

document_content = ""

class MyState(TypedDict):
    message: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """Update the document with the provided content."""
    global document_content
    document_content = content
    return f"Documetn has been updated successfuly! The curret content is:\n{document_content}"

@tool
def save(filename: str) -> str:
    """"Save the curret document to a text file and finish the process.
    Args:
        filename: name for the text file.
    """
    global document_content
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\nDocument has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'"
    except Exception as e:
        return f"Error saving documetn: {str(e)}"
    

tools = [update, save]

model = ChatGroq(model="openai/gpt-oss-120b").bind_tools(tools)

def our_agent(state: MyState) -> MyState:
    system_promt = SystemMessage(content=f"""
    You are Drafter, a helful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete update content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the curret document state after modifications.
    
    The curret document is:{document_content}
    """)
    if not state['message']:
        user_input = "I'am ready to help you to update a document. What would you like to creatre?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document?")
        print(f"\nUSER: {user_input}")
        user_message = HumanMessage(content=user_input)
    
    all_messages = [system_promt] + list(state["message"]) + [user_message]
    response = model.invoke(all_messages)

    print(f"\nAI: {response.contetn}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f" USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
