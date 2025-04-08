from typing import Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")
gemini_api_key=os.getenv("GEMINI_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

class State(TypedDict):
    messages: Annotated[list, add_messages]

def create_langgraph_agent(model: str):
    graph_builder = StateGraph(State)

    memory=MemorySaver()

    # @tool
    # def human_assistance(query:str)->str:
    #     """Request assistance from a human."""
    #     human_response=interrupt({"query":query})
    #     return human_response["data"]

    tool = TavilySearchResults(max_results=5)

    tools = [tool]

    if model=="gemini":
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=gemini_api_key
        )
    elif model=="groq":
        llm=ChatGroq(
            model="qwen-qwq-32b",
            api_key=groq_api_key
        )

    elif model=="openai":
        llm=ChatOpenAI(
            model="gpt-4o"
        )

    else:
        raise ValueError("Please provide one among the following: 'gemini', 'groq', 'openai'")

    llm_with_tools = llm.bind_tools(tools)


    def chatbot(state: State):

        # message=llm_with_tools.invoke(state["messages"])
        # assert len(message.tool_calls)<=1
        # return {"messages":[message]}
        message = llm_with_tools.invoke(state["messages"])
        # Because we will be interrupting during tool execution,
        # we disable parallel tool calling to avoid repeating any
        # tool invocations when we resume.
        assert len(message.tool_calls) <= 1
        return {"messages": [message]}



    graph_builder.add_node("chatbot", chatbot)


    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    memory=MemorySaver()

    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    graph = graph_builder.compile(checkpointer=memory)

    return graph

def get_response(user_input:str,config:dict,agent):

    model=ChatGroq(
        model="llama-3.3-70b-specdec",
        api_key=groq_api_key
    )

    outputs=[]

    parser=StrOutputParser()

    events = agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        user_input
                    ),
                },
            ],
        },
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            outputs.extend(event["messages"])

    prompt=PromptTemplate(
        template="Convert the following into markdown -> {result}",
        input_variables=["result"],
    )

    markdown_chain=prompt|model|parser
    result=markdown_chain.invoke({"result":' '.join(outputs[-1].content)})

    return result