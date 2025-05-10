from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team.team import Team
from src.tools.youtube_api import (
    resolve_channel_id,
    search_youtube_channel_videos,
    fetch_channel_info,
    fetch_videos,
    introspect_channel,
    fetch_video_details,
    fetch_comments,
    search_youtube_channels
)
from agno.tools.reasoning import ReasoningTools
from agno.tools.python import PythonTools
from pathlib import Path
from agno.tools.tavily import TavilyTools

from src.tools.risk import sentiment_score

from src.tools.analysis import predict_next_video_views


from dotenv import load_dotenv
load_dotenv()


# Create individual specialized agents

# Agent 1: Channel ID Resolver
channel_resolver = Agent(
    name="Channel ID Resolver",
    role="Resolves YouTube channel identifiers to their official channel IDs",
    model=OpenAIChat(id="gpt-4.1-mini"),
    tools=[resolve_channel_id],
    instructions=[
        "You are responsible for resolving YouTube channel identifiers to their official channel IDs.",
        "Handle various input formats including channel handles, custom URLs, and direct channel IDs.",
        "Return the resolved channel ID in a clear format."
    ],
    markdown=True
)

# Agent 2: Channel Data Collector
channel_collector = Agent(
    name="Channel Data Collector",
    role="Collects comprehensive data about YouTube channels",
    model=OpenAIChat(id="gpt-4.1-mini"),
    tools=[
        search_youtube_channel_videos,
        fetch_channel_info,
        fetch_videos,
        introspect_channel
    ],
    instructions=[
        "You are responsible for collecting comprehensive data about YouTube channels.",
        "You can search for videos within channels, fetch channel information, and get recent videos.",
        "Use the introspect_channel tool for a complete channel analysis.",
        "Present the data in a well-organized, readable format."
    ],
    markdown=True
)

# Agent 3: Video Data Collector
video_collector = Agent(
    name="Video Data Collector",
    role="Collects detailed information about specific YouTube videos",
    model=OpenAIChat(id="gpt-4.1-mini"),
    tools=[
        fetch_video_details,
        fetch_comments
    ],
    instructions=[
        "You are responsible for collecting detailed information about specific YouTube videos.",
        "You can fetch video metadata and comments.",
        "Present the data in a structured, easy-to-read format.",
        "Focus on providing comprehensive video analysis."
    ],
    markdown=True
)

# Agent 4: Channel Search Specialist
channel_searcher = Agent(
    name="Channel Search Specialist",
    role="Searches and discovers YouTube channels",
    model=OpenAIChat(id="gpt-4.1-mini"),
    tools=[search_youtube_channels],
    instructions=[
        "You are responsible for searching and discovering YouTube channels.",
        "Use the search_youtube_channels tool to find relevant channels based on queries.",
        "Present search results in a clear, organized manner.",
        "Focus on finding the most relevant channels for given topics."
    ],
    markdown=True
)

# Agent 5: Risk Analysis Agent
risk_analyzer = Agent(
    name="Risk Analysis Agent",
    role="Analyzes potential risks associated with YouTube channels and influencers",
    model=OpenAIChat(id="gpt-4.1-mini"),
    tools=[TavilyTools(), sentiment_score],
    instructions=[
        "You are a risk analyst for a company. You are given the influencer and you need to analyze is it a risk to the company.",
        "Use Tavily to search web for at least 3 different topics with keywords like scandal, controversy, cancelled, etc. then summarize together all results. You should also determine the probability that these controversies are related to this specific influencer and not to another one.",
        "Keep in mind that most influencers should be save, and keywords like these will always lead to controversies - and that's why you need to make sure that the controversies are related to this specific influencer and not to another one.",
        "Use sentiment_score tool to analyze the sentiment of the comments and the videos.",
        "At the end, return a JSON object with risk_score, confidence_score and brief_analysis.",
    ],
    expected_output="""Present your analysis in a clear markdown format with the following sections:",
        "1. Risk Score (0-1): A numerical assessment of the overall risk level",
        "2. Confidence Score (0-1): How confident you are in your risk assessment",
        "3. Brief Analysis: A concise summary of your findings and reasoning",
        "4. Verification Notes: How you verified the relevance of findings to this specific influencer""",
    show_tool_calls=True,
    markdown=True
)

# Agent 6: Python Script Generator
python_coder = Agent(
    name="Python Script Generator",
    role="Generates and executes Python scripts based on user queries.",
    model=OpenAIChat(id="gpt-4.1-mini"),
    tools=[PythonTools(base_dir=Path("tmp/python")), predict_next_video_views],
    instructions=[
        "Use the PythonTools to generate and execute Python scripts.",
        "Respond with well-commented Python code that solves the user's request.",
        "Only write code that is safe and relevant to the task.",
        "Avoid unnecessary output; show only the results or saved file paths.",
        "If a graph is generated, save it using matplotlib instead of displaying it.",
        "For predicting the views of the next video use the tool predict_next_video_views."
    ],
    show_tool_calls=True,
    markdown=True
)

# Create the YouTube analysis team
youtube_team = Team(
    name="YouTube Analysis Team",
    mode="coordinate",  # Using coordinate mode to delegate tasks and synthesize responses
    model=OpenAIChat(id="gpt-4.1-mini"),
    members=[channel_resolver, channel_collector, video_collector, channel_searcher, risk_analyzer, python_coder],
    show_tool_calls=True,
    markdown=True,
    description="A team of specialized agents for comprehensive YouTube data analysis",
    instructions=[
        "You are a team of specialized YouTube data collection agents.",
        "Coordinate between different agents to provide comprehensive YouTube data analysis.",
        "Use the appropriate agent for each specific task:",
        "1. Use Channel ID Resolver when you need to find the channel ID of the YouTube channel",
        "2. Use Channel Data Collector for comprehensive channel analysis",
        "3. Use Video Data Collector for detailed video analysis",
        "4. Use Channel Search Specialist for discovering relevant channels",
        "5. Use Risk Analysis Agent for evaluating potential risks and brand safety concerns",
        "6. Use Python Script Generator to execute Python scripts based on user queries.",
        "Present all data in a clear, organized format using markdown.",
        "Ensure proper coordination between agents when tasks require multiple steps.",
        "Maintain context between different analysis steps."
    ],
    show_members_responses=True,
    enable_agentic_context=True  # Enable shared context between agents
)
# Example usage
if __name__ == "__main__":
    # Task: Find the number of views for the 3 videos on the topic of "AI sales agent" on the channel @BenAI92
    request = """
    Please help me find the number of views for the 3 most recent videos about "AI sales agent" on the channel @BenAI92.
    """

    request_2 = """
    Fetch basic info (title and description) of this channel: @BenAI92. When you found the channel, then find 3 videos on the topic of automation in sales on this channel. For this video you should find this info: published date, views count, likes count, comments count.
    """

    request_3 = """
    Fetch basic info (title and description) of this channel: @BenAI92. When you found the channel, then find 3 latest videos on this channel. For every video find the latest 5 comments.
    """

    request_4 = """
    Fetch the title and description of the last 10 videos on the channel @BenAI92. 
    See if he had any sponsored video - and list the brands he has promoted.
    """

    request_5 = "Is this influencer risky: Carryminati. Do a web search to find controversies, and find his YouTube channel to calculate the sentiment score on the comments in his last 3 videos."
    
    request_6 = "Take last 10 video views of the YouTube channel @JamesCharles and predict the 90% two-sided interval of the views for the next video. Plot a graph of historical video views and horizontal lines for lower and upper part. Make sure that all numbers are in the same format and scaled correctly."

    youtube_team.print_response(request_6, stream=True)
    #python_coder.print_response("Write a Python script that plots a bar chart of the top 10 Indian states by population from the 2011 Census.")
