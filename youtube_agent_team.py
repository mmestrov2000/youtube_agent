from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team.team import Team
from src.tools.youtube_api import (
    resolve_channel_id,
    search_channel_videos,
    fetch_channel_info,
    fetch_videos,
    introspect_channel,
    fetch_video_details,
    fetch_comments,
    search_youtube_channels
)

# Create individual specialized agents

# Agent 1: Channel ID Resolver
channel_resolver = Agent(
    name="Channel ID Resolver",
    role="Resolves YouTube channel identifiers to their official channel IDs",
    model=OpenAIChat(id="gpt-4o"),
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
    model=OpenAIChat(id="gpt-4o"),
    tools=[
        search_channel_videos,
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
    model=OpenAIChat(id="gpt-4o"),
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
    model=OpenAIChat(id="gpt-4o"),
    tools=[search_youtube_channels],
    instructions=[
        "You are responsible for searching and discovering YouTube channels.",
        "Use the search_youtube_channels tool to find relevant channels based on queries.",
        "Present search results in a clear, organized manner.",
        "Focus on finding the most relevant channels for given topics."
    ],
    markdown=True
)

# Create the YouTube analysis team
youtube_team = Team(
    name="YouTube Analysis Team",
    mode="coordinate",  # Using coordinate mode to delegate tasks and synthesize responses
    model=OpenAIChat(id="gpt-4o"),
    members=[channel_resolver, channel_collector, video_collector, channel_searcher],
    show_tool_calls=True,
    markdown=True,
    description="A team of specialized agents for comprehensive YouTube data analysis",
    instructions=[
        "You are a team of specialized YouTube data collection agents.",
        "Coordinate between different agents to provide comprehensive YouTube data analysis.",
        "Use the appropriate agent for each specific task:",
        "1. Use Channel ID Resolver when you need to resolve channel identifiers",
        "2. Use Channel Data Collector for comprehensive channel analysis",
        "3. Use Video Data Collector for detailed video analysis",
        "4. Use Channel Search Specialist for discovering relevant channels",
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
    
    youtube_team.print_response(request, stream=True) 