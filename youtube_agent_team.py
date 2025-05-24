from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team.team import Team
from src.tools.youtube_api import (
    resolve_channel_id,
    search_youtube_channel_videos,
    fetch_channel_info,
    fetch_videos,
    fetch_video_statistics,
    introspect_channel,
    fetch_video_details,
    fetch_comments,
    search_youtube_channels
)
from src.tools.video_analysis import video_to_text, analyze_video_content
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
        resolve_channel_id,
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

# Agent 7: Sentiment Analyzer
sentiment_analyzer = Agent(
    name="Sentiment Analyzer",
    role="Assigns sentiment scores to text or list of texts using the sentiment_score tool",
    model=OpenAIChat(id="gpt-4.1-mini"),
    tools=[sentiment_score],
    instructions=[
        "You are a sentiment analyzer. You receive a single text or a list of texts (probably a list of YouTube comments) and must return a sentiment score.",
        "Use the sentiment_score tool to compute the sentiment.",
        "At the end, return a JSON object with field sentiment_score."
    ],
    expected_output="Present your output as JSON with field sentiment_score.",
    show_tool_calls=True,
    markdown=True
)

# Agent 8: Video Statistics Specialist
video_statistics_specialist = Agent(
    name="Video Statistics Specialist",
    role="Analyzes engagement statistics for recent videos on a YouTube channel",
    model=OpenAIChat(id="gpt-4.1-mini"),
    tools=[resolve_channel_id, fetch_video_statistics],
    instructions=[
        "You are a video statistics specialist focused on analyzing engagement metrics.",
        "First resolve the channel identifier to get the official channel ID.",
        "Then fetch statistics for recent videos including views, likes, comments, and favorites.",
        "Present the statistics in a clear, tabular format.",
        "Focus on providing insights about video performance and engagement patterns."
    ],
    show_tool_calls=True,
    markdown=True
)

# Agent 9: Video Content Analyzer
video_content_analyzer = Agent(
    name="Video Content Analyzer",
    role="Analyzes video content for transcription and brand integration detection",
    model=OpenAIChat(id="gpt-4.1-mini"),
    tools=[resolve_channel_id, video_to_text, analyze_video_content],
    instructions=[
        "You are a video content analysis specialist focused on transcribing videos and detecting brand integrations.",
        "Use video_to_text to transcribe video content when requested.",
        "Use analyze_video_content to detect scenes, sponsors, and visual elements in videos.",
        "When analyzing video content:",
        "1. Process the raw tool output and present it in a clear, readable format",
        "2. For sponsor detection, clearly indicate when and where sponsors appear in the video",
        "3. Format timestamps in minutes:seconds format for better readability",
        "4. Group consecutive scenes with the same sponsor together",
        "5. Provide a clear summary of findings at the end",
        "6. If asked about specific timestamps or sponsor mentions, highlight those details prominently"
    ],
    expected_output="""Present your analysis in a clear markdown format with:
        1. A summary of findings
        2. Sponsor mentions with timestamps
        3. Any specific details requested by the user
        4. Clear formatting and organization of the information""",
    show_tool_calls=True,
    markdown=True
)

# Create the YouTube analysis team
youtube_team = Team(
    name="YouTube Analysis Team",
    mode="coordinate",  # Using coordinate mode to delegate tasks and synthesize responses
    model=OpenAIChat(id="gpt-4o"),
    members=[
        channel_resolver,
        channel_collector,
        video_collector,
        channel_searcher,
        risk_analyzer,
        python_coder,
        sentiment_analyzer,
        video_statistics_specialist,
        video_content_analyzer
    ],
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
        "6. Use Python Script Generator to execute Python scripts based on user queries. You can use this agent to do any calculations with numbers also.",
        "7. Use Sentiment Analyzer when you need to compute sentiment scores for text or comments.",
        "8. Use Video Statistics Specialist when you need to analyze engagement metrics for recent videos.",
        "9. Use Video Content Analyzer when you need to transcribe videos or detect brand integrations.",
        "Present all data in a clear, organized format using markdown.",
        "Ensure proper coordination between agents when tasks require multiple steps.",
        "Maintain context between different analysis steps."
    ],
    show_members_responses=True,
    enable_agentic_context=True  # Enable shared context between agents
)

# Example usage
if __name__ == "__main__":
    # Basic prompts for individual agents
    channel_resolver_prompt = "Resolve the channel ID for @BenAI92"
    channel_collector_prompt = "Get basic info and last 5 videos for @BenAI92"
    video_collector_prompt = "Get details and comments for video ID 'ngLyX54e1LU'"
    channel_searcher_prompt = "Find channels about AI sales automation"
    risk_analyzer_prompt = "Analyze risk for influencer Carryminati"
    python_coder_prompt = "Calculate average views for last 5 videos of @BenAI92"
    sentiment_analyzer_prompt = "Analyze sentiment of comments for video 'ngLyX54e1LU'"
    video_stats_prompt = "Get statistics for last 10 videos of @BenAI92"
    video_content_prompt = "Can you find what is the main sponsor of the video 'mqPnSt34Qks' - by looking at the video description. Then I want you to find the time where this brand is mentioned for the first time."
    # Complex prompts for team analysis
    team_prompt_1 = """
    Find a YouTube channel @LslieLawson, summarize the channel info. Take last 7 videos with the views count of this YouTube channel, 
    summarize the content and predict the 75% two-sided interval of the views for the next video. For every video take last 5 comments 
    and give a sentiment score to each video. Do a web search to see if there is a risk working with this influencer. Return a detailed 
    report where you'll write the summary of the channel, explain the content in detail and you can also save a graph where you'll plot 
    the historical views and horizontal lines for lower and upper bound of 75% two-sided predicition interval. You can also plot the 
    sentiment scores for the last 7 videos. Please include the risk analysis in this report. Nicely format the detailed report.
    """

    team_prompt_2 = """
    Fetch basic info (title and description) of this channel: @BenAI92. When you found the channel, then find 3 videos on the topic of 
    automation in sales on this channel. For this video you should find this info: published date, views count, likes count, comments count.
    """

    team_prompt_3 = """
    Analyze the statistics (views, likes, comments, published time) of the last 40 videos of the channel @CashJordan. Write a python code to analyze the views per day of the week and per time of the day. Plot the results in a graph.
    """

    # Individual agent tests (commented out)
    # channel_resolver.print_response(channel_resolver_prompt, stream=True)
    # channel_collector.print_response(channel_collector_prompt, stream=True)
    # video_collector.print_response(video_collector_prompt, stream=True)
    # channel_searcher.print_response(channel_searcher_prompt, stream=True)
    # risk_analyzer.print_response(risk_analyzer_prompt, stream=True)
    # python_coder.print_response(python_coder_prompt, stream=True)
    # sentiment_analyzer.print_response(sentiment_analyzer_prompt, stream=True)
    # video_statistics_specialist.print_response(video_stats_prompt, stream=True)
    # video_content_analyzer.print_response(video_content_prompt, stream=True)

    # Team analysis test
    youtube_team.print_response(team_prompt_1, stream=True)
    # youtube_team.print_response(team_prompt_2, stream=True)
    youtube_team.print_response(team_prompt_3, stream=True)
