from datetime import datetime
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from src.tools.youtube_api import (
    download_video,
    resolve_channel_id,
    fetch_video_details,
    search_youtube_channel_videos,
    fetch_channel_info,
    fetch_videos,
    fetch_comments,
    introspect_channel,
    search_youtube_channels,
    search_and_introspect_channel
)
from agno.team import Team

today = datetime.now().strftime("%Y-%m-%d")

# Test agent for downloading videos
download_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    name="Video Downloader",
    tools=[download_video],
    description=dedent("""\
        You are a YouTube video downloader. Your job is to download videos
        with specified quality settings and save them to the correct location.\
    """),
    instructions=[
        "Download the video with the specified ID.",
        "Use the requested quality settings.",
        "Save to the correct output path.",
        "Report the download status."],
    expected_output=dedent("""\
    # Download Report
    - Video ID: {video_id}
    - Quality: {quality}
    - Status: {status}
    - Output: {output_path}\
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True
)

# Test agent for resolving channel IDs
channel_id_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    name="Channel ID Resolver",
    tools=[resolve_channel_id],
    description=dedent("""\
        You convert YouTube channel identifiers (handles, URLs, or IDs) into
        official YouTube channel IDs.\
    """),
    instructions=[
        "Convert the given channel identifier to an official ID.",
        "Handle different input formats (handle, URL, or ID).",
        "Return the official channel ID."],
    expected_output=dedent("""\
    # Channel ID Resolution
    - Input: {identifier}
    - Official ID: {channel_id}
    - Status: {status}\
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True
)

# Test agent for fetching video details
video_details_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    name="Video Details Fetcher",
    tools=[fetch_video_details],
    description=dedent("""\
        You fetch and organize detailed information about YouTube videos,
        including metadata and statistics.\
    """),
    instructions=[
        "Get complete video information.",
        "Organize the data clearly.",
        "Include key statistics."],
    expected_output=dedent("""\
    # Video Details
    - Title: {title}
    - ID: {video_id}
    - Stats: {statistics}
    - Description: {description}\
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True
)

# Test agent for searching channel videos
channel_video_search_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    name="Channel Video Searcher",
    tools=[search_youtube_channel_videos],
    description=dedent("""\
        You search for videos within a specific YouTube channel based on
        given search terms.
                       
        Example prompt: Return the title and description of 3 videos in the channel with ID UCktMIWomeuj4pwvBb-OBwMw for the search term 'neural networks'.
        Flow: You would call the tool to fetch the videos. Then you'll analyze the output and fetch the title and description of each video.
        Output: 
        # Channel Video Search Results
        - Channel ID: UCktMIWomeuj4pwvBb-OBwMw
        - Search Term: neural networks
        - Videos:
            1. Title: Neural Networks in 100 seconds
               Description: Explore neural networks - the mathematical models behind modern AI...
            2. Title: Understanding Neural Networks
               Description: A comprehensive guide to neural networks...
            3. Title: Neural Networks Explained
               Description: Learn how neural networks work...\
    """),
    instructions=[
        "Fetch the videos from the channel.",
        "Analyze what the user is asking for and return the details."],
    expected_output=dedent("""\
    # Channel Video Search Results
    - Channel ID: {channel_id}
    - Search Term: {search_term}
    - Videos:
        {numbered_list_of_videos_with_titles_and_descriptions}\
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True
)

# Test agent for fetching channel info
channel_info_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    name="Channel Info Fetcher",
    tools=[fetch_channel_info],
    description=dedent("""\
        You fetch and organize information about YouTube channels,
        including statistics and metadata.\
    """),
    instructions=[
        "Get channel information.",
        "Include key statistics.",
        "Present the data clearly."],
    expected_output=dedent("""\
    # Channel Information
    - Name: {channel_name}
    - ID: {channel_id}
    - Stats: {statistics}\
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True
)

# Test agent for fetching channel videos
channel_videos_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    name="Channel Videos Fetcher",
    tools=[fetch_videos],
    description=dedent("""\
        You fetch recent videos from a YouTube channel and organize
        them chronologically.\
    """),
    instructions=[
        "Get recent videos from the channel.",
        "Organize by date.",
        "Include video details."],
    expected_output=dedent("""\
    # Recent Channel Videos
    - Channel: {channel_name}
    - Videos: {video_list}\
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True
)

# Test agent for fetching video comments
comments_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    name="Video Comments Fetcher",
    tools=[fetch_comments],
    description=dedent("""\
        You fetch and organize comments from YouTube videos,
        including engagement metrics.\
    """),
    instructions=[
        "Get video comments.",
        "Include engagement metrics.",
        "Present comments clearly."],
    expected_output=dedent("""\
    # Video Comments
    - Video: {video_title}
    - Comments: {comment_list}\
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True
)

# Test agent for channel introspection
introspect_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    name="Channel Introspector",
    tools=[introspect_channel],
    description=dedent("""\
        You analyze YouTube channels to provide insights about
        their content and performance.\
    """),
    instructions=[
        "Analyze the channel content.",
        "Evaluate performance metrics.",
        "Provide key insights."],
    expected_output=dedent("""\
    # Channel Analysis
    - Channel: {channel_name}
    - Content: {content_analysis}
    - Performance: {metrics}\
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True
)

# Test agent for searching channels
channel_search_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    name="Channel Searcher",
    tools=[search_youtube_channels],
    description=dedent("""\
        You search for YouTube channels based on given criteria
        and evaluate their relevance.\
    """),
    instructions=[
        "Search for relevant channels.",
        "Evaluate channel quality.",
        "Present the best matches."],
    expected_output=dedent("""\
    # Channel Search Results
    - Query: {search_query}
    - Results: {channel_list}\
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True
)

# Test agent for search introspection
search_introspect_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    name="Search Introspector",
    tools=[search_and_introspect_channel],
    description=dedent("""\
        You search for YouTube channels and provide detailed
        analysis of the best matches.\
    """),
    instructions=[
        "Search for relevant channels.",
        "Analyze the best matches.",
        "Provide detailed insights."],
    expected_output=dedent("""\
    # Search Analysis
    - Query: {search_query}
    - Best Match: {channel_analysis}\
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True
)

# Example usage
if __name__ == "__main__":
    # Example prompts for individual agents
    agent_prompts = [
        (download_agent, "Download the video with ID 'dQw4w9WgXcQ' in 720p quality"),
        (channel_id_agent, "Resolve the channel identifier '@InfiniteCodes_' to its official channel ID"),
        (video_details_agent, "Get details about the video with ID 'dQw4w9WgXcQ'"),
        (channel_video_search_agent, "Find title and description for 3 videos about 'neural networks' in the channel with ID UCktMIWomeuj4pwvBb-OBwMw"),
        (channel_info_agent, "Get information about the channel with ID UCHnyfMqiRRG1u-2MsSQLbXA"),
        (channel_videos_agent, "Get the 3 most recent videos from the channel with ID UCHnyfMqiRRG1u-2MsSQLbXA"),
        (comments_agent, "Get the top 5 comments from the video with ID 'dQw4w9WgXcQ'"),
        (introspect_agent, "Analyze the channel with ID UCHnyfMqiRRG1u-2MsSQLbXA and get its 5 most recent videos"),
        (channel_search_agent, "Search for 3 channels about 'machine learning'"),
        (search_introspect_agent, "Search for channels about 'physics education' and analyze the top match"),
    ]

    # Run each agent with its example prompt
    for agent, prompt in agent_prompts:
        print(f"\nTesting {agent.name} with prompt: {prompt}")
        agent.print_response(prompt)
