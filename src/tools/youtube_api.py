# YouTube API Wrapper
# This module will handle YouTube API interactions 
from typing import Annotated, Callable, Any
from agno.agent import Agent
from agno.tools import tool
from typing import Dict, List, Optional
import os
import re
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import yt_dlp

# Load environment variables from .env file
load_dotenv()


def logger_hook(function_name: str, function_call: Callable, arguments: Dict[str, Any]):
    """Pre-hook function that runs before the tool execution"""
    print(f"About to call {function_name} with arguments: {arguments}")
    result = function_call(**arguments)
    print(f"Function call completed with result: {result}")
    return result

class YouTubeAPI:
    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError("YouTube API key not found in environment variables")
        
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }

# Create a singleton instance
youtube_api = YouTubeAPI()


@tool(
    name="download_video",                                   # Custom name for the tool
    description="Download a YouTube video using yt-dlp.",      # Custom description
    show_result=True,                                         # Show result after function call
    stop_after_tool_call=True,                                # Return the result immediately after the tool call
    tool_hooks=[logger_hook],                                 # Hook to run before and after execution
    cache_results=False,                                      # No caching for downloading videos
)
def download_video(
    video_id: Annotated[str, "The unique identifier of the YouTube video. It is the part of the YouTube URL after 'v='. Example: For the URL 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', the 'video_id' is 'dQw4w9WgXcQ.'"],
    output_path: Annotated[str, "The directory where the video will be saved. Defaults to 'downloads'. If not provided, a directory named 'downloads' will be created. Example: 'videos' or 'C:/Users/YourName/Videos'"] = "downloads",
    quality: Annotated[str, "The desired video quality. Options include 'best', 'worst', or specific resolutions like '720p', '1080p'. Defaults to 'best'. Example: '720p', 'best', '1080p'"] = "best"
) -> str:
    """
    Download a YouTube video using yt-dlp.
    
    Args:
        video_id (str): The YouTube video ID
        output_path (str): Directory to save the video (default: "downloads")
        quality (str): Video quality to download (default: "best")
            Options: "best", "worst", "720p", "1080p", etc.
            
    Returns:
        str: Path to the downloaded video file
        
    Raises:
        Exception: If download fails
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': f'bestvideo[height<={quality[:-1]}]+bestaudio/best[height<={quality[:-1]}]' if quality.endswith('p') else quality,
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'progress': True
        }
        
        # Create yt-dlp object
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            info = ydl.extract_info(video_url, download=True)
            
            # Return the path to the downloaded file
            return os.path.join(output_path, f"{info['title']}.{info['ext']}")
            
    except Exception as e:
        raise Exception(f"Error downloading video: {str(e)}")

# Create the agent and add the tool
agent = Agent(tools=[download_video])
agent.print_response("Download the video with ID dQw4w9WgXcQ to my videos folder at 720p quality")


@tool(
    name="search_youtube_channel_videos",
    description=(
        "Search for videos within a specific YouTube channel that match a given term. "
        "Returns a list of relevant video details including title, description, view count, and more."
    ),
    show_result=True,
    stop_after_tool_call=True,
    tool_hooks=[logger_hook],
    cache_results=True,
    cache_dir="/tmp/agno_cache",
    cache_ttl=3600
)
def search_channel_videos(
    channel_id: Annotated[str, "The unique identifier of the YouTube channel. This can be obtained from the channel's URL. Example: For the URL 'https://www.youtube.com/channel/UCXgGY0w3hN4+Vq3po9q7mn' the 'channel_id' is 'UCXgGY0w3hN4+Vq3po9q7mn'."],
    search_term: Annotated[str, "The term to search for in video titles and descriptions. This is the keyword or phrase to look for within the channel's videos. Example: 'python tutorial' or 'travel vlog'."],
    max_results: Annotated[int, "The maximum number of search results to return. This limits the number of videos that will be retrieved from the search. Default is 10. Example: 5 or 20." ] = 10
) -> List[Dict]:
    """
    Search for videos within a specific channel that match the search term.
    
    Args:
        channel_id (str): The YouTube channel ID
        search_term (str): The term to search for in video titles and descriptions
        max_results (int): Maximum number of videos to return (default: 10)
        
    Returns:
        List[Dict]: List of video information including:
            - id: Video ID
            - title: Video title
            - description: Video description
            - publishedAt: Publication date
            - viewCount: Number of views
            - likeCount: Number of likes
            - commentCount: Number of comments
            - duration: Video duration
            - thumbnails: Video thumbnails
    """
    try:
        # Search for videos in the channel
        request = youtube_api.youtube.search().list(
            part="snippet",
            channelId=channel_id,
            q=search_term,
            type="video",
            maxResults=max_results,
            order="relevance"
        )
        response = request.execute()
        
        if not response['items']:
            return []
        
        # Get detailed information for each video
        videos = []
        for item in response['items']:
            video_id = item['id']['videoId']
            video_details = fetch_video_details(video_id)
            videos.append(video_details)
        
        return videos
        
    except HttpError as e:
        raise Exception(f"Error searching channel videos: {str(e)}")



@tool(
    name="resolve_channel_id",
    description=(
        "Resolves a YouTube channel identifier (handle, custom URL, or channel ID) "
        "to the official YouTube channel ID. Useful for ensuring downstream APIs get the correct ID."
    ),
    show_result=True,
    stop_after_tool_call=False,
    tool_hooks=[logger_hook],
    cache_results=True,
    cache_dir="/tmp/agno_cache",
    cache_ttl=3600  # Cache result for 1 hour
)
def resolve_channel_id(
    channel_identifier: Annotated[str, """
        The identifier of the YouTube channel, which can be one of the following:
        - Channel handle (e.g., "@channelname")
        - Custom URL (e.g., "youtube.com/c/channelname")
        - Channel ID (e.g., "UC...")
        The function will resolve this identifier to the YouTube channel ID.
        Example: For "@channelname", "youtube.com/c/channelname", or "UCxxxxxxx", it will return the resolved channel ID.
    """]
) -> str:
    """
    Resolve a YouTube channel handle, custom URL, or channel ID to a channel ID.
    
    Args:
        channel_identifier (str): Can be:
            - Channel handle (e.g., "@channelname")
            - Custom URL (e.g., "youtube.com/c/channelname")
            - Channel ID (e.g., "UC...")
            
    Returns:
        str: The resolved channel ID
        
    Raises:
        ValueError: If the channel cannot be found
    """
    try:
        # If it's already a channel ID (starts with UC), return it
        if re.match(r'^UC[a-zA-Z0-9_-]{22}$', channel_identifier):
            return channel_identifier
            
        # If it's a handle (starts with @), remove the @
        if channel_identifier.startswith('@'):
            channel_identifier = channel_identifier[1:]
            
        # If it's a URL, extract the handle
        if 'youtube.com' in channel_identifier:
            # Handle different URL formats
            if '/c/' in channel_identifier:
                channel_identifier = channel_identifier.split('/c/')[-1].split('/')[0]
            elif '/channel/' in channel_identifier:
                channel_identifier = channel_identifier.split('/channel/')[-1].split('/')[0]
            elif '/user/' in channel_identifier:
                channel_identifier = channel_identifier.split('/user/')[-1].split('/')[0]
                
        # Search for the channel
        request = youtube_api.youtube.search().list(
            part="snippet",
            q=channel_identifier,
            type="channel",
            maxResults=1
        )
        response = request.execute()
        
        if not response['items']:
            raise ValueError(f"Channel not found: {channel_identifier}")
            
        return response['items'][0]['id']['channelId']
        
    except HttpError as e:
        raise Exception(f"Error resolving channel ID: {str(e)}")

@tool(
    name="fetch_channel_info",
    description="Fetches basic metadata about a YouTube channel such as title, subscriber count, view count, and thumbnails.",
    show_result=True,
    stop_after_tool_call=False,
    tool_hooks=[logger_hook],
    cache_results=True,
    cache_dir="/tmp/agno_cache",
    cache_ttl=3600  # Cache for 1 hour
)
def fetch_channel_info(
    channel_id: Annotated[str, """
        The unique identifier of the YouTube channel. 
        This is the part of the YouTube URL after '/channel/'. For example:
        - For the URL 'https://www.youtube.com/channel/UCxxxxxxx', the `channel_id` would be 'UCxxxxxxx'.
        - This can also be the channel ID directly (e.g., 'UC1234567890').
    """]
) -> Dict:
    """
    Fetch basic channel information including subscriber count, view count, and video count.
    
    Args:
        channel_id (str): The YouTube channel ID
        
    Returns:
        Dict: Channel information including:
            - id: Channel ID
            - title: Channel title
            - description: Channel description
            - subscriberCount: Number of subscribers
            - viewCount: Total view count
            - videoCount: Number of videos
            - thumbnails: Channel thumbnails
    """
    try:
        request = youtube_api.youtube.channels().list(
            part="snippet,statistics",
            id=channel_id
        )
        response = request.execute()
        
        if not response['items']:
            raise ValueError(f"Channel not found: {channel_id}")
        
        channel = response['items'][0]
        return {
            "id": channel['id'],
            "title": channel['snippet']['title'],
            "description": channel['snippet']['description'],
            "subscriberCount": int(channel['statistics']['subscriberCount']),
            "viewCount": int(channel['statistics']['viewCount']),
            "videoCount": int(channel['statistics']['videoCount']),
            "thumbnails": channel['snippet']['thumbnails']
        }
    except HttpError as e:
        raise Exception(f"Error fetching channel info: {str(e)}")

def fetch_videos(
    channel_id: Annotated[str, """
        The unique identifier of the YouTube channel. 
        This is the part of the YouTube URL after '/channel/'. For example:
        - For the URL 'https://www.youtube.com/channel/UCxxxxxxx', the `channel_id` would be 'UCxxxxxxx'.
        - This can also be the channel ID directly (e.g., 'UC1234567890').
    """],
    max_results: Annotated[int, """
        The maximum number of videos to fetch from the channel. 
        This value controls how many recent videos will be returned.
        Default is 10, but can be set to any integer value.
    """ ] = 10
) -> List[Dict]:
    """
    Fetch recent videos from a channel.
    
    Args:
        channel_id (str): The YouTube channel ID
        max_results (int): Maximum number of videos to fetch (default: 10)
        
    Returns:
        List[Dict]: List of video information including:
            - id: Video ID
            - title: Video title
            - description: Video description
            - publishedAt: Publication date
            - viewCount: Number of views
            - likeCount: Number of likes
            - commentCount: Number of comments
            - duration: Video duration
            - thumbnails: Video thumbnails
    """
    try:
        # First get the uploads playlist ID
        request = youtube_api.youtube.channels().list(
            part="contentDetails",
            id=channel_id
        )
        response = request.execute()
        
        if not response['items']:
            raise ValueError(f"Channel not found: {channel_id}")
        
        uploads_playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # Then get the videos from the uploads playlist
        request = youtube_api.youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=max_results
        )
        response = request.execute()
        
        videos = []
        for item in response['items']:
            video_id = item['contentDetails']['videoId']
            video_details = fetch_video_details(video_id)
            videos.append(video_details)
        
        return videos
    except HttpError as e:
        raise Exception(f"Error fetching videos: {str(e)}")



@tool(
    name="fetch_video_details",
    description="Fetches detailed metadata about a YouTube video, including view count, likes, duration, etc.",
    show_result=True,
    cache_results=True,
    cache_ttl=3600,
    cache_dir="/tmp/agno_cache"
)
def fetch_video_details(video_id: str) -> Dict:
    """
    Fetch detailed information about a specific video.
    
    Args:
        video_id (str): The YouTube video ID
        
    Returns:
        Dict: Video information including:
            - id: Video ID
            - title: Video title
            - description: Video description
            - publishedAt: Publication date
            - viewCount: Number of views
            - likeCount: Number of likes
            - commentCount: Number of comments
            - duration: Video duration
            - thumbnails: Video thumbnails
    """
    try:
        request = youtube_api.youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        )
        response = request.execute()
        
        if not response['items']:
            raise ValueError(f"Video not found: {video_id}")
        
        video = response['items'][0]
        return {
            "id": video['id'],
            "title": video['snippet']['title'],
            "description": video['snippet']['description'],
            "publishedAt": video['snippet']['publishedAt'],
            "viewCount": int(video['statistics']['viewCount']),
            "likeCount": int(video['statistics'].get('likeCount', 0)),
            "commentCount": int(video['statistics'].get('commentCount', 0)),
            "duration": video['contentDetails']['duration'],
            "thumbnails": video['snippet']['thumbnails']
        }
    except HttpError as e:
        raise Exception(f"Error fetching video details: {str(e)}")


@tool(
    name="fetch_comments",
    description="Fetches top-level comments from a YouTube video using its video ID.",
    show_result=True,
    cache_results=True,
    cache_ttl=3600,
    cache_dir="/tmp/agno_cache"
)
def fetch_comments(
    video_id: Annotated[str, """
        The unique identifier of the YouTube video.
        This is the part of the YouTube URL after 'v='. For example:
        - For the URL 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', the `video_id` would be 'dQw4w9WgXcQ'.
        - This ID is used to fetch the comments for the specific video.
    """],
    max_results: int = 100
) -> List[Dict]:
    """
    Fetch comments for a video.
    
    Args:
        video_id (str): The YouTube video ID
        max_results (int): Maximum number of comments to fetch (default: 100)
        
    Returns:
        List[Dict]: List of comment information including:
            - id: Comment ID
            - author: Author name
            - text: Comment text
            - likeCount: Number of likes
            - publishedAt: Publication date
    """
    try:
        request = youtube_api.youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            order="relevance"
        )
        response = request.execute()
        
        comments = []
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                "id": comment['id'],
                "author": comment['authorDisplayName'],
                "text": comment['textDisplay'],
                "likeCount": comment['likeCount'],
                "publishedAt": comment['publishedAt']
            })
        
        return comments
    except HttpError as e:
        raise Exception(f"Error fetching comments: {str(e)}")

@tool(
    name="introspect_channel",
    description="Fetch comprehensive channel data including basic info and recent videos using either a channel handle or full URL.",
    show_result=True,
    cache_results=True,
    cache_ttl=3600
)
def introspect_channel(
    identifier: Annotated[str, """
        A YouTube channel identifier, which can be one of the following:
        - A full channel URL (e.g., https://www.youtube.com/channel/UCxxxxxx or https://www.youtube.com/@handle)
        - A channel handle (e.g., @veritasium)
        - A direct channel ID (e.g., UCxxxxxxx)
    """],
    max_videos: Annotated[int, """
        The maximum number of recent videos to fetch from the channel.
    """] = 10
) -> Dict:
    """
    Resolve the identifier to a channel ID, fetch channel info and recent videos.
    """
    try:
        # Step 1: Resolve to Channel ID
        channel_id = resolve_channel_id(identifier)

        # Step 2: Fetch channel info
        channel_info = fetch_channel_info(channel_id)

        # Step 3: Fetch videos
        recent_videos = fetch_videos(channel_id, max_videos)

        return {
            "channel_info": channel_info,
            "recent_videos": recent_videos
        }

    except Exception as e:
        return {"error": str(e)}



@tool(
    name="search_youtube_channels",
    description="Search for YouTube channels using a keyword or topic.",
    show_result=True,
    cache_results=True,
    cache_ttl=1800
)
def search_youtube_channels(
        query: Annotated[str, """
        The search query to find relevant YouTube channels. 
        For example, 'AI education', 'finance tips', or 'Veritasium'.
    """],
        max_results: Annotated[int, """
        The maximum number of channel search results to return.
        Default is 5, max is 50.
    """] = 5
) -> List[Dict]:
    """
    Search YouTube for channels related to the query.
    Returns a list of channel summaries including ID, title, description, and thumbnail.
    """
    try:
        request = youtube_api.youtube.search().list(
            part="snippet",
            q=query,
            type="channel",
            maxResults=max_results
        )
        response = request.execute()

        return [{
            "channelId": item['id']['channelId'],
            "title": item['snippet']['title'],
            "description": item['snippet']['description'],
            "thumbnails": item['snippet']['thumbnails']
        } for item in response.get('items', [])]

    except Exception as e:
        return [{"error": str(e)}]


@tool(
        name="search_and_introspect_channel",
        description="Search YouTube channels by keyword, then fetch full info and recent videos for the top match.",
        show_result=True,
        cache_results=True,
        cache_ttl=1800
)
def search_and_introspect_channel(
            query: Annotated[str, """
            A keyword or topic to search for relevant YouTube channels. 
            Example: 'machine learning', 'music reviews', 'travel vlogs'.
        """],
            video_count: Annotated[int, """
            Number of recent videos to fetch from the top-matching channel.
            Defaults to 5.
        """] = 5
    ) -> Dict:
        """
        Searches for YouTube channels by query, then fetches full info and videos for the top result.
        """
        try:
            # Step 1: Search channels
            search_response = youtube_api.youtube.search().list(
                part="snippet",
                q=query,
                type="channel",
                maxResults=1
            ).execute()

            if not search_response['items']:
                return {"error": f"No channels found for query: {query}"}

            top_channel = search_response['items'][0]
            channel_id = top_channel['id']['channelId']

            # Step 2: Fetch channel info
            channel_info = fetch_channel_info(channel_id)

            # Step 3: Fetch recent videos
            videos = fetch_videos(channel_id, max_results=video_count)

            return {
                "query": query,
                "channelInfo": channel_info,
                "recentVideos": videos
            }

        except Exception as e:
            return {"error": str(e)}

