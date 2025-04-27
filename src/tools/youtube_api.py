# YouTube API Wrapper
# This module will handle YouTube API interactions 

from typing import Dict, List, Optional
import os
import re
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import yt_dlp

# Load environment variables from .env file
load_dotenv()

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

def download_video(video_id: str, output_path: str = "downloads", quality: str = "best") -> str:
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

def search_channel_videos(channel_id: str, search_term: str, max_results: int = 10) -> List[Dict]:
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

def resolve_channel_id(channel_identifier: str) -> str:
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

def fetch_channel_info(channel_id: str) -> Dict:
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

def fetch_videos(channel_id: str, max_results: int = 10) -> List[Dict]:
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

def fetch_comments(video_id: str, max_results: int = 100) -> List[Dict]:
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