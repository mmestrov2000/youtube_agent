import os
import yt_dlp
import re
from typing import Dict, List
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat

import whisper
import tempfile


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


def _download_video(video_id: str, output_path: str, quality: str) -> str:
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
    
def _resolve_channel_id(channel_identifier: str) -> str:
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
    
def _fetch_video_details(video_id: str) -> Dict:
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
    
def _search_youtube_channel_videos(channel_id: str, search_term: str, max_results: int = 10) -> List[Dict]:
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
            video_details = _fetch_video_details(video_id)
            videos.append(video_details)
        
        return videos
        
    except HttpError as e:
        raise Exception(f"Error searching channel videos: {str(e)}")
    
def _fetch_channel_info(channel_id: str) -> Dict:
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
    
def _fetch_videos(channel_id: str, max_results: int = 10) -> List[Dict]:
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
            video_details = _fetch_video_details(video_id)
            videos.append(video_details)
        
        return videos
    except HttpError as e:
        raise Exception(f"Error fetching videos: {str(e)}")
    
def _fetch_comments(video_id: str, max_results: int = 25) -> List[Dict]:
    comments: List[Dict] = []
    next_page_token = None

    try:
        while len(comments) < max_results:
            # fetch up to 100 per page (API limit), or however many you still need
            batch_size = min(100, max_results - len(comments))
            request = youtube_api.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=batch_size,
                order="time",            # newest first
                pageToken=next_page_token
            )
            response = request.execute()

            for item in response.get('items', []):
                top = item.get('snippet', {}).get('topLevelComment', {})
                snip = top.get('snippet', {})

                # ensure we at least have an ID and text before appending
                comment_id = top.get('id')
                text = snip.get('textDisplay')
                if not comment_id or text is None:
                    continue

                comments.append({
                    "id": comment_id,
                    "author": snip.get('authorDisplayName', 'Unknown'),
                    "text": text,
                    "likeCount": snip.get('likeCount', 0),
                    "publishedAt": snip.get('publishedAt')
                })

            # prepare for next page (if any)
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

        return comments

    except HttpError as e:
        raise Exception(f"Error fetching comments: {e}")
    
def _introspect_channel(identifier: str, max_videos: int = 10) -> Dict:
    try:
        # Step 1: Resolve to Channel ID
        channel_id = _resolve_channel_id(identifier)

        # Step 2: Fetch channel info
        channel_info = _fetch_channel_info(channel_id)

        # Step 3: Fetch videos
        recent_videos = _fetch_videos(channel_id, max_videos)

        return {
            "channel_info": channel_info,
            "recent_videos": recent_videos
        }

    except Exception as e:
        return {"error": str(e)}
    
def _search_youtube_channels(query: str, max_results: int = 5) -> List[Dict]:
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
    
def _fetch_video_statistics(channel_id: str, max_results: int = 10, months: int = 6, min_duration_minutes: int = 3) -> List[Dict]:
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
            maxResults=50  # Increased to ensure we get enough videos after filtering
        )
        response = request.execute()
        
        # Get video IDs
        video_ids = [item['contentDetails']['videoId'] for item in response['items']]
        
        # Fetch statistics and content details for all videos in one request
        stats_request = youtube_api.youtube.videos().list(
            part="statistics,contentDetails,snippet",
            id=','.join(video_ids)
        )
        stats_response = stats_request.execute()
        
        # Calculate the cutoff date (X months ago)
        from datetime import datetime, timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=30 * months)
        
        # Process and filter statistics
        video_stats = []
        for video in stats_response['items']:
            # Parse publish date
            publish_date = datetime.strptime(video['snippet']['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
            
            # Parse duration (ISO 8601 format)
            duration_str = video['contentDetails']['duration']
            duration_minutes = 0
            
            # Handle hours
            if 'H' in duration_str:
                hours_part = duration_str.split('H')[0]
                if 'T' in hours_part:
                    hours = int(hours_part.split('T')[1])
                else:
                    hours = int(hours_part)
                duration_minutes += hours * 60
            
            # Handle minutes
            if 'M' in duration_str:
                minutes_part = duration_str.split('M')[0]
                if 'H' in minutes_part:
                    minutes = int(minutes_part.split('H')[-1])
                elif 'T' in minutes_part:
                    minutes = int(minutes_part.split('T')[-1])
                else:
                    minutes = int(minutes_part)
                duration_minutes += minutes
            
            # Handle seconds (convert to minutes if needed)
            if 'S' in duration_str:
                seconds_part = duration_str.split('S')[0]
                if 'M' in seconds_part:
                    seconds = int(seconds_part.split('M')[-1])
                elif 'H' in seconds_part:
                    seconds = int(seconds_part.split('H')[-1])
                elif 'T' in seconds_part:
                    seconds = int(seconds_part.split('T')[-1])
                else:
                    seconds = int(seconds_part)
                duration_minutes += seconds / 60
            
            # Apply filters
            if publish_date < cutoff_date or duration_minutes < min_duration_minutes:
                continue
                
            stats = video['statistics']
            video_stats.append({
                "videoId": video['id'],
                "viewCount": int(stats.get('viewCount', 0)),
                "likeCount": int(stats.get('likeCount', 0)),
                "commentCount": int(stats.get('commentCount', 0)),
                "favoriteCount": int(stats.get('favoriteCount', 0)),
                "durationMinutes": round(duration_minutes, 2),
                "publishedAt": video['snippet']['publishedAt']
            })
            
            # Stop if we have enough videos
            if len(video_stats) >= max_results:
                break
        
        return video_stats
    except HttpError as e:
        raise Exception(f"Error fetching video statistics: {str(e)}")

def _search_and_introspect_channel(query: str, video_count: int = 5) -> Dict:
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
            channel_info = _fetch_channel_info(channel_id)

            # Step 3: Fetch recent videos
            videos = _fetch_videos(channel_id, max_results=video_count)

            return {
                "query": query,
                "channelInfo": channel_info,
                "recentVideos": videos
            }

        except Exception as e:
            return {"error": str(e)}

def _video_to_text(video_id: str) -> str:
    # Initialize Whisper model
    model_size = "base"
    whisper_model = whisper.load_model(model_size)
    
    # Download video
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': f'{tempfile.gettempdir()}/%(id)s.%(ext)s'
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        url = f"https://www.youtube.com/watch?v={video_id}"
        info = ydl.extract_info(url, download=True)
        video_path = f"{tempfile.gettempdir()}/{video_id}.mp4"
    
    try:
        # Transcribe the video
        result = whisper_model.transcribe(video_path)
        
        # Clean up the downloaded video
        os.remove(video_path)
        
        return result["text"]
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(video_path):
            os.remove(video_path)
        raise e
    
def _analyze_video_content(video_id: str) -> Dict:
    try:
        # Get video transcription
        transcription = _video_to_text(video_id)
        
        # Download video for metadata
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': f'{tempfile.gettempdir()}/%(id)s.%(ext)s'
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            url = f"https://www.youtube.com/watch?v={video_id}"
            info = ydl.extract_info(url, download=True)
            video_path = f"{tempfile.gettempdir()}/{video_id}.mp4"
            description = info.get('description', '')
            title = info.get('title', '')
        
        try:
            # Split transcription into 60-second scenes
            # Assuming average speaking rate of 150 words per minute
            words = transcription.split()
            words_per_scene = 150  # 150 words per minute
            scenes = []
            
            # Create an agent for scene analysis
            scene_analyzer = Agent(
                name="Scene Analyzer",
                role="Analyze video scenes for content and sponsor mentions",
                model=OpenAIChat(id="gpt-4.1-mini"),
                instructions=[
                    "Analyze the given scene text and provide:",
                    "1. A brief, informative summary of what was discussed in the scene",
                    "2. If any sponsor/brand was mentioned in this specific scene, return the sponsor name",
                    "3. If no sponsor was mentioned, return an empty string",
                    "Return the response in JSON format with 'summary' and 'sponsor' fields."
                ]
            )
            
            for i in range(0, len(words), words_per_scene):
                scene_words = words[i:i + words_per_scene]
                scene_text = ' '.join(scene_words)
                
                # Get scene analysis from LLM
                scene_analysis = scene_analyzer.run(f"Scene text: {scene_text}")
                
                # Parse the LLM response
                try:
                    analysis_data = eval(scene_analysis.content)  # Convert string to dict
                    summary = analysis_data.get('summary', '')
                    sponsor = analysis_data.get('sponsor', '')
                except:
                    # Fallback in case of parsing error
                    summary = scene_text.split('.')[0][:50] + '...'
                    sponsor = ''
                
                scenes.append({
                    'start': i // words_per_scene * 60,
                    'end': (i // words_per_scene + 1) * 60,
                    #'text': scene_text,
                    #'summary': summary,
                    'sponsor': sponsor
                })
            
            # Use Agno agent with GPT-4.1-mini for overall sponsor detection
            sponsor_agent = Agent(
                name="Sponsor Detector",
                role="Detect sponsors from video descriptions",
                model=OpenAIChat(id="gpt-4.1-mini"),
                instructions=[
                    "Analyze the video description and list all sponsors/brands mentioned.",
                    "Return only a comma-separated list of sponsor names, nothing else.",
                    "Be precise and only include actual sponsors, not just mentioned brands."
                ]
            )
            
            sponsor_response = sponsor_agent.run(f"Video description: {description}")
            
            # Parse sponsor response
            sponsors = []
            if sponsor_response and sponsor_response.content:
                sponsor_names = sponsor_response.content.strip().split(',')
                sponsors = [{'name': name.strip()} for name in sponsor_names if name.strip()]
            
            return {
                "scenes": scenes,
                "sponsors": sponsors,
                "metadata": {
                    "title": title,
                    "description": description
                }
            }
            
        finally:
            # Clean up the downloaded video
            if os.path.exists(video_path):
                os.remove(video_path)
                
    except Exception as e:
        raise Exception(f"Failed to analyze video content: {str(e)}")