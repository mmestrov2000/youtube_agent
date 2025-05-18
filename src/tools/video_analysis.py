import os
import torch
from typing import Dict, List, Optional, Tuple, Annotated, Callable, Any
import whisper
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
from pathlib import Path
import tempfile
import yt_dlp
from agno.tools import tool
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from src.tools.helper.helper import _video_to_text

def logger_hook(function_name: str, function_call: Callable, arguments: Dict[str, Any]):
    """Pre-hook function that runs before the tool execution"""
    print(f"About to call {function_name} with arguments: {arguments}")
    result = function_call(**arguments)
    print(f"Function call completed with result: {result}")
    return result

@tool(
    name="video_to_text",
    description="Convert video content to text using Whisper speech recognition model.",
    show_result=True,
    stop_after_tool_call=True,
    tool_hooks=[logger_hook],
    cache_results=True,
    cache_dir="/tmp/agno_cache",
    cache_ttl=3600  # Cache for 1 hour
)
def video_to_text(
    video_id: Annotated[str, """
        The unique identifier of the YouTube video.
        This is the part of the YouTube URL after 'v='. For example:
        - For the URL 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', the `video_id` would be 'dQw4w9WgXcQ'.
    """]
) -> str:
    """
    Convert video content to text using Whisper.
    
    Args:
        video_id (str): YouTube video ID
        model_size (str): Size of the Whisper model to use
        
    Returns:
        str: Transcribed text from the video
        
    Raises:
        Exception: If video download or transcription fails
    """
    return _video_to_text(video_id)

@tool(
    name="analyze_video_content",
    description="Analyze video content with scene-based transcription and sponsor detection.",
    show_result=True,
    stop_after_tool_call=True,
    tool_hooks=[logger_hook],
    cache_results=False,
    cache_dir="/tmp/agno_cache",
    cache_ttl=3600  # Cache for 1 hour
)
def analyze_video_content(
    video_id: Annotated[str, """
        The unique identifier of the YouTube video.
        This is the part of the YouTube URL after 'v='. For example:
        - For the URL 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', the `video_id` would be 'dQw4w9WgXcQ'.
    """]
) -> Dict:
    """
    Analyze video content with scene-based transcription and sponsor detection.
    
    Args:
        video_id (str): YouTube video ID
        
    Returns:
        Dict: Analysis results including:
            - scenes: List of scenes with timestamps, transcriptions, summaries, and sponsor mentions
            - sponsors: List of detected sponsors
            - metadata: Dictionary containing video title and description
            
    Raises:
        Exception: If video download or analysis fails
    """
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