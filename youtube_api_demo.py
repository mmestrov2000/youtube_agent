import os
import requests
from datetime import datetime
from src.tools.youtube_api import resolve_channel_id, fetch_channel_info, fetch_videos

def download_thumbnail(url: str, video_id: str):
    """Download and save a video thumbnail."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Create thumbnails directory if it doesn't exist
            os.makedirs('thumbnails', exist_ok=True)
            
            # Save the thumbnail
            with open(f'thumbnails/{video_id}.jpg', 'wb') as f:
                f.write(response.content)
            print(f"✓ Thumbnail saved for video {video_id}")
        else:
            print(f"✗ Failed to download thumbnail for video {video_id}")
    except Exception as e:
        print(f"✗ Error downloading thumbnail: {e}")

def format_duration(duration: str) -> str:
    """Convert ISO 8601 duration to human-readable format."""
    # Remove 'PT' prefix
    duration = duration[2:]
    
    # Initialize variables
    hours = 0
    minutes = 0
    seconds = 0
    
    # Parse hours
    if 'H' in duration:
        hours, duration = duration.split('H')
        hours = int(hours)
    
    # Parse minutes
    if 'M' in duration:
        minutes, duration = duration.split('M')
        minutes = int(minutes)
    
    # Parse seconds
    if 'S' in duration:
        seconds = int(duration.replace('S', ''))
    
    # Format the duration
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"

def get_channel_handle():
    """Get and validate the channel handle from user input."""
    while True:
        handle = input("\nEnter YouTube channel handle (e.g., @channelname): ").strip()
        
        # Add @ if not present
        if not handle.startswith('@'):
            handle = '@' + handle
            
        # Basic validation
        if len(handle) > 1:  # More than just @
            return handle
        print("Invalid handle. Please try again.")

def main():
    print("\n=== YouTube Channel Analyzer ===\n")
    
    # Get channel handle from user
    channel_handle = get_channel_handle()
    
    try:
        # 1. Resolve channel ID from handle
        print("\n1. Resolving channel ID...")
        channel_id = resolve_channel_id(channel_handle)
        print(f"✓ Channel ID: {channel_id}\n")
        
        # 2. Fetch channel information
        print("2. Fetching channel information...")
        channel_info = fetch_channel_info(channel_id)
        print("\nChannel Details:")
        print(f"Title: {channel_info['title']}")
        print(f"Description: {channel_info['description'][:200]}...")  # First 200 chars
        print(f"Subscribers: {channel_info['subscriberCount']:,}")
        print(f"Total Videos: {channel_info['videoCount']:,}")
        print(f"Total Views: {channel_info['viewCount']:,}\n")
        
        # 3. Fetch recent videos
        print("3. Fetching recent videos...")
        videos = fetch_videos(channel_id, max_results=3)
        
        print("\nRecent Videos:")
        for i, video in enumerate(videos, 1):
            # Parse publish date
            publish_date = datetime.strptime(
                video['publishedAt'], 
                "%Y-%m-%dT%H:%M:%SZ"
            ).strftime("%B %d, %Y at %H:%M UTC")
            
            # Format duration
            duration = format_duration(video['duration'])
            
            print(f"\nVideo {i}:")
            print(f"Title: {video['title']}")
            print(f"Description: {video['description'][:200]}...")  # First 200 chars
            print(f"Published: {publish_date}")
            print(f"Duration: {duration}")
            print(f"Views: {video['viewCount']:,}")
            print(f"Likes: {video['likeCount']:,}")
            print(f"Comments: {video['commentCount']:,}")
            
            # Download thumbnail if available
            if 'maxres' in video['thumbnails']:
                download_thumbnail(
                    video['thumbnails']['maxres']['url'],
                    video['id']
                )
            elif 'high' in video['thumbnails']:
                download_thumbnail(
                    video['thumbnails']['high']['url'],
                    video['id']
                )
            print("-" * 80)
            
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()