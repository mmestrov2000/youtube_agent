import os
import requests
from datetime import datetime
from src.tools.youtube_api import resolve_channel_id, fetch_channel_info, fetch_videos, search_channel_videos, download_video
import dotenv

dotenv.load_dotenv()

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

def display_video_details(video: dict, index: int = None):
    """Display detailed information about a video."""
    # Parse publish date
    publish_date = datetime.strptime(
        video['publishedAt'], 
        "%Y-%m-%dT%H:%M:%SZ"
    ).strftime("%B %d, %Y at %H:%M UTC")
    
    # Format duration
    duration = format_duration(video['duration'])
    
    # Print video details
    if index is not None:
        print(f"\nVideo {index}:")
    else:
        print("\nVideo Details:")
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

def get_video_quality():
    """Get and validate the video quality from user input."""
    print("\nAvailable quality options:")
    print("1. Best quality")
    print("2. 1080p")
    print("3. 720p")
    print("4. 480p")
    print("5. 360p")
    
    while True:
        choice = input("\nEnter your choice (1-5): ").strip()
        if choice == "1":
            return "best"
        elif choice == "2":
            return "1080p"
        elif choice == "3":
            return "720p"
        elif choice == "4":
            return "480p"
        elif choice == "5":
            return "360p"
        print("Invalid choice. Please try again.")

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
        
        # 3. Ask user what they want to do
        print("\nWhat would you like to do?")
        print("1. View recent videos")
        print("2. Search for videos")
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "1":
            # Fetch recent videos
            print("\n3. Fetching recent videos...")
            videos = fetch_videos(channel_id, max_results=5)
            
            print("\nRecent Videos:")
            for i, video in enumerate(videos, 1):
                display_video_details(video, i)
                
        elif choice == "2":
            # Search for videos
            search_term = input("\nEnter search term: ").strip()
            max_results = int(input("Enter maximum number of results (default 5): ").strip() or "5")
            
            print(f"\nSearching for '{search_term}' in {channel_info['title']}...")
            videos = search_channel_videos(channel_id, search_term, max_results)
            
            if videos:
                print(f"\nFound {len(videos)} videos:")
                for i, video in enumerate(videos, 1):
                    display_video_details(video, i)
            else:
                print(f"\nNo videos found matching '{search_term}'")
        else:
            print("\nInvalid choice. Please run the script again.")
            return
            
        # 4. Ask if user wants to download a video
        print("\nWould you like to download a video?")
        print("1. Yes")
        print("2. No")
        download_choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if download_choice == "1":
            # Get video number to download
            video_number = int(input("\nEnter the video number to download: ").strip())
            if 1 <= video_number <= len(videos):
                video = videos[video_number - 1]
                
                # Get video quality
                quality = get_video_quality()
                
                # Download the video
                print(f"\nDownloading video: {video['title']}")
                try:
                    output_path = download_video(video['id'], quality=quality)
                    print(f"✓ Video downloaded successfully to: {output_path}")
                except Exception as e:
                    print(f"✗ Error downloading video: {e}")
            else:
                print("\nInvalid video number.")
            
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()