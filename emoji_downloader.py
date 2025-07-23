#!/usr/bin/env python3
"""
Emoji Downloader for FluentUI Emoji Repository

This script downloads PNG files from the FluentUI emoji repository's 3D folders
and renames them based on the "mappedToEmoticons" field in their metadata.json files.

Repository: https://github.com/microsoft/fluentui-emoji/tree/main/assets
"""

import os
import json
import requests
from pathlib import Path
import time
from urllib.parse import quote
import re

class EmojiDownloader:
    def __init__(self, base_url="https://api.github.com/repos/microsoft/fluentui-emoji/contents/assets", github_token=None, output_folder="downloaded_emojis"):
        self.base_url = base_url
        self.raw_base_url = "https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets"
        self.output_dir = Path(output_folder)
        self.output_dir.mkdir(exist_ok=True)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'emoji-downloader/1.0'
        })
        
        # Add GitHub token if provided
        if github_token:
            self.session.headers.update({
                'Authorization': f'token {github_token}'
            })
            print("✓ Using GitHub token for authentication")
        else:
            print("⚠ No GitHub token provided - using unauthenticated requests (rate limited)")
            print("  Consider setting GITHUB_TOKEN environment variable for higher rate limits")
        
    def get_folder_list(self):
        """Get list of all folders in the assets directory"""
        print("Fetching folder list from GitHub API...")
        
        # Add a warning for users without tokens
        if 'Authorization' not in self.session.headers:
            print("⚠ Running without GitHub token - using conservative rate limiting")
            print("💡 Consider using 'python alternative_downloader.py' for faster downloads")
        
        try:
            response = self.session.get(self.base_url)
            
            # Handle rate limiting
            if response.status_code == 403:
                rate_limit_remaining = response.headers.get('X-RateLimit-Remaining', 'unknown')
                rate_limit_reset = response.headers.get('X-RateLimit-Reset', 'unknown')
                
                if 'rate limit' in response.text.lower():
                    print(f"❌ GitHub API rate limit exceeded!")
                    print(f"Rate limit remaining: {rate_limit_remaining}")
                    if rate_limit_reset != 'unknown':
                        import datetime
                        reset_time = datetime.datetime.fromtimestamp(int(rate_limit_reset))
                        print(f"Rate limit resets at: {reset_time}")
                    print("\nSolutions:")
                    print("1. Wait for rate limit to reset")
                    print("2. Set GITHUB_TOKEN environment variable with a GitHub personal access token")
                    print("3. Use the alternative folder-list-based approach:")
                    print("   python alternative_downloader.py")
                    return []
            
            response.raise_for_status()
            
            folders = []
            for item in response.json():
                if item['type'] == 'dir':
                    folders.append(item['name'])
            
            print(f"Found {len(folders)} emoji folders")
            return folders
        except requests.RequestException as e:
            print(f"Error fetching folder list: {e}")
            return []
    
    def get_metadata(self, folder_name):
        """Get metadata.json for a specific emoji folder"""
        metadata_url = f"{self.raw_base_url}/{quote(folder_name)}/metadata.json"
        try:
            response = self.session.get(metadata_url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Warning: Could not fetch metadata for {folder_name}: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in metadata for {folder_name}: {e}")
            return None
    
    def get_png_filename(self, folder_name):
        """Get the PNG filename from the 3D folder"""
        # Based on repository structure, PNG files follow pattern: {folder_name_with_underscores}_3d.png
        folder_name_underscore = folder_name.replace(' ', '_').lower()
        expected_filename = f"{folder_name_underscore}_3d.png"
        
        # Verify the file exists by checking the 3D folder contents
        folder_3d_url = f"{self.base_url}/{quote(folder_name)}/3D"
        try:
            response = self.session.get(folder_3d_url)
            response.raise_for_status()
            
            available_files = [item['name'] for item in response.json() if item['name'].endswith('.png')]
            
            if expected_filename in available_files:
                return expected_filename
            elif available_files:
                # Return the first PNG file found
                return available_files[0]
            else:
                print(f"Warning: No PNG file found in 3D folder for {folder_name}")
                return None
                
        except requests.RequestException as e:
            print(f"Warning: Could not access 3D folder for {folder_name}: {e}")
            # Try to use the expected filename anyway
            return expected_filename
    
    def sanitize_filename(self, filename):
        """Sanitize filename for Windows/cross-platform compatibility"""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Remove leading/trailing spaces and dots
        filename = filename.strip(' .')
        
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        
        return filename
    
    def download_emoji(self, folder_name, png_filename, output_name):
        """Download the PNG file and save with the specified name"""
        png_url = f"{self.raw_base_url}/{quote(folder_name)}/3D/{png_filename}"
        output_path = self.output_dir / f"{output_name}.png"
        
        try:
            response = self.session.get(png_url)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ Downloaded: {folder_name} -> {output_name}.png")
            return True
        except requests.RequestException as e:
            print(f"✗ Failed to download {folder_name}: {e}")
            return False
    
    def process_emojis(self, limit=None):
        """Process emojis with optional limit"""
        folders = self.get_folder_list()
        if not folders:
            print("No folders found. Exiting.")
            return
        
        if limit:
            folders = folders[:limit]
            print(f"\nProcessing first {len(folders)} emoji folders...")
        else:
            print(f"\nProcessing all {len(folders)} emoji folders...")
        
        # Calculate estimated time for users without tokens
        if 'Authorization' not in self.session.headers:
            estimated_time_minutes = (len(folders) * 2.5) / 60  # 2.5 seconds per emoji (including processing time)
            print(f"⏱ Estimated time: {estimated_time_minutes:.1f} minutes (due to rate limiting)")
            print("💡 For faster downloads, consider using: python alternative_downloader.py")
        
        successful_downloads = 0
        failed_downloads = 0
        
        print("=" * 50)
        
        for i, folder_name in enumerate(folders, 1):
            print(f"\n[{i}/{len(folders)}] Processing: {folder_name}")
            
            # Get metadata
            metadata = self.get_metadata(folder_name)
            if not metadata:
                failed_downloads += 1
                continue
            
            # Extract mappedToEmoticons
            mapped_emoticons = metadata.get('mappedToEmoticons', [])
            if not mapped_emoticons:
                print(f"Warning: No mappedToEmoticons found for {folder_name}")
                # Use folder name as fallback
                output_name = self.sanitize_filename(folder_name)
            else:
                # Use the first mapped emoticon, or join multiple with underscore
                if len(mapped_emoticons) == 1:
                    output_name = self.sanitize_filename(mapped_emoticons[0])
                else:
                    output_name = self.sanitize_filename('_'.join(mapped_emoticons))
            
            # Check if file already exists
            output_path = self.output_dir / f"{output_name}.png"
            if output_path.exists():
                print(f"⚠ Skipping: {output_name}.png (already exists)")
                successful_downloads += 1
                continue
            
            # Get PNG filename
            png_filename = self.get_png_filename(folder_name)
            if not png_filename:
                failed_downloads += 1
                continue
            
            # Download the file
            if self.download_emoji(folder_name, png_filename, output_name):
                successful_downloads += 1
            else:
                failed_downloads += 1
            
            # Small delay to be respectful to the API (longer if no token)
            if 'Authorization' in self.session.headers:
                time.sleep(0.1)  # Shorter delay with token
            else:
                remaining_items = len(folders) - i
                if remaining_items > 0:
                    print(f"  💤 Waiting 2 seconds... ({remaining_items} items remaining)")
                time.sleep(2.0)  # Much longer delay without token to avoid rate limits
        
        print("\n" + "=" * 50)
        print(f"Download Summary:")
        print(f"✓ Successful: {successful_downloads}")
        print(f"✗ Failed: {failed_downloads}")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
    
    def process_all_emojis(self):
        """Main process to download all emojis"""
        self.process_emojis(limit=None)

def main():
    """Main entry point"""
    print("FluentUI Emoji Downloader")
    print("=" * 30)
    
    # Check for GitHub token
    github_token = os.environ.get('GITHUB_TOKEN')
    
    downloader = EmojiDownloader(github_token=github_token)
    
    try:
        downloader.process_all_emojis()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    
    print(f"\n✅ All downloads saved to: {downloader.output_dir.absolute()}")
    print("Done!")

if __name__ == "__main__":
    main()
