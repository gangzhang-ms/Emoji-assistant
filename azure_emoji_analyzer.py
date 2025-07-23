#!/usr/bin/env python3
"""
Azure OpenAI Emoji Emotion Analyzer - Vision-Enabled LLM Required

A streamlined version using Azure OpenAI Vision service for emoji emotion analysis.
This version analyzes actual emoji images using computer vision and requires a 
vision-enabled Azure OpenAI model (like GPT-4o) for image analysis.
"""

import os
import json
import time
import base64
from pathlib import Path
from typing import Dict, List, Optional

from prompts import EMOJI_ANALYSIS_PROMPT, EMOJI_ANALYSIS_SYSTEM_MESSAGE

try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False
    print("⚠️ OpenAI library not available. Install with: pip install openai")


class AzureEmojiAnalyzer:
    """Azure OpenAI Vision-enabled emoji emotion analyzer - Vision LLM Required"""
    
    def __init__(self, api_key: Optional[str] = None, config_file: str = "azure_config.json"):
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Get Azure OpenAI credentials
        self.api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY') or self.config.get('azure_openai_api_key')
        self.endpoint = os.getenv('AZURE_OPENAI_ENDPOINT') or self.config.get('azure_openai_endpoint')
        self.api_version = self.config.get('azure_openai_api_version', '2024-02-15-preview')
        self.deployment_name = self.config.get('deployment_name', 'gpt-35-turbo')
        
        self.client = None
        
        if not AZURE_OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is required. Install with: pip install openai")
        
        if not self.api_key or self.api_key == "your-azure-openai-api-key-here":
            raise ValueError("Azure OpenAI API key is required. Set it in azure_config.json, AZURE_OPENAI_API_KEY environment variable, or pass api_key parameter.")
        
        if not self.endpoint or self.endpoint == "your-azure-openai-endpoint-here":
            raise ValueError("Azure OpenAI endpoint is required. Set it in azure_config.json or AZURE_OPENAI_ENDPOINT environment variable.")
        
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )
    
    def load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            print(f"⚠️ Warning: Could not load config file {config_file}: {e}")
            return {}
        
    def extract_emoji_name(self, filename: str) -> str:
        """Extract readable name from filename"""
        name = filename.replace('.png', '').replace('_', ' ')
        
        # Handle unicode hex codes
        if name.startswith(('1f', '2', '00')):
            parts = name.split(' ')
            if len(parts) > 1:
                return ' '.join(parts[1:]).title()
        
        return name.title()
    
    def analyze_with_azure_openai(self, emoji_file_path: str) -> Dict:
        """Analyze emoji using Azure OpenAI Vision API"""
        try:
            # Read and encode the image file
            with open(emoji_file_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            prompt = EMOJI_ANALYSIS_PROMPT
            
            response = self.client.chat.completions.create(
                model=self.deployment_name,  # This should be a vision-enabled deployment (e.g., gpt-4o)
                messages=[
                    {
                        "role": "system", 
                        "content": EMOJI_ANALYSIS_SYSTEM_MESSAGE
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                temperature=self.config.get("temperature", 1.0),
                max_tokens=self.config.get("max_tokens", 800)
            )
            
            content = response.choices[0].message.content.strip()
            return json.loads(content)
            
        except FileNotFoundError:
            print(f"  ❌ Image file not found: {emoji_file_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON parsing error: {e}")
            return None
        except Exception as e:
            print(f"  ❌ Azure OpenAI Vision API error: {e}")
            return None
    
    def analyze_emoji(self, filename: str, directory: str = "downloaded_emojis") -> Dict:
        """Analyze a single emoji using Azure OpenAI Vision only"""
        emoji_name = self.extract_emoji_name(filename)
        emoji_file_path = os.path.join(directory, filename)
        
        result = self.analyze_with_azure_openai(emoji_file_path)
        
        if not result:
            raise ValueError(f"Failed to analyze emoji '{emoji_name}' with Azure OpenAI Vision")
        
        # Add metadata
        result['filename'] = filename
        result['emoji_name'] = emoji_name
        
        return result
    
    def analyze_all(self, directory: str, limit: Optional[int] = None) -> Dict[str, Dict]:
        """Analyze all emojis in directory"""
        emoji_dir = Path(directory)
        png_files = list(emoji_dir.glob("*.png"))
        
        if limit:
            png_files = png_files[:limit]
        
        print(f"🔍 Analyzing {len(png_files)} emojis...")
        print(f"🤖 Using: Azure OpenAI Vision API (Required)")
        print(f"📍 Endpoint: {self.endpoint}")
        print(f"🚀 Deployment: {self.deployment_name}")
        print("👁️ Note: Using vision-enabled model for actual image analysis")
        print("-" * 50)
        
        results = {}
        
        for i, png_file in enumerate(png_files, 1):
            filename = png_file.name
            print(f"[{i:3d}/{len(png_files)}] {filename[:30]:<30} -> ", end="", flush=True)
            
            try:
                analysis = self.analyze_emoji(filename, directory)
                # Use filename without suffix as key
                key = filename.replace('.png', '')
                results[key] = analysis
                print(f"✅ {analysis['primary_emotion']}")
                
                # Rate limiting for API calls
                if i < len(png_files):
                    time.sleep(self.config.get("rate_limit_delay", 0.3))
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                results[filename] = {
                    'filename': filename,
                    'emoji_name': self.extract_emoji_name(filename),
                    'error': str(e),
                    'primary_emotion': 'unknown'
                }
        
        return results


def main():
    print("🎭 Azure OpenAI Vision Emoji Emotion Analyzer")
    print("=" * 50)
    print("👁️ Note: This version uses vision-enabled models (e.g., GPT-4o) to analyze actual emoji images")
    print()
    
    # Check directory
    emoji_dir = "downloaded_emojis"
    if not os.path.exists(emoji_dir):
        print(f"❌ Directory '{emoji_dir}' not found!")
        return
    
    # Check for Azure config
    config_file = "azure_config.json"
    if not os.path.exists(config_file):
        print(f"⚠️ Azure config file '{config_file}' not found!")
        print("Please run: python setup_azure_config.py")
        return
    
    # Verify Azure OpenAI library is available
    if not AZURE_OPENAI_AVAILABLE:
        print("❌ OpenAI library is not installed.")
        print("Please install it with: pip install openai")
        return
    
    # Get limit
    limit_str = input("How many emojis to analyze? (number or 'all'): ").strip()
    limit = None
    if limit_str.lower() != 'all':
        try:
            limit = int(limit_str)
        except ValueError:
            pass
    
    # Analyze
    try:
        analyzer = AzureEmojiAnalyzer()
        results = analyzer.analyze_all(emoji_dir, limit)
        
        # Save results
        output_file = "azure_emoji_emotions.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Summary
        emotion_counts = {}
        for analysis in results.values():
            emotion = analysis.get('primary_emotion', 'unknown')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"\n📊 Analysis Summary:")
        print(f"Total emojis: {len(results)}")
        print("Emotion distribution:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion:12} | {count:3d}")
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your Azure OpenAI credentials are configured correctly.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
