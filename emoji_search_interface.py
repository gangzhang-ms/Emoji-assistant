#!/usr/bin/env python3
"""
Emoji Vector Search Interface

Interactive search interface for finding similar emojis using vector similarity
"""

import os
import json
import sys
from typing import List, Dict, Optional
from emoji_vector_storage import EmojiVectorStorage

class EmojiSearchInterface:
    """Interactive search interface for emoji vector database"""
    
    def __init__(self, config_file: str = "azure_config.json"):
        """Initialize the search interface"""
        self.storage = EmojiVectorStorage(config_file)
        self.storage.logger.info("🔍 Emoji search interface initialized")
    
    def search_emojis(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for emojis similar to the query"""
        return self.storage.search_similar_emojis(query, top_k)
    
    def search_by_emotion(self, emotion: str, top_k: int = 10) -> List[Dict]:
        """Search for emojis by primary emotion"""
        try:
            # Text-based search for emotion
            results = self.storage.search_client.search(
                search_text=f"primary_emotion:{emotion}",
                select=["id", "emoji_name", "primary_emotion", "secondary_emotions", "usage_scenarios", "tone"],
                top=top_k
            )
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result['id'],
                    'emoji_name': result['emoji_name'],
                    'primary_emotion': result['primary_emotion'],
                    'secondary_emotions': result['secondary_emotions'],
                    'usage_scenarios': result['usage_scenarios'],
                    'tone': result['tone'],
                    'score': result.get('@search.score', 0)
                })
            
            return formatted_results
            
        except Exception as e:
            self.storage.logger.error(f"❌ Emotion search failed: {e}")
            return []
    
    def get_emoji_stats(self) -> Dict:
        """Get statistics about the emoji database"""
        try:
            # Get total count
            results = self.storage.search_client.search(
                search_text="*",
                select=["primary_emotion", "tone"],
                top=1000  # Assuming we don't have more than 1000 emojis
            )
            
            emotions = {}
            tones = {}
            total_count = 0
            
            for result in results:
                total_count += 1
                
                # Count emotions
                emotion = result.get('primary_emotion', 'unknown')
                emotions[emotion] = emotions.get(emotion, 0) + 1
                
                # Count tones
                tone = result.get('tone', 'unknown')
                tones[tone] = tones.get(tone, 0) + 1
            
            return {
                'total_count': total_count,
                'emotions': emotions,
                'tones': tones
            }
            
        except Exception as e:
            self.storage.logger.error(f"❌ Stats retrieval failed: {e}")
            return {}
    
    def display_results(self, results: List[Dict], title: str = "Search Results"):
        """Display search results in a formatted way"""
        if not results:
            print("❌ No results found.")
            return
        
        print(f"\n🔍 {title}")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. 📱 {result['emoji_name']}")
            print(f"   🎭 Primary Emotion: {result['primary_emotion']}")
            
            if result.get('secondary_emotions'):
                print(f"   💭 Secondary: {', '.join(result['secondary_emotions'])}")
            
            print(f"   🎵 Tone: {result['tone']}")
            
            if result.get('usage_scenarios'):
                scenarios = result['usage_scenarios'][:2]  # Show first 2 scenarios
                print(f"   💡 Usage: {', '.join(scenarios)}")
            
            if result.get('score'):
                print(f"   ⭐ Similarity Score: {result['score']:.3f}")
    
    def display_stats(self, stats: Dict):
        """Display database statistics"""
        if not stats:
            print("❌ Unable to retrieve statistics.")
            return
        
        print(f"\n📊 Database Statistics")
        print("=" * 40)
        print(f"Total Emojis: {stats['total_count']}")
        
        print(f"\n🎭 Emotions Distribution:")
        for emotion, count in sorted(stats['emotions'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_count']) * 100
            print(f"  {emotion:12} | {count:3d} ({percentage:4.1f}%)")
        
        print(f"\n🎵 Tones Distribution:")
        for tone, count in sorted(stats['tones'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_count']) * 100
            print(f"  {tone:12} | {count:3d} ({percentage:4.1f}%)")
    
    def run_interactive_search(self):
        """Run the interactive search interface"""
        print("🎭 Emoji Vector Search Interface")
        print("=" * 50)
        print("Search for emojis using natural language or emotions!")
        print("Commands:")
        print("  /emotion <emotion>  - Search by primary emotion")
        print("  /stats             - Show database statistics")
        print("  /help              - Show this help message")
        print("  /quit              - Exit the interface")
        print()
        
        while True:
            try:
                query = input("🔍 Enter search query: ").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.startswith('/'):
                    if query == '/quit':
                        print("👋 Goodbye!")
                        break
                    elif query == '/help':
                        print("\nCommands:")
                        print("  /emotion <emotion>  - Search by primary emotion")
                        print("  /stats             - Show database statistics")
                        print("  /help              - Show this help message")
                        print("  /quit              - Exit the interface")
                        continue
                    elif query == '/stats':
                        stats = self.get_emoji_stats()
                        self.display_stats(stats)
                        continue
                    elif query.startswith('/emotion '):
                        emotion = query[9:].strip()
                        if emotion:
                            results = self.search_by_emotion(emotion)
                            self.display_results(results, f"Emojis with '{emotion}' emotion")
                        else:
                            print("❌ Please provide an emotion. Example: /emotion joy")
                        continue
                    else:
                        print("❌ Unknown command. Type /help for available commands.")
                        continue
                
                # Regular vector search
                results = self.search_emojis(query)
                self.display_results(results, f"Similar emojis for '{query}'")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    """Main function"""
    print("🎯 Emoji Vector Search Interface")
    print("=" * 50)
    
    # Check for configuration
    config_file = "azure_config.json"
    if not os.path.exists(config_file):
        print(f"❌ Configuration file '{config_file}' not found!")
        print("Please run 'python setup_vector_storage.py' to create configuration.")
        return
    
    # Check if emoji data has been uploaded
    try:
        interface = EmojiSearchInterface(config_file)
        
        # Test if index exists and has data
        stats = interface.get_emoji_stats()
        if not stats or stats.get('total_count', 0) == 0:
            print("❌ No emoji data found in the vector database!")
            print("Please run 'python emoji_vector_storage.py' to upload emoji data first.")
            return
        
        print(f"✅ Connected to vector database with {stats['total_count']} emojis")
        
        # Run interactive search
        interface.run_interactive_search()
        
    except Exception as e:
        print(f"❌ Error initializing search interface: {e}")
        print("Make sure your Azure services are configured correctly.")


if __name__ == "__main__":
    main()
