#!/usr/bin/env python3
"""
Prompts for the Azure Emoji Analyzer application.
"""

# Azure OpenAI Vision API prompt for emoji analysis
EMOJI_ANALYSIS_PROMPT = """
Analyze this emoji image and provide the following information:
  "primary_emotion": "main emotion (joy/sadness/anger/fear/surprise/love/neutral)",
  "secondary_emotions": ["up to 3 additional emotions"],
  "usage_scenarios": ["4-6 specific usage scenarios"],
  "tone": "overall tone (playful/serious/casual/formal/romantic)",
  "context_suggestions": ["3-4 communication contexts"]

Return the response in JSON format where keys are from the above list and values are the corresponding analysis.
"""

# System message for Azure OpenAI chat completion
EMOJI_ANALYSIS_SYSTEM_MESSAGE = "You are an emoji expert. Analyze the visual content of emoji images. Respond only with raw JSON. Do not include markdown formatting or code block markers."

# Text analysis prompt template for extracting user scenarios
TEXT_ANALYSIS_PROMPT_TEMPLATE = """
Analyze the following text and extract the user scenario.

Text: "{text}"

Please provide a JSON response with the following structure:
{{
  "usage_scenarios": ["4-6 specific usage scenarios"],
}}

Return only valid JSON without any markdown formatting or code blocks. Don't include any additional text or explanations.
"""

# System message for text analysis
TEXT_ANALYSIS_SYSTEM_MESSAGE = "You are an expert in text analysis and scenario identification. Respond only with valid JSON without markdown formatting."
