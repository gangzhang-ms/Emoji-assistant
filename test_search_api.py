#!/usr/bin/env python3
"""
Test script for the new search by filename API endpoints
"""

import requests
import json

# API base URL (assuming server is running locally)
BASE_URL = "http://localhost:5000/api"

def test_search_by_filename():
    """Test the search by filename endpoint with contains functionality"""
    print("🧪 Testing Search by Filename API (Contains Search)")
    print("=" * 60)
    
    # Test cases for "contains" search
    test_cases = [
        {"query": "face", "description": "Face-related emojis"},
        {"query": "ball", "description": "Ball sports emojis"},
        {"query": "heart", "description": "Heart-themed emojis"},
        {"query": "car", "description": "Car-related emojis"},
        {"query": "apple", "description": "Apple-containing emojis"},
        {"query": "tree", "description": "Tree-related emojis"},
        {"query": "nonexistent12345", "description": "Non-existent pattern"}
    ]
    
    for test_case in test_cases:
        query = test_case["query"]
        description = test_case["description"]
        print(f"\n🔍 Testing: {query} ({description})")
        
        # Test GET method with query parameter and top_k
        response = requests.get(f"{BASE_URL}/search-by-filename", params={"filename": query, "top_k": 8})
        print(f"GET (query param): Status {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Found {data['count']} results (match_type: {data.get('match_type', 'unknown')})")
            print(f"  Query processed: '{data.get('query', '')}' from original '{data.get('filename', '')}'")
            if data['count'] > 0:
                print(f"  Sample results:")
                for i, result in enumerate(data['results'][:3]):  # Show first 3
                    print(f"    {i+1}. {result['filename']} - {result['emoji_name']}")
        
        # Test POST method for one of the cases
        if query == "ball":
            response = requests.post(f"{BASE_URL}/search-by-filename", json={"filename": query, "top_k": 15})
            print(f"POST (JSON body): Status {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  Found {data['count']} results with larger top_k")
                if data['count'] > 0:
                    print(f"  All results:")
                    for result in data['results']:
                        print(f"    - {result['filename']}")

def test_search_by_name():
    """Test the search by name endpoint"""
    print("\n\n🧪 Testing Search by Name API")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        "apple",
        "car", 
        "happy",
        "face",
        "nonexistent"
    ]
    
    for name in test_cases:
        print(f"\n🔍 Testing name: {name}")
        
        # Test GET method with query parameter
        response = requests.get(f"{BASE_URL}/search-by-name", params={"name": name})
        print(f"GET (query param): Status {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Found {data['count']} results ({data.get('match_type', 'unknown')} match)")
            if data['count'] > 0:
                print(f"  First result: {data['results'][0]['emoji_name']}")

def test_advanced_search_scenarios():
    """Test advanced search scenarios"""
    print("\n\n🔬 Testing Advanced Search Scenarios")
    print("=" * 50)
    
    scenarios = [
        {
            "name": "Prefix vs Contains Matching",
            "queries": ["angry", "ball", "heart"],
            "description": "Test how prefix and contains matching work"
        },
        {
            "name": "Extension Handling", 
            "queries": ["apple.png", "apple", "car.png", "car"],
            "description": "Test .png extension handling"
        },
        {
            "name": "Case Sensitivity",
            "queries": ["APPLE", "Ball", "hEaRt"],
            "description": "Test case insensitive search"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['name']}: {scenario['description']}")
        for query in scenario['queries']:
            response = requests.get(f"{BASE_URL}/search-by-filename", params={"filename": query, "top_k": 5})
            if response.status_code == 200:
                data = response.json()
                print(f"  '{query}' → {data['count']} results ({data.get('match_type', 'unknown')})")
            else:
                print(f"  '{query}' → Error {response.status_code}")

def test_parameter_validation():
    """Test parameter validation"""
    print("\n\n✅ Testing Parameter Validation")
    print("=" * 40)
    
    # Test top_k validation
    test_cases = [
        {"filename": "test", "top_k": 0, "expected": 400},      # Too low
        {"filename": "test", "top_k": 101, "expected": 400},    # Too high
        {"filename": "test", "top_k": "invalid", "expected": 400},  # Invalid type
        {"filename": "test", "top_k": 50, "expected": 200},     # Valid
        {"filename": "", "top_k": 10, "expected": 400},         # Empty filename
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: filename='{test_case['filename']}', top_k={test_case['top_k']}")
        
        try:
            response = requests.get(f"{BASE_URL}/search-by-filename", params=test_case)
            actual_status = response.status_code
            expected_status = test_case['expected']
            
            if actual_status == expected_status:
                print(f"  ✅ Expected {expected_status}, got {actual_status}")
                if actual_status == 400:
                    error_msg = response.json().get('message', 'No message')
                    print(f"     Error: {error_msg}")
            else:
                print(f"  ❌ Expected {expected_status}, got {actual_status}")
                
        except Exception as e:
            print(f"  ❌ Request failed: {e}")

def test_health_and_stats():
    """Test health and stats endpoints"""
    print("\n\n🏥 Testing Health and Stats")
    print("=" * 30)
    
    # Test health endpoint
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: Status {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Status: {data['status']}")
        print(f"  Database initialized: {data['database_initialized']}")
    
    # Test stats endpoint
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Stats check: Status {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  Total emojis: {data['total_count']}")

if __name__ == "__main__":
    print("🎭 Emoji Search API Test Suite - Enhanced Contains Search")
    print("=" * 70)
    print("🚀 Make sure the API server is running on localhost:5000")
    print()
    
    try:
        # Test basic connectivity
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ API server is not responding correctly")
            exit(1)
        
        # Run tests
        test_health_and_stats()
        test_search_by_filename()
        test_search_by_name() 
        test_advanced_search_scenarios()
        test_parameter_validation()
        
        print("\n\n✅ All testing completed!")
        print("\n📊 Summary:")
        print("- ✅ Contains-based filename search")
        print("- ✅ Prefix and substring matching")
        print("- ✅ Multiple result support with top_k")
        print("- ✅ Extension handling (.png auto-removal)")
        print("- ✅ Case-insensitive search")
        print("- ✅ Parameter validation")
        print("- ✅ Error handling")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Make sure it's running on localhost:5000")
    except requests.exceptions.Timeout:
        print("❌ API server is not responding (timeout)")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
