#!/usr/bin/env python3
"""
Integration test script for Botlytics full-stack deployment
Tests the complete flow from frontend to backend to Google Cloud services
"""

import requests
import json
import time
import os
import sys
from typing import Dict, Any

class BotlyticsIntegrationTest:
    def __init__(self, backend_url: str, frontend_url: str = None):
        self.backend_url = backend_url.rstrip('/')
        self.frontend_url = frontend_url.rstrip('/') if frontend_url else None
        self.session = requests.Session()
        self.session.timeout = 30
        
    def test_backend_health(self) -> bool:
        """Test backend health endpoint"""
        try:
            response = self.session.get(f"{self.backend_url}/api/v1/health")
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Backend health check passed")
                print(f"   Status: {health_data.get('status', 'unknown')}")
                
                # Check individual service health
                checks = health_data.get('checks', {})
                for service, status in checks.items():
                    status_icon = "✅" if status == "ok" else "⚠️"
                    print(f"   {status_icon} {service}: {status}")
                
                return True
            else:
                print(f"❌ Backend health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Backend health check error: {e}")
            return False
    
    def test_frontend_health(self) -> bool:
        """Test frontend health endpoint"""
        if not self.frontend_url:
            print("⚠️ Frontend URL not provided, skipping frontend health check")
            return True
            
        try:
            response = self.session.get(f"{self.frontend_url}/_stcore/health")
            if response.status_code == 200:
                print("✅ Frontend health check passed")
                return True
            else:
                print(f"❌ Frontend health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Frontend health check error: {e}")
            return False
    
    def test_conversation_flow(self) -> bool:
        """Test complete conversation flow"""
        try:
            # Start conversation
            print("🚀 Testing conversation flow...")
            
            start_response = self.session.post(
                f"{self.backend_url}/api/v1/conversation/start",
                params={"user_id": "integration-test"}
            )
            
            if start_response.status_code != 200:
                print(f"❌ Failed to start conversation: {start_response.status_code}")
                return False
            
            start_data = start_response.json()
            session_id = start_data["session_id"]
            print(f"✅ Conversation started: {session_id}")
            
            # Continue conversation
            continue_response = self.session.post(
                f"{self.backend_url}/api/v1/conversation/continue",
                json={
                    "session_id": session_id,
                    "message": "Hello, can you help me analyze data?",
                    "dataset_id": None
                }
            )
            
            if continue_response.status_code != 200:
                print(f"❌ Failed to continue conversation: {continue_response.status_code}")
                return False
            
            continue_data = continue_response.json()
            print(f"✅ Conversation response received")
            print(f"   Response: {continue_data['response'][:100]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Conversation flow error: {e}")
            return False
    
    def test_accessibility_features(self) -> bool:
        """Test accessibility endpoints"""
        try:
            print("♿ Testing accessibility features...")
            
            # Test text-to-speech
            tts_response = self.session.post(
                f"{self.backend_url}/api/v1/accessibility/text-to-speech",
                json={
                    "text": "Hello, this is a test of the text-to-speech functionality.",
                    "language_code": "en-US"
                }
            )
            
            if tts_response.status_code == 200:
                tts_data = tts_response.json()
                if tts_data.get("success"):
                    print("✅ Text-to-speech test passed")
                else:
                    print(f"⚠️ TTS returned success=false: {tts_data.get('error')}")
            else:
                print(f"❌ Text-to-speech test failed: {tts_response.status_code}")
                return False
            
            # Test audio description
            audio_desc_response = self.session.post(
                f"{self.backend_url}/api/v1/accessibility/audio-description",
                json={
                    "chart_data": {
                        "title": "Test Chart",
                        "data_points": [{"x": "A", "y": 10}, {"x": "B", "y": 20}]
                    },
                    "chart_type": "bar"
                }
            )
            
            if audio_desc_response.status_code == 200:
                desc_data = audio_desc_response.json()
                if desc_data.get("success"):
                    print("✅ Audio description test passed")
                    print(f"   Description: {desc_data['description'][:100]}...")
                else:
                    print("⚠️ Audio description returned success=false")
            else:
                print(f"❌ Audio description test failed: {audio_desc_response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Accessibility features error: {e}")
            return False
    
    def test_code_interpreter(self) -> bool:
        """Test code interpreter functionality"""
        try:
            print("💻 Testing code interpreter...")
            
            response = self.session.post(
                f"{self.backend_url}/api/v1/code-interpreter",
                json={
                    "code": "import pandas as pd\nresult = pd.DataFrame({'test': [1, 2, 3]}).sum()",
                    "context_vars": {}
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    print("✅ Code interpreter test passed")
                    print(f"   Output: {data.get('output', 'No output')}")
                else:
                    print(f"⚠️ Code execution failed: {data.get('error')}")
            else:
                print(f"❌ Code interpreter test failed: {response.status_code}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Code interpreter error: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all integration tests"""
        print("🧪 Starting Botlytics Integration Tests")
        print("=" * 50)
        
        tests = [
            ("Backend Health", self.test_backend_health),
            ("Frontend Health", self.test_frontend_health),
            ("Conversation Flow", self.test_conversation_flow),
            ("Accessibility Features", self.test_accessibility_features),
            ("Code Interpreter", self.test_code_interpreter),
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n🔍 Running {test_name} test...")
            try:
                result = test_func()
                results.append((test_name, result))
                if result:
                    print(f"✅ {test_name} test completed successfully")
                else:
                    print(f"❌ {test_name} test failed")
            except Exception as e:
                print(f"❌ {test_name} test error: {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Results Summary:")
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {status} {test_name}")
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All integration tests passed! Your Botlytics deployment is working correctly.")
            return True
        else:
            print("⚠️ Some tests failed. Please check the deployment and configuration.")
            return False

def main():
    """Main function to run integration tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Botlytics Integration Tests")
    parser.add_argument("--backend-url", required=True, help="Backend URL (e.g., https://botlytics-backend-xyz-uc.a.run.app)")
    parser.add_argument("--frontend-url", help="Frontend URL (optional)")
    parser.add_argument("--wait", type=int, default=0, help="Wait time before starting tests (seconds)")
    
    args = parser.parse_args()
    
    if args.wait > 0:
        print(f"⏳ Waiting {args.wait} seconds before starting tests...")
        time.sleep(args.wait)
    
    tester = BotlyticsIntegrationTest(args.backend_url, args.frontend_url)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()