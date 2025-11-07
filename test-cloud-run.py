#!/usr/bin/env python3
"""
Cloud Run specific integration tests
Tests production deployment on GCP Cloud Run
"""

import requests
import json
import time
import sys
import argparse
from typing import Dict, Any, List

class CloudRunTester:
    """Test suite specifically for Cloud Run deployments"""
    
    def __init__(self, backend_url: str, frontend_url: str = None):
        self.backend_url = backend_url.rstrip('/')
        self.frontend_url = frontend_url.rstrip('/') if frontend_url else None
        self.session = requests.Session()
        self.session.timeout = 60  # Cloud Run can have cold starts
        self.results = []
        
    def run_test(self, name: str, test_func) -> bool:
        """Run a single test and track results"""
        print(f"\n🧪 Testing: {name}")
        try:
            result = test_func()
            if result:
                print(f"✅ PASS: {name}")
                self.results.append((name, True, None))
                return True
            else:
                print(f"❌ FAIL: {name}")
                self.results.append((name, False, "Test returned False"))
                return False
        except Exception as e:
            print(f"❌ ERROR: {name} - {str(e)}")
            self.results.append((name, False, str(e)))
            return False
    
    def test_backend_health(self) -> bool:
        """Test backend health endpoint"""
        response = self.session.get(f"{self.backend_url}/api/v1/health")
        
        if response.status_code != 200:
            print(f"   Status code: {response.status_code}")
            return False
        
        data = response.json()
        print(f"   Status: {data.get('status')}")
        print(f"   Environment: {data.get('environment')}")
        
        # Check all services
        checks = data.get('checks', {})
        all_ok = True
        for service, status in checks.items():
            if isinstance(status, dict):
                print(f"   ✓ {service}: {json.dumps(status)}")
            else:
                icon = "✓" if status == "ok" else "⚠"
                print(f"   {icon} {service}: {status}")
                if status not in ["ok", "not_configured"]:
                    all_ok = False
        
        return data.get('status') in ['healthy', 'degraded'] and all_ok
    
    def test_cold_start_performance(self) -> bool:
        """Test cold start performance"""
        print("   Testing cold start latency...")
        
        start_time = time.time()
        response = self.session.get(f"{self.backend_url}/api/v1/health")
        latency = time.time() - start_time
        
        print(f"   Cold start latency: {latency:.2f}s")
        
        # Cloud Run cold starts should be under 10 seconds with cpu-boost
        if latency > 10:
            print(f"   ⚠️  Warning: Cold start took {latency:.2f}s (expected <10s)")
            return False
        
        return response.status_code == 200
    
    def test_warm_request_performance(self) -> bool:
        """Test warm request performance"""
        print("   Testing warm request latency...")
        
        # Make a few requests to warm up
        for _ in range(3):
            self.session.get(f"{self.backend_url}/api/v1/health")
        
        # Measure warm request
        latencies = []
        for i in range(5):
            start_time = time.time()
            response = self.session.get(f"{self.backend_url}/api/v1/health")
            latency = time.time() - start_time
            latencies.append(latency)
            
            if response.status_code != 200:
                return False
        
        avg_latency = sum(latencies) / len(latencies)
        print(f"   Average warm latency: {avg_latency:.3f}s")
        print(f"   Min: {min(latencies):.3f}s, Max: {max(latencies):.3f}s")
        
        # Warm requests should be fast
        return avg_latency < 1.0
    
    def test_conversation_with_memory(self) -> bool:
        """Test multi-turn conversation with memory"""
        print("   Starting conversation...")
        
        # Start conversation
        response = self.session.post(
            f"{self.backend_url}/api/v1/conversation/start",
            params={"user_id": "cloud-run-test"}
        )
        
        if response.status_code != 200:
            print(f"   Failed to start: {response.status_code}")
            return False
        
        data = response.json()
        session_id = data["session_id"]
        print(f"   Session ID: {session_id}")
        
        # Continue conversation
        messages = [
            "Hello, what can you help me with?",
            "Can you analyze data?",
            "What tools do you have available?"
        ]
        
        for msg in messages:
            response = self.session.post(
                f"{self.backend_url}/api/v1/conversation/continue",
                json={
                    "session_id": session_id,
                    "message": msg,
                    "dataset_id": None
                }
            )
            
            if response.status_code != 200:
                print(f"   Failed on message: {msg}")
                return False
            
            result = response.json()
            print(f"   Q: {msg[:50]}...")
            print(f"   A: {result['response'][:100]}...")
        
        return True
    
    def test_code_interpreter_security(self) -> bool:
        """Test code interpreter security"""
        print("   Testing code execution security...")
        
        # Test safe code
        safe_code = "import pandas as pd\nresult = pd.DataFrame({'test': [1, 2, 3]}).sum()"
        response = self.session.post(
            f"{self.backend_url}/api/v1/code-interpreter",
            json={"code": safe_code}
        )
        
        if response.status_code != 200:
            print("   Safe code execution failed")
            return False
        
        data = response.json()
        if not data.get("success"):
            print("   Safe code should succeed")
            return False
        
        print("   ✓ Safe code executed")
        
        # Test dangerous code (should be blocked)
        dangerous_codes = [
            "import os; os.system('ls')",
            "exec('print(1)')",
            "__import__('subprocess')"
        ]
        
        for dangerous_code in dangerous_codes:
            response = self.session.post(
                f"{self.backend_url}/api/v1/code-interpreter",
                json={"code": dangerous_code}
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    print(f"   ⚠️  Dangerous code was not blocked: {dangerous_code[:50]}")
                    return False
        
        print("   ✓ Dangerous code blocked")
        return True
    
    def test_accessibility_features(self) -> bool:
        """Test accessibility endpoints"""
        print("   Testing text-to-speech...")
        
        response = self.session.post(
            f"{self.backend_url}/api/v1/accessibility/text-to-speech",
            json={
                "text": "This is a test of the text-to-speech system.",
                "language_code": "en-US"
            }
        )
        
        if response.status_code != 200:
            print(f"   TTS failed: {response.status_code}")
            return False
        
        data = response.json()
        if not data.get("success"):
            print(f"   TTS error: {data.get('error')}")
            return False
        
        print("   ✓ Text-to-speech working")
        
        # Test audio description
        print("   Testing audio descriptions...")
        
        response = self.session.post(
            f"{self.backend_url}/api/v1/accessibility/audio-description",
            json={
                "chart_data": {
                    "title": "Test Chart",
                    "data_points": [{"x": "A", "y": 10}, {"x": "B", "y": 20}]
                },
                "chart_type": "bar"
            }
        )
        
        if response.status_code != 200:
            print(f"   Audio description failed: {response.status_code}")
            return False
        
        data = response.json()
        if not data.get("success"):
            print("   Audio description should succeed")
            return False
        
        print("   ✓ Audio descriptions working")
        return True
    
    def test_concurrent_requests(self) -> bool:
        """Test concurrent request handling"""
        print("   Testing concurrent requests...")
        
        import concurrent.futures
        
        def make_request():
            response = self.session.get(f"{self.backend_url}/api/v1/health")
            return response.status_code == 200
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        success_count = sum(results)
        print(f"   Successful requests: {success_count}/10")
        
        return success_count >= 9  # Allow 1 failure
    
    def test_error_handling(self) -> bool:
        """Test error handling"""
        print("   Testing error handling...")
        
        # Test invalid dataset
        response = self.session.get(f"{self.backend_url}/api/v1/datasets/invalid-id")
        
        if response.status_code not in [404, 500]:
            print(f"   Expected 404/500, got {response.status_code}")
            return False
        
        print("   ✓ Invalid dataset handled correctly")
        
        # Test invalid conversation
        response = self.session.post(
            f"{self.backend_url}/api/v1/conversation/continue",
            json={
                "session_id": "invalid-session",
                "message": "test"
            }
        )
        
        if response.status_code not in [404, 500]:
            print(f"   Expected 404/500, got {response.status_code}")
            return False
        
        print("   ✓ Invalid conversation handled correctly")
        return True
    
    def test_metrics_endpoint(self) -> bool:
        """Test Prometheus metrics endpoint"""
        print("   Testing metrics endpoint...")
        
        response = self.session.get(f"{self.backend_url}/metrics")
        
        if response.status_code != 200:
            print(f"   Metrics endpoint failed: {response.status_code}")
            return False
        
        metrics_text = response.text
        
        # Check for key metrics
        required_metrics = [
            "botlytics_requests_total",
            "botlytics_request_duration_seconds"
        ]
        
        for metric in required_metrics:
            if metric not in metrics_text:
                print(f"   Missing metric: {metric}")
                return False
        
        print("   ✓ Metrics endpoint working")
        return True
    
    def test_frontend_health(self) -> bool:
        """Test frontend health"""
        if not self.frontend_url:
            print("   Frontend URL not provided, skipping")
            return True
        
        response = self.session.get(f"{self.frontend_url}/_stcore/health")
        
        if response.status_code != 200:
            print(f"   Frontend health check failed: {response.status_code}")
            return False
        
        print("   ✓ Frontend healthy")
        return True
    
    def run_all_tests(self) -> bool:
        """Run all Cloud Run tests"""
        print("=" * 60)
        print("🚀 CLOUD RUN INTEGRATION TESTS")
        print("=" * 60)
        print(f"Backend URL: {self.backend_url}")
        if self.frontend_url:
            print(f"Frontend URL: {self.frontend_url}")
        print("=" * 60)
        
        tests = [
            ("Backend Health Check", self.test_backend_health),
            ("Cold Start Performance", self.test_cold_start_performance),
            ("Warm Request Performance", self.test_warm_request_performance),
            ("Conversation with Memory", self.test_conversation_with_memory),
            ("Code Interpreter Security", self.test_code_interpreter_security),
            ("Accessibility Features", self.test_accessibility_features),
            ("Concurrent Requests", self.test_concurrent_requests),
            ("Error Handling", self.test_error_handling),
            ("Metrics Endpoint", self.test_metrics_endpoint),
            ("Frontend Health", self.test_frontend_health),
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
            time.sleep(1)  # Brief pause between tests
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for _, success, _ in self.results if success)
        total = len(self.results)
        
        for name, success, error in self.results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {name}")
            if error and not success:
                print(f"         Error: {error}")
        
        print("=" * 60)
        print(f"🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests passed! Cloud Run deployment is production-ready.")
            return True
        elif passed >= total * 0.8:
            print("⚠️  Most tests passed. Review failures before production use.")
            return False
        else:
            print("❌ Multiple test failures. Deployment needs attention.")
            return False

def main():
    parser = argparse.ArgumentParser(description="Cloud Run Integration Tests")
    parser.add_argument("--backend-url", required=True, 
                       help="Backend URL (e.g., https://botlytics-backend-xyz.run.app)")
    parser.add_argument("--frontend-url", 
                       help="Frontend URL (optional)")
    parser.add_argument("--wait", type=int, default=30,
                       help="Wait time before starting tests (default: 30s for cold start)")
    
    args = parser.parse_args()
    
    if args.wait > 0:
        print(f"⏳ Waiting {args.wait} seconds for Cloud Run to warm up...")
        time.sleep(args.wait)
    
    tester = CloudRunTester(args.backend_url, args.frontend_url)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
