"""
OpenHands Cloud API Client
==================

Uses OpenHands Cloud (app.all-hands.dev) for agent execution.

This client works with Python 3.10 without needing Docker or SDK.

Usage:
    from openhands_client import OpenHandsClient
    client = OpenHandsClient()
    result = client.execute_task("Fix this bug: ...")
"""

import requests
import time
import json
from typing import Dict, Optional, Any, List

DEFAULT_BASE_URL = "https://app.all-hands.dev/api/v1"
DEFAULT_TIMEOUT = 180
API_KEY = "sk-oh-YIyCdPwr7ucD0V7osJoZJcdlmjq0ahQ1"


class OpenHandsClient:
    """HTTP client for OpenHands Cloud API."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = API_KEY

    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def check_status(self) -> Dict[str, Any]:
        """Check if Cloud API is accessible."""
        try:
            response = requests.get(
                f"{self.base_url}/",
                headers=self._get_headers(),
                timeout=10
            )
            if response.status_code in (200, 401):
                return {"status": "running", "message": "Cloud API accessible"}
            return {"status": "error", "message": f"HTTP {response.status_code}"}
        except requests.exceptions.ConnectionError as e:
            return {"status": "not_running", "message": f"Connection error: {e}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def create_conversation(self, task: str) -> Dict[str, Any]:
        """
        Create a new conversation/task via Cloud API.
        
        POST /api/v1/app-conversations
        """
        payload = {
            "initial_message": {
                "content": [
                    {"type": "text", "text": task}
                ]
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/app-conversations",
                headers=self._get_headers(),
                json=payload,
                timeout=30
            )
            print(f"[DEBUG] Create conversation response: {response.status_code}")
            print(f"[DEBUG] Response text: {response.text[:500]}")
            
            if response.status_code in (200, 201):
                return {"status": "success", "data": response.json()}
            else:
                return {
                    "status": "error",
                    "message": f"HTTP {response.status_code}: {response.text[:200]}"
                }
        except requests.exceptions.ConnectionError as e:
            return {"status": "error", "message": f"Connection error: {e}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def poll_start_task(self, task_id: str, max_wait: int = 120) -> Dict[str, Any]:
        """
        Poll for start task completion.
        
        GET /api/v1/app-conversations/start-tasks?id={task_id}
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(
                    f"{self.base_url}/app-conversations/start-tasks",
                    params={"ids": task_id},
                    headers=self._get_headers(),
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"[DEBUG] Poll response type: {type(data)}, content: {str(data)[:200]}")
                    
                    # Response is a list directly, not a dict with key
                    if isinstance(data, list):
                        if len(data) > 0:
                            task_info = data[0]
                            status = task_info.get("status", "")
                            app_conv_id = task_info.get("app_conversation_id", "")
                            
                            print(f"[DEBUG] Start task status: {status}, app_conv_id: {app_conv_id}")
                            
                            if status == "READY" and app_conv_id:
                                return {"status": "success", "data": task_info}
                            elif status == "FAILED":
                                return {"status": "error", "message": "Start task failed"}
                            elif app_conv_id:
                                # Even if not READY, if we have ID, use it
                                return {"status": "success", "data": task_info}
                
                time.sleep(5)
                
            except Exception as e:
                print(f"[DEBUG] Poll error: {e}")
                time.sleep(5)
        
        return {"status": "timeout", "message": f"No response within {max_wait}s"}

    def get_conversation_status(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation status."""
        try:
            response = requests.get(
                f"{self.base_url}/app-conversations/{conversation_id}",
                headers=self._get_headers(),
                timeout=10
            )
            if response.status_code == 200:
                return {"status": "success", "data": response.json()}
            return {"status": "error", "message": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_conversation_events(self, conversation_id: str) -> List[Dict]:
        """Get conversation events."""
        try:
            response = requests.get(
                f"{self.base_url}/app-conversations/{conversation_id}/events",
                headers=self._get_headers(),
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("events", [])
            return []
        except Exception:
            return []

    def wait_for_completion(self, conversation_id: str, max_wait: int = 180) -> bool:
        """Wait for conversation to complete."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(
                    f"{self.base_url}/app-conversations/search",
                    params={"ids": conversation_id},
                    headers=self._get_headers(),
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    items = data.get("items", [])
                    
                    if items and len(items) > 0:
                        item = items[0]
                        exec_status = item.get("execution_status", "")
                        sandbox_status = item.get("sandbox_status", "")
                        print(f"[DEBUG] Execution status: {exec_status}, Sandbox: {sandbox_status}")
                        
                        if exec_status == "finished" and sandbox_status == "stopped":
                            return True
                        if exec_status == "failed" or sandbox_status == "error":
                            return False
            
            except Exception as e:
                print(f"[DEBUG] Completion poll error: {e}")
            
            time.sleep(5)
        
        return False

    def execute_task(self, task: str) -> Dict[str, Any]:
        """
        Execute a coding task via OpenHands Cloud.
        
        Workflow:
        1. Create conversation (POST /app-conversations)
        2. Poll for start task (GET /app-conversations/start-tasks)
        3. Poll for completion (GET /app-conversations/{id})
        4. Get events (GET /app-conversations/{id}/events)
        """
        print(f"[INFO] Executing task via OpenHands Cloud...")
        
        # Step 1: Create conversation
        conv_result = self.create_conversation(task)
        
        if conv_result.get("status") != "success":
            return {
                "output": "",
                "status": "error",
                "message": f"Failed to create conversation: {conv_result.get('message', 'Unknown')}"
            }
        
        data = conv_result.get("data", {})
        task_id = data.get("id")
        
        if not task_id:
            app_conv_id = data.get("app_conversation_id")
            if app_conv_id:
                conversation_id = app_conv_id
                print(f"[DEBUG] Using conversation ID directly: {conversation_id}")
            else:
                return {"output": "", "status": "error", "message": "No task ID or conversation ID returned"}
        else:
            print(f"[DEBUG] Created task ID: {task_id}")
            
            # Step 2: Poll for start task (if needed)
            poll_result = self.poll_start_task(task_id, max_wait=60)
            
            if poll_result.get("status") != "success":
                return {
                    "output": "",
                    "status": "error",
                    "message": f"Start task failed: {poll_result.get('message', 'Unknown')}"
                }
            
            task_data = poll_result.get("data", {})
            conversation_id = task_data.get("app_conversation_id")
            
            if not conversation_id:
                return {"output": "", "status": "error", "message": "No app_conversation_id in start task"}
            
            print(f"[DEBUG] Conversation ID ready: {conversation_id}")
        
        # Step 3: Wait for completion
        print(f"[DEBUG] Waiting for agent to complete...")
        completed = self.wait_for_completion(conversation_id, max_wait=self.timeout)
        
        if not completed:
            return {
                "output": "",
                "status": "timeout",
                "message": f"Conversation did not complete within {self.timeout}s"
            }
        
        # Step 4: Get events and extract output
        events = self.get_conversation_events(conversation_id)
        
        output_parts = []
        for event in events:
            if event.get("type") == "message":
                content = event.get("message", {}).get("content", [])
                for part in content:
                    if part.get("type") == "text":
                        output_parts.append(part.get("text", ""))
        
        final_output = "\n".join(output_parts)
        
        if not final_output:
            final_output = str(events[-1].get("message", "")) if events else ""
        
        return {
            "output": final_output,
            "status": "success",
            "conversation_id": conversation_id
        }


def create_openhands_client(timeout: int = DEFAULT_TIMEOUT) -> OpenHandsClient:
    """Factory function to create OpenHands client."""
    return OpenHandsClient(timeout=timeout)


def test_connection() -> bool:
    """Quick test to check if Cloud API is accessible."""
    client = OpenHandsClient(timeout=5)
    status = client.check_status()
    is_running = status.get("status") == "running"
    print(f"OpenHands Cloud connection: {is_running}")
    if is_running:
        print(f"  Message: {status.get('message', 'OK')}")
    else:
        print(f"  Details: {status.get('message', 'Unknown error')}")
    return is_running


if __name__ == "__main__":
    print("Testing OpenHands Cloud API connection...")
    if test_connection():
        print("\nOpenHands Cloud is accessible!")
        
        client = OpenHandsClient()
        print("\nRunning test task...")
        result = client.execute_task("Write a simple Python function that returns 'Hello, World!'")
        
        print(f"\nResult status: {result.get('status')}")
        if result.get("output"):
            print(f"Output:\n{result['output'][:500]}...")