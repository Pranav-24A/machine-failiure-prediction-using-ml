"""
Script to test that Streamlit starts and responds properly on HTTP port.
"""

import subprocess
import sys
import time
import requests

def run_server_check():
    print("Launching Streamlit server in subprocess...")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.headless",
        "true",
        "--server.port",
        "8501",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    connected = False
    try:
        # Wait up to 15 seconds for server to start
        for attempt in range(15):
            time.sleep(1)
            try:
                r = requests.get("http://localhost:8501/_stcore/health", timeout=2)
                if r.status_code == 200:
                    print(f"Streamlit server successfully responded with: {r.text.strip()}")
                    connected = True
                    break
            except Exception:
                pass

        if not connected:
            print("Failed to reach Streamlit server within 15 seconds.")
            proc.terminate()
            stdout, stderr = proc.communicate(timeout=5)
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            sys.exit(1)

        print("Streamlit startup and health check PASSED!")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    test_streamlit_server()
