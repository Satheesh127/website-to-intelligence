"""
UI Launcher for Enterprise Knowledge Assistant
==============================================

Quick startup script to launch the Streamlit web interface.
"""

import subprocess
import sys
import os

def main():
    print("🚀 Launching Enterprise Knowledge Assistant Web UI...")
    print("📱 Starting Streamlit server...")
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--theme.base", "light",
            "--theme.primaryColor", "#667eea",
            "--theme.backgroundColor", "#ffffff",
            "--theme.secondaryBackgroundColor", "#f0f2f6",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("💡 Try running: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()