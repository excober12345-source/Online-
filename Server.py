#!/usr/bin/env python3
"""
Simple HTTP server for TradeMaster mobile app
Run this on your computer, then access from iPhone
"""

import http.server
import socketserver
import webbrowser
import os
from datetime import datetime

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def log_message(self, format, *args):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {self.address_string()} - {format % args}")

def main():
    os.chdir(DIRECTORY)
    
    # Create icon files if they don't exist
    create_placeholder_icons()
    
    print(f"""
    ╔═══════════════════════════════════════════════════════╗
    ║            TRADEMASTER MOBILE TRADING APP            ║
    ╚═══════════════════════════════════════════════════════╝
    
    Server starting on: http://localhost:{PORT}
    
    TO INSTALL ON iPHONE:
    1. Make sure your iPhone is on the SAME WIFI as this computer
    2. Find your computer's IP address:
       - Windows: ipconfig (look for IPv4 Address)
       - Mac/Linux: ifconfig or ip addr
    3. On iPhone Safari, go to: http://[YOUR-COMPUTER-IP]:{PORT}
    4. Tap the Share button (⎋) at the bottom
    5. Scroll down and tap "Add to Home Screen"
    6. Name it "TradeMaster" and tap Add
    
    Your trading app will now be on your Home Screen! 🚀
    """)
    
    # Try to open in default browser
    try:
        webbrowser.open(f"http://localhost:{PORT}")
    except:
        pass
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Server running at http://localhost:{PORT}")
        print("Press Ctrl+C to stop the server\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

def create_placeholder_icons():
    """Create placeholder icon files if they don't exist"""
    try:
        # Create a simple icon using PIL if available
        try:
            from PIL import Image, ImageDraw
            import numpy as np
            
            # Create 192x192 icon
            img_192 = Image.new('RGB', (192, 192), color=(30, 58, 138))
            draw = ImageDraw.Draw(img_192)
            draw.ellipse([32, 32, 160, 160], fill=(59, 130, 246))
            draw.text((96, 96), "TM", fill=(255, 255, 255), 
                     font=None, anchor="mm", size=40)
            img_192.save('icon-192.png')
            
            # Create 512x512 icon
            img_512 = Image.new('RGB', (512, 512), color=(30, 58, 138))
            draw = ImageDraw.Draw(img_512)
            draw.ellipse([96, 96, 416, 416], fill=(59, 130, 246))
            draw.text((256, 256), "TRADE", fill=(255, 255, 255), 
                     font=None, anchor="mm", size=80)
            img_512.save('icon-512.png')
            
            print("✅ Created app icons")
            
        except ImportError:
            # Create empty placeholder files
            open('icon-192.png', 'wb').write(b'')
            open('icon-512.png', 'wb').write(b'')
            print("⚠️ Install Pillow for better icons: pip install pillow")
            
    except Exception as e:
        print(f"Note: Could not create icons: {e}")

if __name__ == "__main__":
    main()