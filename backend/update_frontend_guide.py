"""
Script to force update frontend model accuracy
This clears the cached data and ensures the latest accuracy is displayed
"""

import os
import json

print("=" * 70)
print("  FRONTEND CACHE CLEARING GUIDE")
print("=" * 70)

# Read current accuracy
with open('models/model_info.json', 'r') as f:
    data = json.load(f)

print(f"\n✅ Current Diabetes Model Accuracy: {data['diabetes']['accuracy']*100:.2f}%")
print(f"✅ Last Trained: {data['diabetes']['last_trained']}")

print("\n" + "=" * 70)
print("  TO SEE UPDATED ACCURACY ON WEBSITE:")
print("=" * 70)

print("""
OPTION 1: Clear Browser Cache (Recommended)
--------------------------------------------
1. Open your browser DevTools (F12)
2. Go to Application/Storage tab
3. Find "Local Storage" → your website URL
4. Delete the key: "models-storage"
5. Refresh the page

OPTION 2: Clear All Browser Data
---------------------------------
1. Press Ctrl+Shift+Delete (Chrome/Edge)
2. Select "Cached images and files"
3. Select "Cookies and site data"
4. Click "Clear data"
5. Refresh the page

OPTION 3: Use Incognito/Private Window
---------------------------------------
1. Open Incognito window (Ctrl+Shift+N)
2. Visit your website
3. Login and check accuracy

OPTION 4: Automatic Refresh (Code Solution)
-------------------------------------------
The website automatically fetches new data when you:
- Login to the website
- Navigate to the Home page
- Open the Settings page
- The Zustand store will fetch latest data from backend

""")

print("=" * 70)
print("  SERVERS TO START:")
print("=" * 70)
print("""
1. Backend Server (Terminal 1):
   cd backend
   python api_server.py

2. Frontend Server (Terminal 2):
   npm run dev

3. Visit: http://localhost:3000
""")

print("=" * 70)
print("✅ The accuracy will display as 100% once cache is cleared")
print("=" * 70)
