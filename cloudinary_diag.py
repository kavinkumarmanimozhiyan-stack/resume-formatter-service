import os
from dotenv import load_dotenv
load_dotenv()

secret = os.getenv("CLOUDINARY_API_SECRET") or ""
key = os.getenv("CLOUDINARY_API_KEY") or ""
cloud = os.getenv("CLOUDINARY_CLOUD_NAME") or ""

print("cloud_name:", repr(cloud))
print("api_key:", repr(key), "len:", len(key))
print("api_secret length:", len(secret))
print("api_secret first2/last2:", secret[:2], "...", secret[-2:] if len(secret) >= 2 else secret)
print("contains hash:", "#" in secret)
print("contains whitespace:", any(c.isspace() for c in secret))
print("contains quotes still embedded:", secret.startswith('"') or secret.startswith("'"))

import cloudinary
cloudinary.config(cloud_name=cloud, api_key=key, api_secret=secret, secure=True)
try:
    import cloudinary.api
    result = cloudinary.api.ping()
    print("PING OK:", result)
except Exception as e:
    print("PING FAILED:", e)
