import ssl
from pytubefix import YouTube
from pytubefix.cli import on_progress

# Create an unverified context for SSL
ssl._create_default_https_context = ssl._create_unverified_context

url1 = "https://www.youtube.com/watch?v=SOG0GmKts_I"
url2 = "https://www.youtube.com/watch?v=7Lf0jEgz9BA&list=TLPQMjAwNTIwMjXjKus6t9GEQQ"

yt = YouTube(url2, on_progress_callback=on_progress)
print(yt.title)

ys = yt.streams.get_highest_resolution()
ys.download()
