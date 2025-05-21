import glob
# Standard library imports
import json
import os
import pathlib
import pickle
import shutil
import subprocess
import time
import uuid

# Third-party library imports
import boto3
import cv2
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import ffmpegcv  # For efficient video processing, potentially with GPU acceleration
import modal  # For serverless compute and deployment on Modal platform
import numpy as np # For numerical operations, especially with image/video data (e.g., OpenCV)
from pydantic import BaseModel # For data validation and settings management (e.g., request bodies)
from google import genai # Google Gemini API client for content generation/analysis
import pysubs2 # For creating and manipulating subtitle files (e.g., .ass format)
from tqdm import tqdm # For displaying progress bars during long operations
import whisperx # For advanced speech-to-text (transcription) with word-level timestamps

# Pydantic model for the request body of the process_video endpoint
class ProcessVideoRequest(BaseModel):
    s3_key: str

# Modal Image: Defines the Docker environment for the Modal functions
image = (modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    # Install system dependencies
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "wget", "libcudnn8", "libcudnn8-dev"]) # Essential for video processing and CUDA
    # Install Python packages from requirements.txt
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["mkdir -p /usr/share/fonts/truetype/custom", # Create directory for custom fonts
                   "wget -O /usr/share/fonts/truetype/custom/Anton-Regular.ttf https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
                   "fc-cache -f -v"]) # Update font cache
    .add_local_dir("asd", "/asd", copy=True)) # Add local directory 'asd' (likely containing Columbia_test.py and its weights) to the image

# Initialize Modal App
app = modal.App("ai-podcast-clipper", image=image)
# Modal Volume: Persistent storage for caching models or other large files across runs
volume = modal.Volume.from_name(
    "ai-podcast-clipper-model-cache", create_if_missing=True
)
# Path where the volume will be mounted inside the Modal container
mount_path = "/root/.cache/torch"

# FastAPI HTTP Bearer authentication scheme
auth_scheme = HTTPBearer()


# Function to create a vertical video from frames, focusing on faces
def create_vertical_video(tracks, scores, pyframes_path, pyavi_path, audio_path, output_path, framerate=25):
    """
    Creates a vertical video (1080x1920) from input frames, focusing on detected faces.
    If no significant face is found, it resizes the video with a blurred background.

    Args:
        tracks: Face tracking data.
        scores: Scores associated with face tracks.
        pyframes_path: Path to the directory containing input frames (as JPGs).
        pyavi_path: Path to a working directory for intermediate video files.
        audio_path: Path to the audio file to be muxed with the video.
        output_path: Path to save the final vertical video.
        framerate: Framerate of the output video.
    """
    # Define target dimensions for vertical video
    target_width = 1080
    target_height = 1920

    # Get a sorted list of frame image files
    flist = glob.glob(os.path.join(pyframes_path, "*.jpg"))
    flist.sort()

    # Initialize a list to store face data for each frame
    faces = [[] for _ in range(len(flist))]

    # Process face tracking data to associate faces with specific frames and calculate their prominence scores
    for tidx, track in enumerate(tracks):
        score_array = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            # Calculate average score over a window to smooth out detection blips
            slice_start = max(fidx - 30, 0)
            slice_end = min(fidx + 30, len(score_array))
            score_slice = score_array[slice_start:slice_end]
            # Ensure score_slice is not empty before calculating mean
            avg_score = float(np.mean(score_slice)
                              if len(score_slice) > 0 else 0)

            faces[frame].append(
                {'track': tidx, 'score': avg_score, 's': track['proc_track']["s"][fidx], 'x': track['proc_track']["x"][fidx], 'y': track['proc_track']["y"][fidx]})

    # Path for the temporary video file (without audio)
    temp_video_path = os.path.join(pyavi_path, "video_only.mp4")
    vout = None # Video writer object

    # Iterate through each frame image to construct the vertical video
    for fidx, fname in tqdm(enumerate(flist), total=len(flist), desc="Creating vertical video"):
        img = cv2.imread(fname)
        if img is None:
            continue

        current_faces = faces[fidx]
        # Find the face with the highest score in the current frame
        max_score_face = max(
            current_faces, key=lambda face: face['score']) if current_faces else None

        # If the best face score is too low, consider no face detected
        if max_score_face and max_score_face['score'] < 0:
            max_score_face = None

        # Initialize the video writer (ffmpegcv.VideoWriterNV for potential GPU acceleration) on the first valid frame
        if vout is None:
            vout = ffmpegcv.VideoWriterNV(
                file=temp_video_path,
                codec=None,
                fps=framerate,
                resize=(target_width, target_height)
            )

        # Determine processing mode: "crop" if a significant face is detected, "resize" otherwise
        if max_score_face:
            mode = "crop"
        else:
            mode = "resize"

        # Resize mode: If no significant face, resize the original frame to fit width and place it on a blurred, scaled background
        if mode == "resize":
            # Resize the main content to fit the target width
            scale = target_width / img.shape[1]
            resized_height = int(img.shape[0] * scale)
            resized_image = cv2.resize(
                img, (target_width, resized_height), interpolation=cv2.INTER_AREA)

            # Calculate scale for the background to fill the target vertical dimensions
            scale_for_bg = max(
                target_width / img.shape[1], target_height / img.shape[0])
            bg_width = int(img.shape[1] * scale_for_bg)
            bg_heigth = int(img.shape[0] * scale_for_bg)

            # Create a blurred version of the original image as background
            blurred_background = cv2.resize(img, (bg_width, bg_heigth))
            blurred_background = cv2.GaussianBlur(
                blurred_background, (121, 121), 0)

            # Crop the blurred background to target dimensions
            crop_x = (bg_width - target_width) // 2
            crop_y = (bg_heigth - target_height) // 2
            blurred_background = blurred_background[crop_y:crop_y +
                                                    target_height, crop_x:crop_x + target_width]

            # Place the resized main content onto the center of the blurred background
            center_y = (target_height - resized_height) // 2
            blurred_background[center_y:center_y +
                               resized_height, :] = resized_image
            
            vout.write(blurred_background)
        
        # Crop mode: If a significant face is detected, scale the frame to fit height and crop horizontally around the face
        elif mode == "crop":
            # Scale image to fit target height
            scale = target_height / img.shape[0]
            resized_image = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frame_width = resized_image.shape[1]

            # Calculate the cropping window around the detected face
            center_x = int(
                # Use face's x-coordinate if available, otherwise center of the frame
                max_score_face["x"] * scale if max_score_face else frame_width // 2)
            top_x = max(min(center_x - target_width // 2,
                        frame_width - target_width), 0)

            image_cropped = resized_image[0:target_height,
                                          top_x:top_x + target_width]

            vout.write(image_cropped)

    # Release the video writer if it was initialized
    if vout:
        vout.release()

    # Use ffmpeg to combine the processed video (video_only.mp4) with the original audio track
    ffmpeg_command = (f"ffmpeg -y -i {temp_video_path} -i {audio_path} "
                      f"-c:v h264 -preset fast -crf 23 -c:a aac -b:a 128k "
                      f"{output_path}")
    subprocess.run(ffmpeg_command, shell=True, check=True, text=True)


def create_subtitles_with_ffmpeg(transcript_segments: list, clip_start: float, clip_end: float, clip_video_path: str, output_path: str, max_words: int = 5):
    """
    Generates an ASS subtitle file from transcript segments and then uses ffmpeg
    to burn these subtitles onto the video using pysubs2.

    Args:
        transcript_segments: List of word segments with start, end, and word.
        clip_start: Start time of the current clip (in seconds).
        clip_end: End time of the current clip (in seconds).
        clip_video_path: Path to the video clip to add subtitles to.
        output_path: Path to save the video with burned-in subtitles.
        max_words: Maximum number of words per subtitle line.
    """
    temp_dir = os.path.dirname(output_path)
    # Define path for the temporary ASS subtitle file
    subtitle_path = os.path.join(temp_dir, "temp_subtitles.ass")

    # Filter segments that fall within the current clip's timeframe
    clip_segments = [segment for segment in transcript_segments
                     if segment.get("start") is not None
                     and segment.get("end") is not None
                     and segment.get("end") > clip_start
                     and segment.get("start") < clip_end
                     ]
    
    # Logic to group words from transcript segments into subtitle lines
    subtitles = []
    current_words = []
    current_start = None
    current_end = None

    for segment in clip_segments:
        word = segment.get("word", "").strip()
        seg_start = segment.get("start")
        seg_end = segment.get("end")

        if not word or seg_start is None or seg_end is None:
            continue

        # Adjust segment times relative to the clip's start time
        start_relative = max(0.0, seg_start - clip_start) # Time relative to the start of the clip
        end_relative = max(0.0, seg_end - clip_start) 

        # Skip words that end before or at the very start of the relative clip timeline
        if end_relative <= 0 and start_relative <=0 : # Ensure word is actually within the clip's duration
            continue

        # Start a new subtitle line
        if not current_words:
            current_start = start_relative
            current_end = end_relative
            current_words = [word]
        elif len(current_words) >= max_words: # If max words per line reached, finalize current subtitle and start new one
            subtitles.append((current_start, current_end, ' '.join(current_words)))
            current_words = [word]
            current_start = start_relative
            current_end = end_relative
        else:
            current_words.append(word)
            current_end = end_relative

    # Add any remaining words as the last subtitle
    if current_words:
        subtitles.append(
            (current_start, current_end, ' '.join(current_words)))
    
    # Initialize an SSAFile object from pysubs2 for creating ASS subtitles
    subs = pysubs2.SSAFile()

    # Set metadata for the subtitle file (important for rendering consistency)
    subs.info["WrapStyle"] = 0
    subs.info["ScaledBorderAndShadow"] = "yes"
    subs.info["PlayResX"] = 1080
    subs.info["PlayResY"] = 1920
    subs.info["ScriptType"] = "v4.00+"

    # Define a new style for the subtitles
    style_name = "Default"
    new_style = pysubs2.SSAStyle()
    new_style.fontname = "Anton"
    new_style.fontsize = 140
    new_style.primarycolor = pysubs2.Color(255, 255, 255)
    new_style.outline = 2.0
    new_style.shadow = 2.0
    new_style.shadowcolor = pysubs2.Color(0, 0, 0, 128)
    new_style.alignment = 2
    new_style.marginl = 50
    new_style.marginr = 50
    new_style.marginv = 50
    new_style.spacing = 0.0

    # Add the defined style to the subtitle file's styles collection
    subs.styles[style_name] = new_style

    # Create SSAEvent (a single subtitle line) for each grouped subtitle
    for i, (start, end, text) in enumerate(subtitles):
        start_time = pysubs2.make_time(s=start)
        end_time = pysubs2.make_time(s=end)
        line = pysubs2.SSAEvent(
            start=start_time, end=end_time, text=text, style=style_name)
        subs.events.append(line)

    # Save the generated subtitles to the .ass file
    subs.save(subtitle_path)

    # Use ffmpeg to burn the subtitles onto the video clip
    ffmpeg_cmd = (f"ffmpeg -y -i {clip_video_path} -vf \"ass={subtitle_path}\" "
                  f"-c:v h264 -preset fast -crf 23 {output_path}")

    subprocess.run(ffmpeg_cmd, shell=True, check=True)


def process_clip(base_dir: str, original_video_path: str, s3_key: str, start_time: float, end_time: float, clip_index: int, transcript_segments: list):
    """
    Processes a single video clip:
    1. Cuts the segment from the original video.
    2. Extracts audio from the segment.
    3. Runs face/speaker analysis using `Columbia_test.py`.
    4. Creates a vertical version of the video segment.
    5. (Optionally) Adds subtitles.
    6. Uploads the final clip to S3.
    """
    clip_name = f"clip_{clip_index}"
    s3_key_dir = os.path.dirname(s3_key)
    output_s3_key = f"{s3_key_dir}/{clip_name}.mp4"
    print(f"Output S3 key: {output_s3_key}")

    clip_dir = base_dir / clip_name # Directory for this specific clip's files
    clip_dir.mkdir(parents=True, exist_ok=True)
    
    # Define paths for various intermediate and output files within the clip's directory
    clip_segment_path = clip_dir / f"{clip_name}_segment.mp4" # Path for the cut video segment
    vertical_mp4_path = clip_dir / "pyavi" / "video_out_vertical.mp4" # Path for the vertical video
    subtitle_output_path = clip_dir / "pyavi" / "video_with_subtitles.mp4" # Path for video with subtitles

    # Create necessary subdirectories for intermediate files (frames, working files, audio)
    (clip_dir / "pywork").mkdir(exist_ok=True)
    pyframes_path = clip_dir / "pyframes"
    pyavi_path = clip_dir / "pyavi"
    audio_path = clip_dir / "pyavi" / "audio.wav"

    pyframes_path.mkdir(exist_ok=True)
    pyavi_path.mkdir(exist_ok=True)

    # 1. Cut the video segment from the original video using ffmpeg
    duration = end_time - start_time
    cut_command = (f"ffmpeg -i {original_video_path} -ss {start_time} -t {duration} "
                   f"{clip_segment_path}")
    subprocess.run(cut_command, shell=True, check=True,
                   capture_output=True, text=True)

    # 2. Extract audio from the cut segment
    extract_cmd = f"ffmpeg -i {clip_segment_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
    subprocess.run(extract_cmd, shell=True,
                   check=True, capture_output=True)

    # Prepare for Columbia_test.py: Copy the cut segment to a location it expects
    shutil.copy(clip_segment_path, base_dir / f"{clip_name}.mp4")

    # 3. Run Columbia_test.py script (for face tracking / active speaker detection)
    #    This script is located in the "/asd" directory within the Modal container.
    # and output 'tracks.pckl' and 'scores.pckl' into the clip's 'pywork' directory.
    columbia_command = (f"python Columbia_test.py --videoName {clip_name} "
                        f"--videoFolder {str(base_dir)} "
                        f"--pretrainModel weight/finetuning_TalkSet.model")

    columbia_start_time = time.time()
    subprocess.run(columbia_command, cwd="/asd", shell=True)
    columbia_end_time = time.time() # Time the execution of the script
    print(
        f"Columbia script completed in {columbia_end_time - columbia_start_time:.2f} seconds")

    # Load the face tracking/scoring results produced by Columbia_test.py
    tracks_path = clip_dir / "pywork" / "tracks.pckl"
    scores_path = clip_dir / "pywork" / "scores.pckl"
    if not tracks_path.exists() or not scores_path.exists():
        raise FileNotFoundError("Tracks or scores not found for clip")

    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)

    with open(scores_path, "rb") as f:
        scores = pickle.load(f)

    # 4. Create the vertical video using the face tracks, scores, extracted frames, and audio
    cvv_start_time = time.time()
    create_vertical_video(
        tracks, scores, pyframes_path, pyavi_path, audio_path, vertical_mp4_path
    )
    cvv_end_time = time.time()
    print(
        f"Clip {clip_index} vertical video creation time: {cvv_end_time - cvv_start_time:.2f} seconds")

    # 5. (Optional) Add subtitles to the vertical video
    #    The call is currently commented out. Uncomment to enable.
    create_subtitles_with_ffmpeg(transcript_segments, start_time, end_time, str(vertical_mp4_path), str(subtitle_output_path))
    final_video_to_upload = subtitle_output_path # If subtitles are added, upload this version
    
    # Determine which video file to upload (with or without subtitles)
    # final_video_to_upload = vertical_mp4_path # Default: upload the vertical video without subtitles

    # 6. Upload the processed clip to S3
    s3_client = boto3.client("s3")
    s3_client.upload_file(
        str(final_video_to_upload), "iaiai-podcast-clipper", output_s3_key)

# Modal class definition: specifies resources, secrets, and methods that run in the Modal environment.
@app.cls(gpu="L40S", timeout=900, retries=0, scaledown_window=20, secrets=[modal.Secret.from_name("ai-podcast-clipper-secret")], volumes={mount_path: volume})
class AiPodcastClipper:
    @modal.enter()  # This method runs once when the Modal container for this class starts up.
    def load_model(self):
        """Loads all necessary models into memory."""
        print("Loading models")

        # Load WhisperX model for speech-to-text
        self.whisperx_model = whisperx.load_model(
            "large-v2", device="cuda", compute_type="float16")

        # Load WhisperX alignment model for word-level timestamps
        self.alignment_model, self.metadata = whisperx.load_align_model(
            language_code="en", device="cuda")

        print("Creating gemini client...")
        # Initialize Google Gemini client using API key from Modal secrets
        self.gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        print("created gemini client...")

        print("transcription models loaded...")

    def identify_moments(self, transcript: dict):
        """
        Uses Google Gemini to identify interesting moments (clips) in the transcript.
        Args:
            transcript: A dictionary or string representation of the transcript (word segments).
        Returns:
            A string containing a JSON list of identified moments (start/end times).
        """
        # Construct the prompt for Gemini, including the transcript
        response = self.gemini_client.models.generate_content(model="gemini-2.5-flash-preview-04-17", contents="""
    This is a podcast video transcript consisting of word, along with each words's start and end time. I am looking to create clips between a minimum of 30 and maximum of 60 seconds long. The clip should never exceed 60 seconds.

    Your task is to find and extract stories, or question and their corresponding answers from the transcript.
    Each clip should begin with the question and conclude with the answer.
    It is acceptable for the clip to include a few additional sentences before a question if it aids in contextualizing the question.

    Please adhere to the following rules:
    - Ensure that clips do not overlap with one another.
    - Start and end timestamps of the clips should align perfectly with the sentence boundaries in the transcript.
    - Only use the start and end timestamps provided in the input. modifying timestamps is not allowed.
    - Format the output as a list of JSON objects, each representing a clip with 'start' and 'end' timestamps: [{"start": seconds, "end": seconds}, ...clip2, clip3]. The output should always be readable by the python json.loads function.
    - Aim to generate longer clips between 40-60 seconds, and ensure to include as much content from the context as viable.

    Avoid including:
    - Moments of greeting, thanking, or saying goodbye.
    - Non-question and answer interactions.

    If there are no valid clips to extract, the output should be an empty list [], in JSON format. Also readable by json.loads() in Python.

    The transcript is as follows:\n\n""" + str(transcript))
        print(f"Identified moments response: {response.text}") # Log Gemini's response
        return response.text

    def transcribe_video(self, base_dir: str, video_path: str) -> str:
        """
        Transcribes the video file using WhisperX.
        Args:
            base_dir: Temporary directory to store intermediate files (like extracted audio).
            video_path: Path to the input video file.
        Returns:
            A JSON string representing the transcript with word-level segments.
        """
        audio_path = base_dir / "audio.wav"
        # Extract audio from the input video file using ffmpeg
        extract_cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(extract_cmd, shell=True,
                       check=True, capture_output=True)

        print("Starting transcription with WhisperX...")
        start_time = time.time()
        
        # Load the extracted audio file
        audio = whisperx.load_audio(str(audio_path))
        # Transcribe the audio using the preloaded WhisperX model
        result = self.whisperx_model.transcribe(audio, batch_size=16)

        # Align the transcription results to get word-level timestamps
        result = whisperx.align(
            result["segments"],
            self.alignment_model,
            self.metadata,
            audio,
            device="cuda",
            return_char_alignments=False
        )

        duration = time.time() - start_time
        print("Transcription and alignment took " + str(duration) + " seconds")

        # Format the transcription result (word segments) into a list of dictionaries
        segments = []

        if "word_segments" in result:
            for word_segment in result["word_segments"]:
                segments.append({
                    "start": word_segment["start"],
                    "end": word_segment["end"],
                    "word": word_segment["word"],
                })

        return json.dumps(segments)

    # Defines a FastAPI POST endpoint that will be served by Modal.
    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        """
        Main API endpoint to process a video.
        Downloads video from S3, transcribes, identifies clips, processes each clip, and uploads results.
        """
        s3_key = request.s3_key

        # Authenticate the request using the provided bearer token
        if token.credentials != os.environ["AUTH_TOKEN"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Incorrect bearer token", headers={"WWW-Authenticate": "Bearer"})

        # Create a unique temporary directory for this processing run using a UUID
        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)

        # Define path for the downloaded video
        video_path = base_dir / "input.mp4"
        print(f"Downloading video s3://iaiai-podcast-clipper/{s3_key} to {video_path}") # Log download
        s3_client = boto3.client("s3")
        s3_client.download_file("iaiai-podcast-clipper",
                                s3_key, str(video_path))

        # Step 1: Transcribe the downloaded video
        transcript_segments_json = self.transcribe_video(base_dir, video_path)
        transcript_segments = json.loads(transcript_segments_json)

        # Step 2: Identify interesting moments for clips using Gemini
        print("Identifying clip moments")
        identified_moments_raw = self.identify_moments(transcript_segments)

        # Clean the JSON string received from Gemini:
        # Remove potential markdown code block fences (```json ... ```)
        cleaned_json_string = identified_moments_raw.strip()
        if cleaned_json_string.startswith("```json"):
            cleaned_json_string = cleaned_json_string[len("```json"):].strip()
        if cleaned_json_string.endswith("```"):
            cleaned_json_string = cleaned_json_string[:-len("```")].strip()
        
        # Parse the cleaned JSON string into a list of clip moments
        clip_moments = json.loads(cleaned_json_string)
        if not clip_moments or not isinstance(clip_moments, list):
            print("Error: Identified moments is not a list")
            clip_moments = []

        print(clip_moments)

        # Step 3: Process each identified clip moment.
        # Note: Currently processes only the first identified moment `[:1]` for testing/dev.
        for index, moment in enumerate(clip_moments[:1]):
            if "start" in moment and "end" in moment:
                print("Processing clip" + str(index) + " from " +
                      str(moment["start"]) + " to " + str(moment["end"]))
                process_clip(base_dir, video_path, s3_key,
                             moment["start"], moment["end"], index, transcript_segments)

        # Clean up the temporary directory created for this run
        if base_dir.exists():
            print(f"Cleaning up temp dir after {base_dir}")
            shutil.rmtree(base_dir, ignore_errors=True)
        
        return {"message": "Video processing initiated.", "run_id": run_id, "processed_clips": len(clip_moments[:1])}

# Local entrypoint for testing the Modal app without deploying.
@app.local_entrypoint()
def main():
    """Function to test the AiPodcastClipper locally."""
    import requests

    # Instantiate the AiPodcastClipper class. If running in a Modal local context,
    # this might trigger the @modal.enter method to load models.
    ai_podcast_clipper = AiPodcastClipper()
    # Get the web URL of the process_video FastAPI endpoint.
    # If running locally via `modal run`, this will be a local URL.
    url = ai_podcast_clipper.process_video.web_url
    # Define an example payload for the POST request
    payload = {
        "s3_key": "test1/mi6-5min.mp4"
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 123123" # Replace with your actual test auth token
    }
    # Make a POST request to the local endpoint
    response = requests.post(url, json=payload,
                             headers=headers)
    response.raise_for_status()
    result = response.json()
    print(result)
