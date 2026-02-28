import requests

BASE_URL = "http://localhost:12345/v1/chat/completions"
MODEL = "/software/data/pytorch/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307/"

video_url = "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/video/N1cdUjctpG8.mp4"

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize what happens in this video in 5 sentences."},
                {"type": "video_url", "video_url": {"url": video_url}},
            ],
        }
    ],
    "max_tokens": 256,
}

r = requests.post(BASE_URL, json=payload, timeout=300)
print(r.status_code)
print(r.text)
