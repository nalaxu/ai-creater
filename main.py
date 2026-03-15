import asyncio
import time
import uuid
import os
import re
import base64
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai
import PIL.Image

# Init SDK
genai.configure(
    api_key="sk-b5b6a796fe94489987b09ff932cfa7b4",
    transport='rest',
    client_options={'api_endpoint': 'http://127.0.0.1:8045'}
)

image_model = genai.GenerativeModel('gemini-3.1-flash-image')

app = FastAPI()

os.makedirs("outputs", exist_ok=True)
os.makedirs("static", exist_ok=True)

class JobQueue:
    def __init__(self):
        self.jobs = {}
        self.queue = asyncio.Queue()

    async def add_job(self, mode, prompts, source_image_paths=None, system_prompt=""):
        job_id = str(uuid.uuid4())
        
        # Calculate total tasks (Cartesian product: images x prompts)
        total_tasks = len(prompts)
        if mode == 'i2i' and source_image_paths:
            total_tasks = len(prompts) * max(1, len(source_image_paths))
            
        self.jobs[job_id] = {
            "id": job_id,
            "mode": mode,
            "status": "queued",
            "total": total_tasks,
            "completed": 0,
            "failed": 0,
            "results": [],
            "created_at": time.time(),
            "started_at": None,
            "eta": None,
            "system_prompt": system_prompt
        }
        await self.queue.put((job_id, prompts, source_image_paths, system_prompt))
        return self.jobs[job_id]

job_queue = JobQueue()

async def process_queue():
    while True:
        job_id, prompts, source_image_paths, system_prompt = await job_queue.queue.get()
        job = job_queue.jobs[job_id]
        job["status"] = "processing"
        job["started_at"] = time.time()
        
        mode = job["mode"]
        
        # Flatten tasks list [ (prompt, image_path), ... ]
        tasks = []
        if mode == 'i2i' and source_image_paths:
            for img_path in source_image_paths:
                for p in prompts:
                    tasks.append((p, img_path))
        else:
            for p in prompts:
                tasks.append((p, None))
                
        avg_time = 5.0
        
        for i, (prompt, img_path) in enumerate(tasks):
            start_time = time.time()
            try:
                contents = []
                
                # Insert System Prompt if provided
                if system_prompt and system_prompt.strip():
                    contents.append(f"System Instructions:\n{system_prompt.strip()}\n\n")
                
                # Handle Image-to-Image mode
                if img_path and os.path.exists(img_path):
                    try:
                        img = PIL.Image.open(img_path)
                        enhanced_prompt = f"Reference image attached. Instruction: {prompt}"
                        contents.append(img)
                        contents.append(enhanced_prompt)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
                        contents.append(prompt) # fallback
                else:
                    contents.append(prompt)
                
                # Generate Content
                response = await asyncio.to_thread(image_model.generate_content, contents)
                
                generated_images = []
                final_text = ""
                
                # Case 1: 模型直接返回完整的图像数据结构（inline_data）
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    data = base64.b64decode(part.inline_data.data) if isinstance(part.inline_data.data, str) else part.inline_data.data
                                    mime = part.inline_data.mime_type
                                    ext = mime.split('/')[-1] if mime else 'png'
                                    filename = f"gen_{uuid.uuid4().hex[:8]}.{ext}"
                                    filepath = os.path.join("outputs", filename)
                                    with open(filepath, "wb") as f:
                                        f.write(data)
                                    generated_images.append(f"/outputs/{filename}")
                                    
                # Case 2: 代理返回了包含Base64数据的Markdown文本（或图片链接）
                if not generated_images:
                    try:
                        text = response.text
                        if text:
                            # 提取 Markdown 格式中的 base64 数据
                            base64_patterns = re.findall(r'!\[.*?\]\((data:image/([^;]+);base64,([^)]+))\)', text)
                            if base64_patterns:
                                for full_data_uri, ext, b64_data in base64_patterns:
                                    filename = f"gen_{uuid.uuid4().hex[:8]}.{ext}"
                                    filepath = os.path.join("outputs", filename)
                                    with open(filepath, "wb") as f:
                                        f.write(base64.b64decode(b64_data))
                                    generated_images.append(f"/outputs/{filename}")
                                
                                # 移除原文本中超大的base64字符串，避免前端崩溃
                                final_text = re.sub(r'!\[.*?\]\((data:image/[^;]+;base64,[^)]+)\)', '', text)
                            else:
                                final_text = text
                    except Exception:
                        pass
                        
                # 拼接生成的本地链接，提供给前端渲染
                result_display = final_text.strip()
                for img_url in generated_images:
                    result_display += f"\n\n![Generated Image]({img_url})"
                    
                if not result_display.strip():
                    result_display = "生成完毕，但未能提取到图片。"

                job["results"].append({
                    "prompt": prompt, 
                    # Tag result with image hash if multi-image to distinguish
                    "source_img": os.path.basename(img_path) if img_path else None,
                    "result": result_display, 
                    "status": "success"
                })
                job["completed"] += 1
            except Exception as e:
                import traceback
                traceback.print_exc()
                job["results"].append({
                    "prompt": prompt, 
                    "source_img": os.path.basename(img_path) if img_path else None,
                    "error": str(e), 
                    "status": "error"
                })
                job["failed"] += 1
                
            elapsed = time.time() - start_time
            avg_time = (avg_time * i + elapsed) / (i + 1)
            rem = job["total"] - job["completed"] - job["failed"]
            job["eta"] = max(0, rem * avg_time)
            
        job["status"] = "completed"
        job["eta"] = 0
        job_queue.queue.task_done()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_queue())

@app.post("/api/jobs")
async def create_job(
    prompts: str = Form(...),
    mode: str = Form(...),
    system_prompt: str = Form(""),
    images: Optional[List[UploadFile]] = File(None)
):
    prompt_list = [p.strip() for p in prompts.split("\n") if p.strip()]
    if not prompt_list:
        return {"error": "No prompts provided"}
        
    source_image_paths = []
    if mode == "i2i" and images:
        for img in images:
            if img and getattr(img, "filename", None):
                path = f"outputs/{uuid.uuid4().hex[:8]}_{img.filename}"
                with open(path, "wb") as f:
                    f.write(await img.read())
                source_image_paths.append(path)
            
    job = await job_queue.add_job(mode, prompt_list, source_image_paths, system_prompt)
    return job

@app.get("/api/jobs")
def get_jobs():
    jobs = sorted(list(job_queue.jobs.values()), key=lambda x: x['created_at'], reverse=True)
    return jobs

# 允许外部访问 outputs 目录的本地文件
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/", StaticFiles(directory="static", html=True), name="static")
