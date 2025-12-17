"""
Clear Communicator - Expression Analysis MVP
A web application that analyzes speech clarity across four dimensions.
"""

import os
import json
import uuid
import secrets
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
import httpx

# Prompt 管理和对比测试相关导入
try:
    from prompt_variants import (
        get_all_prompts, create_prompt_variant, get_prompt_variant,
        update_prompt_variant, delete_prompt_variant
    )
    from test_data_manager import (
        get_all_topics, get_test_cases, add_test_case, get_statistics
    )
    from prompt_comparison import compare_prompts, load_latest_comparison
    PROMPT_OPTIMIZATION_ENABLED = True
except ImportError:
    PROMPT_OPTIMIZATION_ENABLED = False
    print("Warning: Prompt optimization modules not available")

# Load environment variables
load_dotenv()

app = FastAPI(title="Clear Communicator")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API configuration
API_BASE_URL = "https://space.ai-builders.com/backend/v1"
# Support both AI_BUILDER_TOKEN (for deployment) and SUPER_MIND_API_KEY (for local dev)
API_KEY = os.getenv("AI_BUILDER_TOKEN") or os.getenv("SUPER_MIND_API_KEY")

if not API_KEY:
    raise ValueError("API key not found. Please set AI_BUILDER_TOKEN or SUPER_MIND_API_KEY in environment variables")

# User isolation and data collection
USERS_DIR = "users"
ANALYTICS_DIR = "analytics"
ANALYTICS_FILE = os.path.join(ANALYTICS_DIR, "anonymous_data.json")

def get_or_create_user_id(request: Request) -> str:
    """Get or create user ID from cookie."""
    user_id = request.cookies.get('user_id')
    if not user_id:
        # Generate a new user ID
        user_id = secrets.token_urlsafe(16)
    return user_id

def get_user_topics_file(user_id: str) -> str:
    """Get user-specific topics file path."""
    user_dir = os.path.join(USERS_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    return os.path.join(user_dir, "topics.json")

def load_topics(user_id: str):
    """Load topics from user-specific JSON file."""
    topics_file = get_user_topics_file(user_id)
    if os.path.exists(topics_file):
        try:
            with open(topics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    return []
                return data
        except Exception as e:
            print(f"Error loading topics for user {user_id}: {e}")
            return []
    return []

def save_topics(user_id: str, topics):
    """Save topics to user-specific JSON file."""
    topics_file = get_user_topics_file(user_id)
    with open(topics_file, 'w', encoding='utf-8') as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)

def save_anonymous_data(result: dict):
    """Save anonymous data for prompt optimization (includes raw transcription)."""
    os.makedirs(ANALYTICS_DIR, exist_ok=True)
    
    # Prepare anonymous entry with raw transcription
    anonymous_entry = {
        "timestamp": datetime.now().isoformat(),
        "transcription": result.get("transcription", ""),
        "raw_transcription": result.get("raw_transcription", ""),
        "topic_summary": result.get("topic_summary", ""),
        "overall_summary": result.get("overall_summary", {}),
        "dimensions": [
            {
                "name": dim.get("name", ""),
                "score": dim.get("score", 0),
                "analysis": dim.get("analysis", ""),
                "highlights": dim.get("highlights", []),
                "issues": dim.get("issues", []),
                "suggestions": dim.get("suggestions", [])
            }
            for dim in result.get("dimensions", [])
        ]
    }
    
    # Load existing analytics data
    if os.path.exists(ANALYTICS_FILE):
        try:
            with open(ANALYTICS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        except Exception as e:
            print(f"Error loading analytics data: {e}")
            data = []
    else:
        data = []
    
    # Append new entry
    data.append(anonymous_entry)
    
    # Keep only last 10000 entries to prevent file from growing too large
    data = data[-10000:]
    
    # Save analytics data
    try:
        with open(ANALYTICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving analytics data: {e}")

async def generate_title(transcript: str, client: httpx.AsyncClient) -> str:
    """Generate a short title (max 10 characters) from the transcript."""
    transcript_preview = transcript[:200] if len(transcript) > 200 else transcript
    title_prompt = f"""请为以下中文语音转录文本生成一个简洁的标题，用于历史记录。

转录文本：
{transcript_preview}{"..." if len(transcript) > 200 else ""}

要求：
1. 标题不超过10个中文字符
2. 能够概括文本的核心内容或主题
3. 简洁明了，便于识别
4. 只输出标题本身，不要添加"标题："、"Title："等前缀
5. 不要添加引号、冒号或其他标点符号
6. 不要添加任何解释或说明

请直接输出标题，例如：关于工作计划的讨论"""

    try:
        title_response = await client.post(
            f"{API_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-4-fast",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一位专业的文本编辑，擅长为文本生成简洁的标题。你的任务是分析文本内容，提取核心主题，然后生成一个不超过10个中文字符的标题。请只输出标题本身，不要添加任何前缀、引号、markdown标记或解释。"
                    },
                    {
                        "role": "user",
                        "content": title_prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 50
            }
        )
        
        if title_response.status_code == 200:
            title_data = title_response.json()
            title = title_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            # Clean up title
            if title.startswith("```"):
                lines = title.split("\n")
                if len(lines) > 1:
                    title = "\n".join(lines[1:-1]) if len(lines) > 2 else lines[1] if len(lines) > 1 else title
                title = title.replace("```", "").strip()
            
            title = title.strip('"').strip("'").strip('"').strip("'").strip()
            title = title.replace("标题：", "").replace("标题:", "").strip()
            title = title.replace("Title:", "").replace("Title：", "").strip()
            
            if len(title) > 10:
                title = title[:10]
            
            if title and len(title) > 0:
                return title
    except Exception as e:
        print(f"Error generating title: {e}")
        pass
    
    return transcript[:10] if transcript else "未命名表达"

async def compare_attempts(current_result: dict, previous_attempts: List[dict], client: httpx.AsyncClient) -> tuple:
    """Compare current attempt with previous attempts and generate improvement praise and tags."""
    if not previous_attempts:
        return False, None, []
    
    # Get the most recent previous attempt
    previous_attempt = previous_attempts[-1]
    previous_result = previous_attempt.get("result", {})
    
    comparison_prompt = f"""请比较以下两次表达尝试，识别改进之处并生成表扬和标签。

第一次尝试（之前的）：
转录：{previous_result.get('transcription', '')[:500]}
各维度评分：
{json.dumps({dim.get('name', ''): dim.get('score', 0) for dim in previous_result.get('dimensions', [])}, ensure_ascii=False, indent=2)}

第二次尝试（当前的）：
转录：{current_result.get('transcription', '')[:500]}
各维度评分：
{json.dumps({dim.get('name', ''): dim.get('score', 0) for dim in current_result.get('dimensions', [])}, ensure_ascii=False, indent=2)}

请分析：
1. 是否有明显的改进（如结构更清晰、例子更具体、重复减少等）
2. 如果有改进，生成一段鼓励性的表扬文字（50-100字）
3. 生成2-3个简短的标签来描述改进点（如「结构更清晰」「例子更具体」「重复减少」）

请以JSON格式输出：
{{
    "has_improvement": true/false,
    "praise": "表扬文字",
    "tags": ["标签1", "标签2", "标签3"]
}}"""

    try:
        comparison_response = await client.post(
            f"{API_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-4-fast",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一位鼓励性的表达教练，擅长识别表达的改进并给予积极的反馈。请以JSON格式输出比较结果。"
                    },
                    {
                        "role": "user",
                        "content": comparison_prompt
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 300
            }
        )
        
        if comparison_response.status_code == 200:
            comparison_data = comparison_response.json()
            comparison_content = comparison_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            
            # Clean up JSON
            clean_content = comparison_content.strip()
            if clean_content.startswith("```json"):
                clean_content = clean_content[7:]
            if clean_content.startswith("```"):
                clean_content = clean_content[3:]
            if clean_content.endswith("```"):
                clean_content = clean_content[:-3]
            clean_content = clean_content.strip()
            
            comparison_result = json.loads(clean_content)
            has_improvement = comparison_result.get("has_improvement", False)
            praise = comparison_result.get("praise", "")
            tags = comparison_result.get("tags", [])
            
            return has_improvement, praise, tags
    except Exception as e:
        print(f"Error comparing attempts: {e}")
        pass
    
    return False, None, []

@app.get("/")
async def serve_landing():
    """Serve the landing page."""
    return FileResponse("landing.html")

@app.get("/app")
async def serve_app():
    """Serve the main application page."""
    return FileResponse("index.html")

@app.post("/api/analyze")
async def analyze_speech(
    request: Request,
    audio_file: UploadFile = File(...),
    topic_id: Optional[str] = Form(None)
):
    """
    Analyze speech recording:
    1. Transcribe audio using the API
    2. Analyze expression quality across four dimensions
    3. If topic_id is provided, compare with previous attempts
    """
    try:
        # Get or create user ID
        user_id = get_or_create_user_id(request)
        
        audio_content = await audio_file.read()
        
        topics = load_topics(user_id)
        previous_attempts = []
        is_retry = False
        
        if topic_id:
            topic = next((t for t in topics if t["id"] == topic_id), None)
            if topic:
                is_retry = True
                previous_attempts = topic.get("attempts", [])
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Step 1: Transcribe
            transcription_response = await client.post(
                f"{API_BASE_URL}/audio/transcriptions",
                headers={"Authorization": f"Bearer {API_KEY}"},
                files={"audio_file": (audio_file.filename or "recording.webm", audio_content, audio_file.content_type or "audio/webm")}
            )
            
            if transcription_response.status_code != 200:
                raise HTTPException(
                    status_code=transcription_response.status_code,
                    detail=f"Transcription failed: {transcription_response.text}"
                )
            
            transcription_data = transcription_response.json()
            raw_transcript = transcription_data.get("text", "")
            
            if not raw_transcript.strip():
                return JSONResponse(content={
                    "success": False,
                    "error": "No speech detected in the recording. Please try again."
                })
            
            # Step 2: Add punctuation
            punctuation_prompt = f"""请为以下中文语音转录文本添加合适的标点符号和断句，使其更易读。保持原意不变，只添加标点符号和适当的换行。

原始文本：
{raw_transcript}

要求：
1. 根据语义和停顿添加适当的标点符号（句号、逗号、问号、感叹号等）
2. 在合适的地方添加换行，使文本更易读
3. 保持原意和用词不变
4. 如果文本本身已经有标点符号，请优化和完善它们

请直接输出格式化后的文本，不要添加任何解释、markdown标记或其他内容。只输出格式化后的文本本身。"""

            punctuation_response = await client.post(
                f"{API_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "grok-4-fast",
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一位专业的文本编辑，擅长为中文语音转录文本添加标点符号和格式化。你的任务是保持原意不变，只添加标点符号和适当的换行，使文本更易读。请只输出格式化后的文本，不要添加任何解释或markdown标记。"
                        },
                        {
                            "role": "user",
                            "content": punctuation_prompt
                        }
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1500
                }
            )
            
            if punctuation_response.status_code != 200:
                transcript = raw_transcript
            else:
                punctuation_data = punctuation_response.json()
                formatted_text = punctuation_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                
                if formatted_text.startswith("```"):
                    lines = formatted_text.split("\n")
                    if len(lines) > 2:
                        formatted_text = "\n".join(lines[1:-1])
                    else:
                        formatted_text = formatted_text.replace("```", "").strip()
                
                formatted_text = formatted_text.strip('"').strip("'").strip()
                
                if formatted_text and len(formatted_text) > len(raw_transcript) * 0.5:
                    transcript = formatted_text
                else:
                    transcript = raw_transcript
            
            # Step 3: Analyze
            analysis_prompt = f"""请分析以下中文语音转录文本的表达质量。根据四个维度进行评估，每个维度给出具体的分析和建议。

转录文本：
"{transcript}"

请按照以下格式输出JSON结果（请确保输出有效的JSON格式）：

{{
    "transcription": "原始转录文本",
    "topic_summary": "话题总结：这段表达主要讲的是什么",
    "overall_summary": {{
        "strengths": "做得好的地方（具体指出1-2个亮点）",
        "main_limitation": "当前最主要的限制点（只点1-2个）"
    }},
    "dimensions": [
        {{
            "name": "表达指向性（自我 vs 他人）",
            "score": 1-10的评分,
            "analysis": "详细分析说明：这段表达主要停留在个人经历，还是已经向外迁移，对他人产生理解或启发？",
            "highlights": ["表现好的地方（如出现从个人经验走向他人的迁移句）"],
            "issues": ["最多1-2个主要问题（如主语长期停留在'我/我的感受'）"],
            "suggestions": ["具体改进建议（如何从自我走向他人）"]
        }},
        {{
            "name": "结构清晰度（核心能力）",
            "score": 1-10的评分,
            "analysis": "详细分析说明：听者是否能顺畅地跟上表达的推进过程？是否有清楚的开头、自然的收束、清晰的衔接？",
            "highlights": ["表现好的地方（如使用了框架、有清晰的逻辑结构）"],
            "issues": ["最多1-2个主要问题（如在同一层级反复绕圈、突然跳转）"],
            "suggestions": ["具体改进建议。如果结构清晰度较弱（评分低于7），在suggestions中添加框架推荐，格式为：'框架建议：如果你想让表达更有逻辑，可以试试[框架名称]（过去-现在-未来/空-雨-伞/3C之一），因为[原因]。' 如果结构清晰度好，则不需要框架推荐。"]
        }},
        {{
            "name": "具体程度（Concrete & Specific）",
            "score": 1-10的评分,
            "analysis": "详细分析说明：抽象观点是否被具体行为、场景或细节支撑？",
            "highlights": ["表现好的地方（如用例子代替评价、出现可视化细节）"],
            "issues": ["最多1-2个主要问题（如停留在'大词'而没有拆解）"],
            "suggestions": ["具体改进建议（如何添加具体细节）"]
        }},
        {{
            "name": "主题与重点（Key Message）",
            "score": 1-10的评分,
            "analysis": "详细分析说明：如果听众只能记住一句话，会记住什么？是否存在清晰的中心观点？",
            "highlights": ["表现好的地方（如存在清晰的中心观点、结尾强化核心信息）"],
            "issues": ["最多1-2个主要问题（如信息分散、没有核心重点）"],
            "suggestions": ["具体改进建议（如何突出主题与重点）"]
        }}
    ]
}}

评估标准：
1. 表达指向性（自我 vs 他人）：主语是否长期停留在"我/我的感受"，是否出现从个人经验走向他人的迁移句，结尾是否指向他人的意义
2. 结构清晰度（核心能力）：是否有清楚的开头，是否在同一层级反复绕圈，是否存在突然跳转，是否有自然的收束，不同部分之间是否有清晰的衔接
3. 具体程度：是否用例子代替评价，是否出现可视化细节（人/时间/行为/场景），是否把"大词"拆解成具体行动
4. 主题与重点：是否存在清晰的中心观点，信息是否围绕同一主题展开，是否在结尾强化核心信息

框架推荐原则（仅用于结构清晰度维度）：
- 如果结构清晰度较弱（评分低于9），在结构清晰度维度的suggestions中添加框架推荐
- 推荐格式：\"框架建议：如果你想让表达更有逻辑，可以试试[框架名称]，因为[原因]。\"
- 框架名称必须是以下之一：\"过去-现在-未来\"、\"空-雨-伞\"、\"3C\"
- 不要指责用户没有使用框架，而是作为建议工具
- 如果结构清晰度好（评分9及以上），不需要框架推荐

请直接输出JSON，不要添加markdown代码块标记。"""

            analysis_response = await client.post(
                f"{API_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "grok-4-fast",
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一位专业的表达能力分析师，擅长分析中文口语表达的质量。你的任务是客观、建设性地评估用户的表达，既要指出问题，也要给予鼓励。请始终以有效的JSON格式输出结果。"
                        },
                        {
                            "role": "user",
                            "content": analysis_prompt
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
            )
            
            if analysis_response.status_code != 200:
                raise HTTPException(
                    status_code=analysis_response.status_code,
                    detail=f"Analysis failed: {analysis_response.text}"
                )
            
            analysis_data = analysis_response.json()
            analysis_content = analysis_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            try:
                clean_content = analysis_content.strip()
                if clean_content.startswith("```json"):
                    clean_content = clean_content[7:]
                if clean_content.startswith("```"):
                    clean_content = clean_content[3:]
                if clean_content.endswith("```"):
                    clean_content = clean_content[:-3]
                clean_content = clean_content.strip()
                
                analysis_result = json.loads(clean_content)
                analysis_result["transcription"] = transcript
                analysis_result["raw_transcription"] = raw_transcript
                
                # Compare with previous attempts if retry
                improvement_praise = None
                attempt_tags = []
                has_improvement = False
                
                if is_retry and previous_attempts:
                    has_improvement, improvement_praise, attempt_tags = await compare_attempts(
                        analysis_result, previous_attempts, client
                    )
                
                # Generate title (only for new topics)
                if not topic_id:
                    title = await generate_title(transcript, client)
                else:
                    topic = next((t for t in topics if t["id"] == topic_id), None)
                    title = topic["title"] if topic else await generate_title(transcript, client)
                
                # Create attempt
                attempt_id = str(uuid.uuid4())
                attempt = {
                    "id": attempt_id,
                    "created_at": datetime.now().isoformat(),
                    "result": analysis_result,
                    "tags": attempt_tags
                }
                
                # Save to topics
                if topic_id:
                    # Add to existing topic
                    topic = next((t for t in topics if t["id"] == topic_id), None)
                    if topic:
                        topic["attempts"].append(attempt)
                        topic["updated_at"] = datetime.now().isoformat()
                        # Update topic tags if there are improvements
                        if attempt_tags:
                            existing_tags = set(topic.get("tags", []))
                            existing_tags.update(attempt_tags)
                            topic["tags"] = list(existing_tags)
                    else:
                        # Topic not found, create new one
                        topic_id = None
                
                if not topic_id:
                    # Create new topic
                    topic_id = str(uuid.uuid4())
                    topic = {
                        "id": topic_id,
                        "title": title,
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                        "attempts": [attempt],
                        "tags": attempt_tags
                    }
                    topics.insert(0, topic)
                else:
                    # Update existing topic in list
                    for i, t in enumerate(topics):
                        if t["id"] == topic_id:
                            topics[i] = topic
                            break
                
                # Keep only last 50 topics
                topics[:] = topics[:50]
                save_topics(user_id, topics)
                
                # Save anonymous data for prompt optimization
                save_anonymous_data(analysis_result)
                
                # Create response with user_id cookie
                response = JSONResponse(content={
                    "success": True,
                    "result": analysis_result,
                    "topic_id": topic_id,
                    "attempt_id": attempt_id,
                    "is_improvement": has_improvement,
                    "improvement_praise": improvement_praise
                })
                # Set cookie to persist user_id (1 year expiration)
                response.set_cookie(key="user_id", value=user_id, max_age=31536000, httponly=True, samesite="lax")
                return response
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                title = await generate_title(transcript, client) if not topic_id else None
                
                attempt_id = str(uuid.uuid4())
                attempt = {
                    "id": attempt_id,
                    "created_at": datetime.now().isoformat(),
                    "result": {
                        "transcription": transcript,
                        "raw_transcription": raw_transcript,
                        "raw_analysis": analysis_content
                    },
                    "tags": []
                }
                
                if topic_id:
                    topic = next((t for t in topics if t["id"] == topic_id), None)
                    if topic:
                        topic["attempts"].append(attempt)
                        topic["updated_at"] = datetime.now().isoformat()
                    else:
                        topic_id = None
                
                if not topic_id:
                    topic_id = str(uuid.uuid4())
                    topic = {
                        "id": topic_id,
                        "title": title or "未命名表达",
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                        "attempts": [attempt],
                        "tags": []
                    }
                    topics.insert(0, topic)
                
                topics[:] = topics[:50]
                save_topics(user_id, topics)
                
                # Save anonymous data (even if analysis failed)
                fallback_result = {
                    "transcription": transcript,
                    "raw_transcription": raw_transcript,
                    "topic_summary": "",
                    "overall_summary": {},
                    "dimensions": []
                }
                save_anonymous_data(fallback_result)
                
                # Create response with user_id cookie
                response = JSONResponse(content={
                    "success": True,
                    "result": {
                        "transcription": transcript,
                        "raw_transcription": raw_transcript,
                        "raw_analysis": analysis_content
                    },
                    "topic_id": topic_id,
                    "attempt_id": attempt_id,
                    "is_improvement": False,
                    "improvement_praise": None
                })
                response.set_cookie(key="user_id", value=user_id, max_age=31536000, httponly=True, samesite="lax")
                return response
                
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Request timed out. Please try again.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history(request: Request):
    """Get all topics with their attempts summary (user-specific)."""
    try:
        user_id = get_or_create_user_id(request)
        topics = load_topics(user_id)
        # Return topics with attempt summaries
        topics_list = []
        for topic in topics:
            topics_list.append({
                "id": topic["id"],
                "title": topic["title"],
                "created_at": topic["created_at"],
                "updated_at": topic.get("updated_at", topic["created_at"]),
                "attempts": [
                    {
                        "id": attempt["id"],
                        "created_at": attempt["created_at"],
                        "tags": attempt.get("tags", [])
                    }
                    for attempt in topic.get("attempts", [])
                ],
                "tags": topic.get("tags", [])
            })
        response = JSONResponse(content={
            "success": True,
            "topics": topics_list
        })
        # Set cookie if not already set
        if not request.cookies.get('user_id'):
            response.set_cookie(key="user_id", value=user_id, max_age=31536000, httponly=True, samesite="lax")
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error loading history: {str(e)}")

@app.get("/api/topics/{topic_id}")
async def get_topic(request: Request, topic_id: str):
    """Get a specific topic with all attempts (user-specific)."""
    user_id = get_or_create_user_id(request)
    topics = load_topics(user_id)
    topic = next((t for t in topics if t["id"] == topic_id), None)
    if topic:
        response = JSONResponse(content={
            "success": True,
            "topic": topic
        })
        if not request.cookies.get('user_id'):
            response.set_cookie(key="user_id", value=user_id, max_age=31536000, httponly=True, samesite="lax")
        return response
    raise HTTPException(status_code=404, detail="Topic not found")

@app.get("/api/topics/{topic_id}/attempts/{attempt_id}")
async def get_attempt(request: Request, topic_id: str, attempt_id: str):
    """Get a specific attempt from a topic (user-specific)."""
    user_id = get_or_create_user_id(request)
    topics = load_topics(user_id)
    topic = next((t for t in topics if t["id"] == topic_id), None)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    
    attempt = next((a for a in topic.get("attempts", []) if a["id"] == attempt_id), None)
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")
    
    response = JSONResponse(content={
        "success": True,
        "topic": {
            "id": topic["id"],
            "title": topic["title"],
            "created_at": topic["created_at"]
        },
        "result": attempt["result"]
    })
    if not request.cookies.get('user_id'):
        response.set_cookie(key="user_id", value=user_id, max_age=31536000, httponly=True, samesite="lax")
    return response

# Prompt 管理和对比测试 API
if PROMPT_OPTIMIZATION_ENABLED:
    @app.get("/api/prompts")
    async def get_prompts(active_only: bool = True):
        """获取所有 prompt 变体"""
        prompts = get_all_prompts(active_only=active_only)
        return JSONResponse(content={
            "success": True,
            "prompts": prompts
        })

    @app.post("/api/prompts")
    async def create_prompt(
        name: str = Form(...),
        prompt_template: str = Form(...),
        system_message: Optional[str] = Form(None),
        temperature: float = Form(0.7),
        max_tokens: int = Form(2000),
        description: str = Form(""),
        tags: Optional[str] = Form(None)
    ):
        """创建新的 prompt 变体"""
        tag_list = tags.split(",") if tags else []
        variant = create_prompt_variant(
            name=name,
            prompt_template=prompt_template,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            description=description,
            tags=tag_list
        )
        return JSONResponse(content={
            "success": True,
            "prompt": variant
        })

    @app.get("/api/prompts/{prompt_id}")
    async def get_prompt(prompt_id: str):
        """获取指定的 prompt 变体"""
        prompt = get_prompt_variant(prompt_id)
        if prompt:
            return JSONResponse(content={
                "success": True,
                "prompt": prompt
            })
        raise HTTPException(status_code=404, detail="Prompt not found")

    @app.put("/api/prompts/{prompt_id}")
    async def update_prompt(prompt_id: str, data: dict):
        """更新 prompt 变体"""
        prompt = update_prompt_variant(prompt_id, **data)
        if prompt:
            return JSONResponse(content={
                "success": True,
                "prompt": prompt
            })
        raise HTTPException(status_code=404, detail="Prompt not found")

    @app.delete("/api/prompts/{prompt_id}")
    async def delete_prompt(prompt_id: str):
        """删除 prompt 变体"""
        success = delete_prompt_variant(prompt_id)
        if success:
            return JSONResponse(content={"success": True})
        raise HTTPException(status_code=404, detail="Prompt not found")

    @app.get("/api/test-data/topics")
    async def get_topics():
        """获取所有话题"""
        topics = get_all_topics()
        return JSONResponse(content={
            "success": True,
            "topics": topics
        })

    @app.get("/api/test-data")
    async def get_test_data(
        topic: Optional[str] = None,
        category: Optional[str] = None,
        source: Optional[str] = None
    ):
        """获取测试数据"""
        cases = get_test_cases(topic=topic, category=category, source=source)
        return JSONResponse(content={
            "success": True,
            "cases": cases,
            "count": len(cases)
        })

    @app.post("/api/test-data")
    async def add_test_data(data: dict):
        """添加测试数据"""
        case = add_test_case(
            transcript=data.get("transcript", ""),
            topic=data.get("topic", ""),
            category=data.get("category"),
            expected_features=data.get("expected_features", []),
            source=data.get("source", "manual"),
            notes=data.get("notes", "")
        )
        return JSONResponse(content={
            "success": True,
            "case": case
        })

    @app.get("/api/test-data/statistics")
    async def get_test_statistics():
        """获取测试数据统计"""
        stats = get_statistics()
        return JSONResponse(content={
            "success": True,
            "statistics": stats
        })

    @app.post("/api/comparison/run")
    async def run_comparison(data: dict):
        """运行 prompt 对比测试"""
        try:
            prompt_ids = data.get("prompt_ids", [])
            topic = data.get("topic")
            category = data.get("category")
            max_cases = data.get("max_cases")
            
            if not prompt_ids or len(prompt_ids) < 2:
                raise HTTPException(status_code=400, detail="至少需要2个 prompt 变体")
            
            comparison = await compare_prompts(
                prompt_ids=prompt_ids,
                topic=topic,
                category=category,
                max_cases=max_cases
            )
            
            return JSONResponse(content={
                "success": True,
                "comparison": comparison
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/comparison/latest")
    async def get_latest_comparison():
        """获取最新的对比结果"""
        comparison = load_latest_comparison()
        if comparison:
            return JSONResponse(content={
                "success": True,
                "comparison": comparison
            })
        return JSONResponse(content={
            "success": False,
            "comparison": None
        })

    @app.get("/dashboard")
    async def serve_dashboard():
        """Serve the dashboard HTML page."""
        return FileResponse("dashboard.html")

@app.post("/api/generate-framework-example")
async def generate_framework_example(framework_name: str = Form(...), transcript: str = Form(...)):
    """Generate a framework example based on user's transcript."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        framework_prompt = f"""请基于以下用户表达内容，使用{framework_name}框架生成一个改进示例。

用户原始表达：
{transcript}

要求：
1. 保持用户的核心观点和意图不变
2. 使用{framework_name}框架重新组织表达
3. 使表达更清晰、更有逻辑
4. 用自然的中文表达，适当分段
5. 只输出改进后的表达内容，不要添加解释或说明

请直接输出改进后的表达内容。"""

        try:
            response = await client.post(
                f"{API_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "grok-4-fast",
                    "messages": [
                        {
                            "role": "system",
                            "content": f"你是一位表达教练，擅长使用{framework_name}框架改进表达。你的任务是保持用户原意，使用框架重新组织表达，使其更清晰。"
                        },
                        {
                            "role": "user",
                            "content": framework_prompt
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                example = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                
                # Clean up markdown if present
                if example.startswith("```"):
                    lines = example.split("\n")
                    if len(lines) > 2:
                        example = "\n".join(lines[1:-1])
                    else:
                        example = example.replace("```", "").strip()
                
                example = example.strip('"').strip("'").strip()
                
                return JSONResponse(content={
                    "success": True,
                    "example": example
                })
            else:
                raise HTTPException(status_code=response.status_code, detail="Failed to generate example")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating example: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
