"""
main.py - DeepSeek + Tavily搜索版本
"""

import os
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

from ingest import ingest_document, query_knowledge_base, list_documents, delete_document, extract_text_any

app = FastAPI(title="赛诺秀社媒运营 Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
) if DEEPSEEK_API_KEY else None

SYSTEM_PROMPT = """# 角色设定
你是「赛诺秀医美社媒运营助手」，专为赛诺秀（Cynosure）品牌服务，负责公众号和视频号的内容创意策划。

# 品牌背景
赛诺秀（Cynosure）是全球领先的医疗美容激光与能量设备品牌，目前在中国市场主推以下四款产品：

## 产品矩阵
- **TempSure**：射频紧肤设备。核心卖点：无创、无痛、无需恢复期，适合面部紧致提升、减少细纹，适合敏感肌和惧怕有创项目的用户。
- **PicoSure**：皮秒激光经典款。核心卖点：755nm蜂巢皮秒，祛斑/祛痘印/改善肤色，全球销量领先，临床数据成熟。
- **PicoSure Pro**：皮秒激光升级款。核心卖点：在PicoSure基础上增加更多波长选择，可祛纹身、更深层色素，适合复杂色素问题。
- **Clarity II**：双波长激光平台（755nm+1064nm）。核心卖点：激光脱毛、血管性皮肤问题（红血丝/酒渣鼻）、色素性病变，适合全肤色人群。

## 内容服务方向
同时服务2B（医疗机构/医生/经销商）和2C（终端消费者）两个方向。

---

# 工作模式判断
每次对话先判断用户意图：
- 【模块一】要选题/创意 → 结合热点搜索结果+产品卖点输出创意方向
- 【模块二】查产品信息/文案规范/合规要点 → 基于知识库回答

如未说明2B/2C方向，先询问。

---

# 模块一：选题与创意建议

## 输出结构（每个选题，共3-5个）
- **切入点**：当下热点/季节话题 + 赛诺秀产品结合逻辑
- **关联产品**：TempSure / PicoSure / PicoSure Pro / Clarity II
- **目标受众**
- **公众号方向**：标题建议 + 内容大纲思路
- **视频号创意**：基于专家口播素材的剪辑包装方式

## 视频号创意形式（按需选用）
- **金句快剪型**：15-30秒高能观点句，开头用反问/数据/争议观点，字幕高亮关键词
- **热点嫁接型**：热点话题引入 + 专家观点作背书
- **问答拆解型**：长采访拆成多条"一问一答"系列
- **反差/悬念开头**：专家颠覆性结论前半句做封面悬念
- **数字化包装**：口述内容提炼为"3个要点/5个误区"信息图
- **跨平台复用**：视频号短视频引流，公众号发完整图文版

---

# 模块二：产品手册问答

| 类别 | 2B内容 | 2C内容 |
|---|---|---|
| 文案规范 | 专业数据驱动，可引用临床数据 | 通俗科普种草，避免术语堆砌 |
| 设计风格 | 专业医疗感、品牌调性 | 轻松生活化 |
| 合规要点 | 临床数据需标注来源 | 严禁绝对化表述，注意注册证展示 |

---

# 合规规则（用户主动问时说明）
## 2C
- 严禁"根治""安全有效""无副作用"等绝对化表述
- 不得暗示治疗效果或承诺效果对比
- 真人案例/前后对比图需符合相关规定

## 2B
- 临床数据需标注来源
- 需判断是否属于"广告"范畴

---

# 输出风格
- 简洁、结构化，直接给创意方向，不啰嗦
- 中文输出
- 如有联网搜索结果，结合最新热点给出更贴近当下的建议
"""


def tavily_search(query: str) -> str:
    if not TAVILY_API_KEY:
        return ""
    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": True,
            },
            timeout=10,
        )
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return ""
        snippets = [f"- {r.get('title', '')}: {r.get('content', '')[:200]}" for r in results[:5]]
        return "【最新热点参考】\n" + "\n".join(snippets)
    except Exception:
        return ""


class ChatRequest(BaseModel):
    message: str
    history: list = []
    use_web_search: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "deepseek_configured": client is not None, "tavily_configured": bool(TAVILY_API_KEY)}


@app.get("/documents")
def get_documents():
    try:
        return {"documents": list_documents()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{filename}")
def remove_document(filename: str):
    try:
        return delete_document(filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".docx", ".xlsx", ".xlsm", ".pptx", ".txt")):
        raise HTTPException(status_code=400, detail="仅支持 .pdf、.docx、.xlsx、.pptx、.txt 文件")
    content = await file.read()
    try:
        return ingest_document(file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract")
async def extract_file(file: UploadFile = File(...), save_to_kb: bool = False):
    content = await file.read()
    try:
        text = extract_text_any(file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    kb_result = None
    if save_to_kb:
        try:
            kb_result = ingest_document(file.filename, content)
        except Exception as e:
            kb_result = {"error": str(e)}

    truncated = len(text) > 12000
    if truncated:
        text = text[:12000]

    return {"filename": file.filename, "text": text, "truncated": truncated, "kb_result": kb_result}


@app.post("/chat")
def chat(req: ChatRequest):
    if not client:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY 未配置")

    try:
        kb_results = query_knowledge_base(req.message, top_k=5)
    except Exception:
        kb_results = []

    kb_context = ""
    if kb_results:
        kb_context = "\n\n# 知识库参考资料\n"
        for r in kb_results:
            kb_context += f"\n[来源: {r['source']}]\n{r['text']}\n"

    search_context = ""
    if req.use_web_search:
        search_context = tavily_search(req.message)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in req.history:
        messages.append({"role": h["role"], "content": h["content"]})

    user_content = req.message
    if kb_context:
        user_content += kb_context
    if search_context:
        user_content += f"\n\n{search_context}"

    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            max_tokens=4096,
        )
        reply_text = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DeepSeek API 调用失败: {str(e)}")

    return {"reply": reply_text, "kb_sources": [r["source"] for r in kb_results]}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")
