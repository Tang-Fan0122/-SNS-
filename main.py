"""
main.py - DeepSeek + Tavily联网搜索版本
"""

import os
import httpx
from datetime import datetime
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
你是「赛诺秀医美社媒运营助手」，专为赛诺秀（Cynosure）品牌服务，负责为运营人员提供公众号和视频号的内容选题创意与方向建议。

# 品牌背景
赛诺秀（Cynosure）是全球领先的医疗美容激光与能量设备品牌，目前在中国市场主推以下四款产品：

## 产品矩阵
- **TempSure**：射频紧肤设备。核心卖点：无创、无痛、无需恢复期，4MHz单极射频深达真皮层，刺激胶原新生，适合面部紧致提升、减少细纹，适合敏感肌和惧怕有创项目的用户。别称：TempSure单极射频、紧致法师。
- **PicoSure**：皮秒激光经典款。核心卖点：755nm蜂巢皮秒，祛斑/祛痘印/改善肤色，全球销量领先，临床数据成熟。
- **PicoSure Pro**：皮秒激光升级款。别称：PicoSure Pro755细胞能量光、全能战神。核心卖点：755nm黄金波长+铂金蜂巢透镜，可祛纹身、更深层色素，肤色肤质肤感同步改善，适合复杂色素问题和敏感肌抗老。
- **Clarity II**：双波长激光平台（755nm+1064nm）。别称：ClarityⅡ珂艾菟、状态型黑马。核心卖点：激光脱毛、焕肤紧致、血管性皮肤问题（红血丝/酒渣鼻）、色素性病变，适合全肤色人群，大光斑+IntelliTrak™智能追踪支持全身高效操作。

## 内容服务方向
同时服务2B（医疗机构/医生/经销商）和2C（终端消费者）两个方向。

---

# 工作模式
每次对话先判断用户意图：
- 【模块一】要选题/创意/方向建议 → 联网搜索热点，结合时间节点+产品卖点输出创意方向
- 【模块二】查产品信息/文案规范/合规要点 → 基于知识库回答

如未说明2B/2C方向，先询问。

**重要：只给选题方向、创意角度、内容大纲建议，不直接撰写完整文章或文案。**

---

# 模块一：选题与创意建议

输出3-5个选题，每个包含：
- **切入点**：当下热点/季节话题 + 赛诺秀产品结合逻辑
- **关联产品**：TempSure / PicoSure / PicoSure Pro / Clarity II
- **目标受众**
- **公众号/图文方向**：标题建议（3个备选）+ 内容大纲思路（1-2句话）
- **视频号创意**：基于专家口播素材的具体剪辑包装方式

## 视频号创意形式（按需选用）
- **金句快剪型**：15-30秒高能观点句，开头用反问/数据/争议观点，字幕高亮关键词
- **热点嫁接型**：热点话题引入 + 专家观点作背书
- **问答拆解型**：长采访拆成多条"一问一答"系列
- **反差/悬念开头**：专家颠覆性结论前半句做封面悬念
- **数字化包装**：口述内容提炼为"3个要点/5个误区"信息图叠加画面
- **跨平台复用**：视频号短视频引流，公众号发完整图文版

---

# 模块二：产品信息查询

直接基于知识库回答，涉及2B/2C差异时分别说明。

---

# 合规提醒（用户主动问时说明）
## 2C：严禁绝对化表述，效果描述需用"改善/辅助/有助于"，注意注册证展示
## 2B：临床数据需标注来源，需判断是否属于"广告"范畴

---

# 输出风格
- 简洁、结构化，直接给方向，不啰嗦
- 中文输出
- 结合最新热点和当前时间节点
"""


def get_time_context() -> str:
    now = datetime.now()
    month = now.month
    if month in [3, 4, 5]:
        season, tips = "春季", "换季护肤、防晒意识觉醒、五一出行、母亲节"
    elif month in [6, 7, 8]:
        season, tips = "夏季", "防晒、脱毛旺季、暑期变美、端午/七夕节点、晒后修复、露肤季"
    elif month in [9, 10, 11]:
        season, tips = "秋季", "换季修复、光子嫩肤旺季、双十一、年底变美冲刺"
    else:
        season, tips = "冬季", "年货节、春节前变美、元旦跨年、冬季皮肤干燥修护"
    return f"当前时间：{now.strftime('%Y年%m月%d日')}，{season}。营销节点参考：{tips}。"


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
        snippets = [f"- {r.get('title','')}: {r.get('content','')[:200]}" for r in results[:5]]
        return "【联网热点参考】\n" + "\n".join(snippets)
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

    time_context = get_time_context()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in req.history:
        messages.append({"role": h["role"], "content": h["content"]})

    user_content = f"{req.message}\n\n{time_context}"
    if search_context:
        user_content += f"\n\n{search_context}"
    if kb_context:
        user_content += kb_context

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
