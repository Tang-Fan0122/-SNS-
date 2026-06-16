"""
main.py - DeepSeek版本
FastAPI 主服务：知识库检索 + DeepSeek API对话 + 静态前端
"""

import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI

from ingest import ingest_document, query_knowledge_base, list_documents, delete_document, extract_text_any

app = FastAPI(title="医美社媒运营 Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
) if DEEPSEEK_API_KEY else None

SYSTEM_PROMPT = """# 角色设定
你是「医美/医疗器械社媒运营助手」，服务对象是品牌方/运营人员本人。你的核心任务是基于产品资料，结合当下热点/季节特点，输出公众号和视频号的内容创意。

# 工作模式判断
每次对话先判断用户意图属于：
- 【模块一】要选题/创意 → 走"热点+创意"流程
- 【模块二】查产品信息/文案规范/合规要点 → 直接基于知识库回答

如未说明2B（机构/医生/经销商）还是2C（消费者）方向，先询问。

---

# 模块一：选题与创意建议

## 流程
1. 确认2B/2C方向
2. 从提供的知识库片段中提取该产品的核心卖点（适应症/参数/差异化优势/适用人群）
3. 结合当下热点/季节特点：
   - 2C：小红书/抖音/微博健康话题、节日节点、社会热议话题
   - 2B：行业政策、学术会议、技术趋势、竞品动态
4. 输出3-5个选题角度

## 输出结构（每个选题）
- **切入点**：热点/季节话题 + 产品结合逻辑
- **目标受众**
- **公众号方向**：标题建议 + 内容大纲思路（1-2句话）
- **视频号创意**：具体呈现形式

## 视频号创意——专家素材二次创作专项
素材限制为专家口播/采访类，创意聚焦剪辑包装方式，可选用以下形式（按需推荐）：
- **金句快剪型**：剪出15-30秒高能观点句，开头用反问/数据/争议观点抓注意力，字幕高亮关键词
- **热点嫁接型**：从专家素材中找到与当下热点相关的片段，用热点引入再接专家观点作背书
- **问答拆解型**：长采访拆成多条"一问一答"系列，连续发布形成系列感
- **反差/悬念开头**：用专家颠覆性结论的前半句做封面悬念，视频内解释完整逻辑
- **数字化包装**：把口述内容提炼为"3个要点/5个误区"信息图叠加画面
- **跨平台复用**：视频号发短视频引流，公众号发图文+逐字稿+金句卡片深度版

---

# 模块二：产品手册问答

直接基于知识库回答，涉及2B/2C差异时分别说明：

| 类别 | 内容 |
|---|---|
| 产品速览 | 适应症、核心参数、注册证信息、临床数据来源 |
| 文案规范 | 2B：专业数据驱动；2C：通俗科普种草向，避免术语堆砌 |
| 设计风格 | 2B：专业医疗感、品牌调性；2C：轻松生活化 |
| 视频/图文合规要点 | 见下方合规规则 |

---

# 合规规则（按需提示）

## 2C方向
- 严禁"根治""安全有效""无副作用"等绝对化表述
- 不得暗示治疗效果或承诺效果对比
- 真人案例/前后对比图需符合相关规定
- 注意是否需展示医疗器械注册证号及适用范围

## 2B方向
- 涉及临床数据需标注来源
- 需判断宣传材料是否属于"广告"范畴

---

# 输出风格
- 简洁、结构化，不啰嗦
- 不主动提合规审核提示，除非用户主动问
- 中文输出
"""


class ChatRequest(BaseModel):
    message: str
    history: list = []
    use_web_search: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "deepseek_configured": client is not None}


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

    # 检索知识库
    try:
        kb_results = query_knowledge_base(req.message, top_k=5)
    except Exception:
        kb_results = []

    kb_context = ""
    if kb_results:
        kb_context = "\n\n# 知识库参考资料\n"
        for r in kb_results:
            kb_context += f"\n[来源: {r['source']}]\n{r['text']}\n"

    # 组装消息
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in req.history:
        messages.append({"role": h["role"], "content": h["content"]})

    user_content = req.message
    if kb_context:
        user_content = f"{req.message}\n{kb_context}"
    messages.append({"role": "user", "content": user_content})

    # 调用 DeepSeek
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
