import asyncio
import base64
import os

import aiohttp
import discord

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-3.5-flash:generateContent?key={GEMINI_API_KEY}"
)
SYSTEM_PROMPT = (
    "당신은 친절하고 유능한 공부 도우미입니다. "
    "학생이 올린 문제, 지문, 개념 등을 분석하여 단계별로 친절하게 한국어로 설명해주세요. "
    "풀이 과정은 명확하게 구조화하고, 핵심 개념은 쉽게 풀어서 설명해주세요."
)
SKIP_PREFIXES = ("분석 중", "생각 중")

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# 봇이 생성한 스레드 ID 추적 (메모리)
bot_thread_ids: set[int] = set()


@client.event
async def on_ready():
    print(f"Logged in as {client.user} (id={client.user.id})")


@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # 봇이 만든 스레드 안에서 후속 질문 처리
    if isinstance(message.channel, discord.Thread):
        if message.channel.id in bot_thread_ids:
            await handle_followup(message)
        return

    # 일반 채널에 이미지가 올라왔을 때 처리
    images = [
        a for a in message.attachments
        if a.content_type and a.content_type.startswith("image/")
    ]
    if images:
        await handle_new_image(message, images)


async def handle_new_image(message: discord.Message, images: list[discord.Attachment]):
    thread_name = f"📚 {message.author.display_name}의 질문"
    thread = await message.create_thread(name=thread_name[:100])
    bot_thread_ids.add(thread.id)

    loading = await thread.send("분석 중...")
    try:
        async with aiohttp.ClientSession() as session:
            parts: list[dict] = []
            for att in images:
                mime, data = await fetch_b64(session, att.url)
                parts.append({"inline_data": {"mime_type": mime, "data": data}})
            if message.content:
                parts.append({"text": message.content})
            else:
                parts.append({"text": "이미지를 분석하고 내용을 설명해주세요."})

            reply = await call_gemini(session, [{"role": "user", "parts": parts}])
    except Exception as e:
        reply = f"오류가 발생했습니다: {e}"

    await loading.delete()
    await send_chunks(thread, reply)


async def handle_followup(message: discord.Message):
    thread = message.channel
    loading = await thread.send("생각 중...")

    try:
        async with aiohttp.ClientSession() as session:
            # 스레드 히스토리 수집 (로딩 메시지·현재 메시지 제외)
            history_msgs: list[discord.Message] = []
            async for msg in thread.history(limit=200, oldest_first=True):
                if msg.id in (loading.id, message.id):
                    continue
                if msg.author.bot and any(msg.content.startswith(p) for p in SKIP_PREFIXES):
                    continue
                history_msgs.append(msg)

            # 원본 이미지 메시지(스레드 시작 메시지) 가져오기
            starter: discord.Message | None = None
            if thread.parent:
                try:
                    starter = await thread.parent.fetch_message(thread.id)
                except discord.NotFound:
                    pass

            # 시간 순 정렬
            ordered: list[tuple[str, discord.Message]] = []
            if starter:
                ordered.append(("user", starter))
            for msg in history_msgs:
                role = "model" if msg.author.bot else "user"
                ordered.append((role, msg))
            ordered.append(("user", message))

            # 이미지 일괄 다운로드
            img_cache: dict[int, tuple[str, str]] = {}
            for _, msg in ordered:
                for att in msg.attachments:
                    if att.content_type and att.content_type.startswith("image/"):
                        if att.id not in img_cache:
                            img_cache[att.id] = await fetch_b64(session, att.url)

            # Gemini contents 배열 구성
            raw: list[dict] = []
            for role, msg in ordered:
                parts: list[dict] = []
                for att in msg.attachments:
                    if att.content_type and att.content_type.startswith("image/") and att.id in img_cache:
                        mime, data = img_cache[att.id]
                        parts.append({"inline_data": {"mime_type": mime, "data": data}})
                if msg.content:
                    parts.append({"text": msg.content})
                if parts:
                    raw.append({"role": role, "parts": parts})

            # 연속된 동일 role 병합 (Gemini 요구사항)
            contents: list[dict] = []
            for entry in raw:
                if contents and contents[-1]["role"] == entry["role"]:
                    contents[-1]["parts"].extend(entry["parts"])
                else:
                    contents.append({"role": entry["role"], "parts": list(entry["parts"])})

            # 첫 메시지는 반드시 user여야 함
            if contents and contents[0]["role"] == "model":
                contents.insert(0, {"role": "user", "parts": [{"text": "이미지를 분석해주세요."}]})

            reply = await call_gemini(session, contents)
    except Exception as e:
        reply = f"오류가 발생했습니다: {e}"

    await loading.delete()
    await send_chunks(thread, reply)


async def fetch_b64(session: aiohttp.ClientSession, url: str) -> tuple[str, str]:
    async with session.get(url) as resp:
        mime = resp.headers.get("Content-Type", "image/png").split(";")[0].strip()
        raw = await resp.read()
    return mime, base64.b64encode(raw).decode()


async def call_gemini(session: aiohttp.ClientSession, contents: list[dict]) -> str:
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
    }
    async with session.post(GEMINI_URL, json=payload) as resp:
        data = await resp.json()
    if "candidates" not in data:
        error = data.get("error", {}).get("message", str(data))
        return f"Gemini API 오류: {error}"
    return data["candidates"][0]["content"]["parts"][0]["text"]


async def send_chunks(dest: discord.abc.Messageable, text: str):
    for i in range(0, len(text), 1900):
        await dest.send(text[i : i + 1900])


client.run(DISCORD_TOKEN)
