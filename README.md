# Chào mừng các bạn đến với Giai đoạn 2, Track 3, Day 17: Memory Systems for AI Agent

Trong Day 17 này, các bạn sẽ tập trung vào một câu hỏi rất thực tế: làm sao để AI agent **không chỉ trả lời tốt trong một lượt chat**, mà còn **nhớ đúng thông tin quan trọng qua nhiều phiên làm việc** mà vẫn kiểm soát được chi phí token.

Trong bài lab này, các bạn sẽ xây dựng và so sánh hai agent:

- `Baseline Agent`: chỉ có short-term memory trong cùng một thread
- `Advanced Agent`: có short-term memory, `User.md` bền vững, và compact memory để nén hội thoại dài

Mục tiêu cuối cùng không phải chỉ là “agent nhớ nhiều hơn”, mà là hiểu rõ trade-off giữa:

- độ nhớ dài hạn
- chất lượng phản hồi
- chi phí token
- độ phức tạp của hệ thống memory

## Các bạn sẽ làm gì trong track này?

Sau khi hoàn thành, các bạn cần có khả năng:

- phân biệt `short-term memory`, `persistent memory`, và `compact memory`
- xây dựng agent baseline và advanced trên cùng một benchmark
- lưu hồ sơ người dùng bằng `User.md`
- kích hoạt compact memory khi hội thoại dài vượt ngưỡng
- benchmark hai agent bằng cùng một bộ dữ liệu tiếng Việt
- đọc kết quả benchmark theo các chỉ số recall, token, memory growth, chất lượng phản hồi

## Cấu trúc codebase

Repo này được chia thành ba phần rõ ràng:

- `src/`: bản scaffold dành cho sinh viên, chứa pseudocode và TODO để hoàn thiện
- `data/`: dữ liệu benchmark ở root để dùng cho cả benchmark chuẩn và stress benchmark

## Provider hỗ trợ

Trong bản solved lab, runtime hỗ trợ các provider sau:

- `openai`
- `custom` (OpenAI-compatible base URL)
- `gemini`
- `anthropic`
- `ollama`
- `openrouter`

Điều này quan trọng vì memory system không nên bị khóa vào một provider duy nhất.

## Chỉ số benchmark cần hiểu

Khi hoàn thiện bài, benchmark nên cho các cột sau:

- `Agent tokens only`: token sinh ra trực tiếp trong hội thoại của agent
- `Prompt tokens processed`: lượng ngữ cảnh agent phải kéo theo qua các lượt
- `Cross-session recall`: khả năng nhớ facts qua thread hoặc session mới
- `Response quality`: chất lượng phản hồi
- `Memory growth (bytes)`: tốc độ phình của file memory
- `Compactions`: số lần compact memory đã nén lịch sử cũ

Điểm quan trọng nhất của track này là:

- ở hội thoại ngắn, `Advanced` có thể tốn hơn `Baseline` về token usage
- ở hội thoại rất dài, compact memory nên giúp `Advanced` xử lý ngữ cảnh hiệu quả hơn đáng kể + tiết kiệm usage.

## Cách dùng repo này

## Setup môi trường

Các bạn cần chuẩn bị môi trường Python `>= 3.11` và cài các package cần thiết cho LangChain, LangGraph, provider SDK, `python-dotenv`, `tabulate`, và `pytest`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install langchain langgraph langchain-openai langchain-google-genai langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest
```

Sau đó làm việc trực tiếp với `src/` và `data/` ở root repo.

Nếu các bạn là sinh viên:

- làm bài trong `src/`
- dùng `data/` làm benchmark input

Nếu các bạn là giảng viên hoặc reviewer:

- dùng `src/` để đánh giá scaffold giao cho sinh viên và kết quả hoàn thiện cuối cùng

## Tài liệu nên đọc tiếp

- `Guide.md`: hướng dẫn từng bước để hoàn thành lab
- `Rubric.md`: tiêu chí chấm điểm và bonus

Track này được thiết kế để các bạn không chỉ “dùng agent”, mà còn bắt đầu nghĩ như một người thiết kế **memory system** cho agent production.

## Chạy benchmark — có cần API không?

Benchmark và pytest mặc định dùng `AGENT_MODE=offline` (không cần API).

| `AGENT_MODE` | Hành vi |
|---|---|
| `offline` | Deterministic, không gọi LLM (mặc định) |
| `live` | Gọi Gemini qua LangChain |
| `auto` | Live nếu có API key, không thì offline |

Demo chat live:

```powershell
cd src
$env:AGENT_MODE='live'
..\.venv\Scripts\python.exe chat_live.py
```

### Cấu hình Gemini (`gemini-3.1-flash-lite`)

1. Vào [Google AI Studio → API Keys](https://aistudio.google.com/apikey) và tạo key.
2. Ở **root repo** (cùng cấp với `Guide.md`), tạo file `.env`:

```powershell
Copy-Item .env.example .env
```

3. Mở `.env`, dán key vào:

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.1-flash-lite
GEMINI_API_KEY=AIza...your_key...
```

4. Kiểm tra kết nối:

```powershell
cd src
..\.venv\Scripts\python.exe -c "from pathlib import Path; from config import load_config; from model_provider import build_chat_model; c=load_config(Path('..').resolve()); m=build_chat_model(c.model); print(m.invoke('Xin chao').content)"
```

**Lưu ý:** File `.env` đã nằm trong `.gitignore` — không commit key lên git.

Xem thêm phân tích trade-off và bonus features trong `ANALYSIS.md`.

## Dashboard trực quan (UI demo)

```powershell
cd src
..\.venv\Scripts\streamlit.exe run demo_ui.py
```

Mở trình duyệt tại `http://localhost:8501`. UI gồm 4 tab:

1. **Tổng quan** — sơ đồ Baseline vs Advanced
2. **So sánh Chat** — gửi tin nhắn, xem User.md / compact realtime
3. **Playback Data** — từng lượt trong `conversations.json` / stress JSON
4. **Benchmark** — chạy và xem report
