# Phân tích kết quả — Day 17 Memory Systems

Tài liệu này tóm tắt trade-off của hệ thống memory và giải thích các mở rộng bonus hướng tới Rubric 90–100.

## 1. Câu chuyện benchmark

| Giai đoạn | Baseline | Advanced |
|---|---|---|
| Hội thoại ngắn (Standard) | Recall ≈ 0, prompt thấp hơn một chút | Recall cao nhờ `User.md`, prompt cao hơn vì luôn nạp profile |
| Hội thoại dài (Stress) | Prompt tokens tăng gần tuyến tính theo lịch sử | Compact nén message cũ → prompt thấp hơn rõ rệt, vẫn recall tốt |

Luồng logic reviewer mong đợi:

1. Baseline không nhớ dài hạn qua thread mới.
2. Advanced thêm `User.md` nên cross-session recall tăng mạnh.
3. Thread dài làm prompt cost của Baseline bùng nổ.
4. Compact memory giúp Advanced giữ prompt cost ở mức kiểm soát được.
5. Hệ thống mạnh hơn nhưng cần guardrail (confidence, conflict, decay).

## 2. Vì sao compact không thắng ở hội thoại ngắn?

Compact chỉ kích hoạt khi token vượt `COMPACT_THRESHOLD_TOKENS`. Ở Standard Benchmark, hội thoại chưa đủ dài nên **compactions = 0**.

Advanced vẫn tốn hơn Baseline vì mỗi lượt phải mang theo toàn bộ `User.md` (persistent memory). Đây là chi phí cố định để đổi lấy recall.

Compact chủ yếu tối ưu **`Prompt tokens processed`**, không phải `Agent tokens only` (token output của câu trả lời).

## 3. Bonus features

### 3.1 Confidence threshold

**Vấn đề:** Regex extraction dễ ghi nhầm fact từ câu hỏi hoặc nhiễu trong stress dataset.

**Giải pháp:** `extract_profile_candidates()` gán `confidence` cho từng fact. Chỉ ghi vào `User.md` khi `confidence >= PROFILE_CONFIDENCE_THRESHOLD` (mặc định `0.75`).

| Nguồn | Confidence ví dụ |
|---|---|
| Correction ("đính chính", "giờ chuyển sang") | 0.95–0.98 |
| Fact explicit ("tên là", "đồ uống yêu thích") | 0.90–0.94 |
| Pattern generic ("mình ở", "đang làm") | 0.80–0.86 |
| Priority / interests mơ hồ | 0.76–0.80 |
| Câu hỏi thuần | 0 (bỏ qua) |

**Cải thiện:** Giảm lưu sai → recall ổn định hơn, `User.md` gọn hơn.

**Rủi ro:** Ngưỡng quá cao có thể bỏ sót fact hợp lệ but mơ hồ.

### 3.2 Conflict handling

**Vấn đề:** User đính chính (backend → MLOps, Đà Nẵng → Huế) hoặc nhắc fact cũ trong câu phủ định ("đừng nói backend engineer nữa").

**Giải pháp:**

- Fact correction có `is_correction=True` và confidence cao → luôn ghi đè.
- `upsert_fact()` từ chối update nếu confidence mới **thấp hơn** fact đang có (trừ correction).
- Bỏ qua generic profession extraction khi message chứa "đừng nói", "thông tin cũ", "câu đùa".

**Cải thiện:** Recall phản ánh thông tin **mới nhất**, không giữ đồng thời fact mâu thuẫn.

**Rủi ro:** Regex correction chưa cover mọi cách diễn đạt tự nhiên.

### 3.3 Memory decay

**Vấn đề:** `User.md` phình to theo thời gian; fact ít quan trọng vẫn chiếm prompt.

**Giải pháp:** Mỗi fact lưu metadata `<!-- c=...;t=...;corr=... -->`. Khi recall, `get_recall_facts()` tính:

```
effective_conf = confidence * (0.5 ** (age / half_life))
```

Fact correction (`corr=1`) và các **core keys** (`name`, `location`, `profession`, …) **không decay**. Decay chỉ áp dụng cho fact phụ như `interests`, `priority`.

**Cải thiện:** Giảm nhiễu prompt từ fact cũ / ít quan trọng.

**Rủi ro:** Half-life quá ngắn có thể làm mất fact hợp lệ nếu user lâu không nhắc lại.

## 4. Ba lớp memory

| Lớp | Vị trí | Vai trò |
|---|---|---|
| Short-term | `CompactMemoryManager.messages` | Message gần nhất trong thread |
| Persistent | `state/profiles/{user_id}.md` | Fact ổn định cross-session |
| Compact | `CompactMemoryManager.summary` | Nén lịch sử cũ khi thread dài |

## 5. Có cần API key không?

**Không bắt buộc** cho benchmark và pytest.

- `benchmark.py` và `test_agents.py` dùng `force_offline=True` → logic deterministic, không gọi LLM.
- Chỉ cần Python venv + dependencies đã cài.

**Khi nào cần API?**

- `AGENT_MODE=offline` (mặc định): benchmark / pytest — **không cần API**
- `AGENT_MODE=live`: gọi Gemini qua `live_runtime.invoke_live_chat`
- `AGENT_MODE=auto`: live nếu có key, không thì fallback offline

Live fail (key sai, mạng lỗi) → tự động fallback offline, field `mode` trong response.

## 6. Lệnh chạy

```powershell
cd src
..\.venv\Scripts\python.exe -m pytest test_agents.py -v
..\.venv\Scripts\python.exe benchmark.py
```

Sau benchmark, xem profile tại `state/profiles/dungct.md` (file thật, không phải tên `User.md`).
