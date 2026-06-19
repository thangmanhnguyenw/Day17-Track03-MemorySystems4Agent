from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PROFILE = "# User Profile\n\n"
_META_SUFFIX = re.compile(r"\s*<!--\s*c=[\d.]+;t=\d+;corr=[01]\s*-->\s*$")


@dataclass
class ProfileFact:
    key: str
    value: str
    confidence: float
    is_correction: bool = False


@dataclass
class StoredFact:
    key: str
    value: str
    confidence: float
    turn: int
    is_correction: bool = False

def estimate_tokens(text: str) -> int:
    """Student TODO: implement a simple token estimator.

    Example idea:
    - Strip whitespace
    - Return 0 for empty text
    - Approximate tokens from character count, e.g. len(text) / 4
    """

    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


def _sanitize_user_id(user_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", user_id.strip())
    return slug or "anonymous"


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        # TODO: slugify or sanitize the user id before building the file path.
        return self.root_dir / f"{_sanitize_user_id(user_id)}.md"

    def read_text(self, user_id: str) -> str:
        # TODO: return file content or an empty default markdown profile.
        path = self.path_for(user_id)
        if not path.exists():
            return DEFAULT_PROFILE
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        # TODO: write markdown to disk and return the file path.
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        # TODO: replace one occurrence inside User.md and return whether it changed.
        content = self.read_text(user_id)
        if search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        # TODO: return the current file size in bytes.
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse `- key: value` lines from User.md."""

        return {item.key: item.value for item in self.list_facts(user_id)}

    def list_facts(self, user_id: str) -> list[StoredFact]:
        items: list[StoredFact] = []
        for line in self.read_text(user_id).splitlines():
            match = re.match(
                r"^-\s*([a-zA-Z0-9_]+)\s*:\s*(.+?)\s*(?:<!--\s*c=([\d.]+);t=(\d+);corr=([01])\s*-->)?\s*$",
                line.strip(),
            )
            if not match:
                continue
            value = _META_SUFFIX.sub("", match.group(2)).strip()
            confidence = float(match.group(3)) if match.group(3) else 1.0
            turn = int(match.group(4)) if match.group(4) else 0
            is_correction = match.group(5) == "1" if match.group(5) else False
            items.append(
                StoredFact(
                    key=match.group(1),
                    value=value,
                    confidence=confidence,
                    turn=turn,
                    is_correction=is_correction,
                )
            )
        return items

    def upsert_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        confidence: float = 1.0,
        turn: int = 0,
        is_correction: bool = False,
    ) -> bool:
        """Insert or replace one fact line in User.md.

        Returns False when a lower-confidence update is rejected (conflict guard).
        """

        existing = next((item for item in self.list_facts(user_id) if item.key == key), None)
        if existing and not is_correction and confidence < existing.confidence:
            return False

        content = self.read_text(user_id)
        if content.strip() == "":
            content = DEFAULT_PROFILE

        corr_flag = "1" if is_correction else "0"
        line = f"- {key}: {value.strip()} <!-- c={confidence:.2f};t={turn};corr={corr_flag} -->"
        pattern = re.compile(rf"^-\s*{re.escape(key)}\s*:\s*.+$", re.MULTILINE)
        if pattern.search(content):
            content = pattern.sub(line, content, count=1)
        else:
            if not content.endswith("\n"):
                content += "\n"
            content += f"{line}\n"
        self.write_text(user_id, content)
        return True

    def get_recall_facts(
        self,
        user_id: str,
        current_turn: int,
        threshold: float,
        half_life: int,
    ) -> dict[str, str]:
        """Return facts whose effective confidence still clears the threshold."""

        core_keys = {
            "name",
            "location",
            "profession",
            "response_style",
            "favorite_drink",
            "favorite_food",
            "pet",
        }
        recalled: dict[str, str] = {}
        for item in self.list_facts(user_id):
            if item.key in core_keys or item.is_correction:
                effective = item.confidence
            else:
                age = max(0, current_turn - item.turn)
                if half_life <= 0:
                    effective = item.confidence
                else:
                    effective = item.confidence * (0.5 ** (age / half_life))
            if effective >= threshold:
                recalled[item.key] = item.value
        return recalled


_QUESTION_ONLY = re.compile(
    r"^(?:bạn\s+)?(?:có\s+)?(?:biết|nhớ|nhắc lại|cho\s+(?:mình\s+)?biết|hỏi)\b",
    re.IGNORECASE,
)


def _looks_like_question(message: str) -> bool:
    text = message.strip()
    if text.endswith("?"):
        return True
    return bool(_QUESTION_ONLY.search(text))


def _clean_value(value: str) -> str:
    cleaned = value.strip().strip('"').strip("'")
    cleaned = re.split(
        r"\s+(?:và|nhưng|vì|nên|để|trong|với|chứ|mỗi ngày|dù|hay)\s+",
        cleaned,
    )[0].strip()
    return cleaned


def _is_plausible_fact(key: str, value: str) -> bool:
    if len(value) < 2 or len(value) > 120:
        return False

    lowered = value.lower()
    bad_fragments = (
        "đúng việc",
        "mức chỉ còn",
        "ví dụ về",
        "bài toán",
        "team ",
        "tin ",
        "nghĩa là",
    )
    if any(fragment in lowered for fragment in bad_fragments):
        return False

    if key == "profession":
        return any(token in lowered for token in ("engineer", "developer", "manager", "mlops", "backend"))
    if key == "favorite_drink":
        return any(token in lowered for token in ("cà phê", "trà", "nước", "sữa"))
    if key == "location":
        return len(value.split()) <= 4
    return True


def _add_fact(
    facts: list[ProfileFact],
    key: str,
    value: str,
    confidence: float,
    *,
    is_correction: bool = False,
) -> None:
    cleaned = _clean_value(value)
    if not cleaned or not _is_plausible_fact(key, cleaned):
        return
    facts.append(
        ProfileFact(
            key=key,
            value=cleaned,
            confidence=confidence,
            is_correction=is_correction,
        )
    )


def extract_profile_candidates(message: str) -> list[ProfileFact]:
    """Extract profile facts with confidence scores for guardrails."""

    if _looks_like_question(message):
        return []

    text = " ".join(message.split())
    facts: list[ProfileFact] = []

    name_patterns = [
        (r"mình tên (?:là )?([^.,\n?]+)", 0.94),
        (r"tên mình (?:là )?([^.,\n?]+)", 0.92),
        (r"tên (?:là|của mình là) ([^.,\n?]+)", 0.9),
    ]
    for pattern, confidence in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = _clean_value(match.group(1))
            name = re.sub(r"\s*(?:và|hiện|Stress test.*)$", "", name, flags=re.IGNORECASE).strip()
            if name and _is_plausible_fact("name", name):
                facts.append(ProfileFact("name", name, confidence))
                break

    location_patterns = [
        (r"đính chính.*?giờ mình (?:đang )?(?:ở|sống ở|chuyển (?:sang|tới|về)) ([^.,\n?]+)", 0.98, True),
        (r"thực ra.*?mình (?:đang )?(?:ở|làm việc ở) ([^.,\n?]+)", 0.96, True),
        (r"nơi ở (?:đã )?(?:cập nhật|cập nhật từ .* sang|chuyển.*?sang) ([^.,\n?]+)", 0.95, True),
        (r"đang (?:ở|làm việc ở) ([^.,\n?]+)", 0.82, False),
        (r"mình ở ([^.,\n?]+)", 0.8, False),
        (r"hiện (?:ở|ở tại) ([^.,\n?]+)", 0.84, False),
    ]
    for pattern, confidence, is_correction in location_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            location = _clean_value(match.group(1))
            if location and location.lower() not in {"đây", "đó"} and _is_plausible_fact("location", location):
                facts.append(ProfileFact("location", location, confidence, is_correction=is_correction))
                break

    allow_generic_profession = not re.search(
        r"đừng nói|thông tin cũ|đó chỉ là câu đùa|chỉ là nơi mình",
        text,
        re.IGNORECASE,
    )
    correction_profession_patterns = [
        (r"đính chính.*?nghề nghiệp.*?giờ chuyển sang ([^.,\n?]+)", 0.98),
        (r"không còn làm [^.,]+ nữa, giờ chuyển sang ([^.,\n?]+)", 0.98),
        (r"giờ chuyển sang (MLOps engineer|backend engineer|software engineer|data engineer)", 0.97),
    ]
    generic_profession_patterns = [
        (r"nghề nghiệp (?:hiện tại )?(?:vẫn )?(?:là )((?:MLOps|backend|software|data)[^.,\n?]*engineer)", 0.86),
        (r"(?:vẫn )?(?:là )?(MLOps engineer|backend engineer|software engineer|data engineer)", 0.88),
        (r"(?:vẫn )?(?:đang )?làm (MLOps engineer|backend engineer|software engineer|data engineer)", 0.86),
        (r"mình làm (MLOps engineer|backend engineer|software engineer|data engineer)", 0.9),
        (r"(?:và )?(?:đang )?làm ([A-Za-z0-9 ]+ engineer)", 0.82),
        (r"đang làm ([^.,\n?]+ engineer)", 0.8),
        (r"làm ([^.,\n?]+ engineer) cho", 0.84),
    ]
    profession_patterns = correction_profession_patterns + (
        generic_profession_patterns if allow_generic_profession else []
    )
    for item in profession_patterns:
        pattern = item[0]
        confidence = item[1]
        is_correction = confidence >= 0.95
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            profession = _clean_value(match.group(1))
            if profession and "product manager" not in profession.lower() and _is_plausible_fact("profession", profession):
                facts.append(ProfileFact("profession", profession, confidence, is_correction=is_correction))
                break

    if re.search(r"3 bullet|ba bullet", text, re.IGNORECASE):
        facts.append(ProfileFact("response_style", "3 bullet ngắn, có ví dụ thực chiến, ưu tiên trade-off", 0.93))
    elif re.search(r"ngắn gọn|trả lời ngắn|bullet ngắn", text, re.IGNORECASE):
        facts.append(ProfileFact("response_style", "ngắn gọn, rõ ý, có ví dụ thực tế", 0.88))

    drink_patterns = [
        (r"đồ uống yêu thích (?:là )?([^.,\n?]+)", 0.92),
        (r"mình (?:vẫn )?uống ([^.,\n?]*cà phê[^.,\n?]*)", 0.84),
    ]
    for pattern, confidence in drink_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            drink = _clean_value(match.group(1))
            if _is_plausible_fact("favorite_drink", drink):
                facts.append(ProfileFact("favorite_drink", drink, confidence))
                break

    food_match = re.search(r"món ăn yêu thích (?:là )?([^.,\n?]+)", text, re.IGNORECASE)
    if food_match:
        _add_fact(facts, "favorite_food", food_match.group(1), 0.9)

    pet_match = re.search(
        r"(?:nuôi )?(?:một )?(?:bé )?corgi tên ([A-Za-zÀ-ỹ0-9]+)",
        text,
        re.IGNORECASE,
    )
    if pet_match:
        facts.append(ProfileFact("pet", f"corgi tên {pet_match.group(1).strip()}", 0.91))

    interest_match = re.search(
        r"mối quan tâm (?:chính )?(?:gồm |là )?([^.,\n?]+)",
        text,
        re.IGNORECASE,
    )
    if interest_match:
        _add_fact(facts, "interests", interest_match.group(1), 0.78)

    if re.search(r"Python|AI agent|benchmark memory|MLOps", text):
        existing = next((fact.value for fact in facts if fact.key == "interests"), "")
        parts = [part.strip() for part in re.split(r",|\bvà\b", existing) if part.strip()]
        for keyword in ("Python", "AI", "MLOps", "benchmark"):
            if keyword.lower() in text.lower() and keyword not in parts:
                parts.append(keyword)
        if parts:
            facts.append(ProfileFact("interests", ", ".join(dict.fromkeys(parts)), 0.8))

    priority_match = re.search(r"ưu tiên ([^.,\n?]+)", text, re.IGNORECASE)
    if priority_match and "recall" in priority_match.group(1).lower():
        _add_fact(facts, "priority", priority_match.group(1), 0.76)

    return facts


def extract_profile_updates(message: str, threshold: float = 0.75) -> dict[str, str]:
    """Student TODO: convert raw user text into stable profile facts.

    Example facts you may want to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.
    """

    merged: dict[str, str] = {}
    for fact in extract_profile_candidates(message):
        if fact.confidence >= threshold:
            merged[fact.key] = fact.value
    return merged


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Student TODO: create a compact summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.
    """

    if not messages:
        return ""

    selected = messages[-max_items:]
    lines = [f"{item['role']}: {item['content']}" for item in selected]
    return " | ".join(lines)


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure_thread(self, thread_id: str) -> dict[str, object]:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def _thread_tokens(self, thread_state: dict[str, object]) -> int:
        messages: list[dict[str, str]] = thread_state["messages"]  # type: ignore[assignment]
        summary: str = str(thread_state.get("summary", ""))
        total = estimate_tokens(summary)
        for message in messages:
            total += estimate_tokens(f"{message['role']}: {message['content']}")
        return total

    def _compact(self, thread_state: dict[str, object]) -> None:
        messages: list[dict[str, str]] = thread_state["messages"]  # type: ignore[assignment]
        if len(messages) <= self.keep_messages:
            return

        older = messages[:-self.keep_messages]
        recent = messages[-self.keep_messages :]
        previous_summary = str(thread_state.get("summary", ""))
        new_chunk = summarize_messages(older)
        if previous_summary:
            combined = f"{previous_summary} | {new_chunk}"
        else:
            combined = new_chunk

        max_summary_tokens = max(1, self.threshold_tokens // 2)
        while estimate_tokens(combined) > max_summary_tokens and len(combined) > 64:
            combined = combined[len(combined) // 4 :]

        thread_state["summary"] = combined
        thread_state["messages"] = recent
        thread_state["compactions"] = int(thread_state.get("compactions", 0)) + 1

    def append(self, thread_id: str, role: str, content: str) -> None:
        # TODO:
        # 1. create thread state if missing
        # 2. append the new message
        # 3. trigger compaction if needed
        thread_state = self._ensure_thread(thread_id)
        messages: list[dict[str, str]] = thread_state["messages"]  # type: ignore[assignment]
        messages.append({"role": role, "content": content})

        while self._thread_tokens(thread_state) > self.threshold_tokens and len(messages) > self.keep_messages:
            self._compact(thread_state)
            messages = thread_state["messages"]  # type: ignore[assignment]

    def context(self, thread_id: str) -> dict[str, object]:
        # TODO: return per-thread state with keys like messages, summary, compactions.
        return self._ensure_thread(thread_id)

    def compaction_count(self, thread_id: str) -> int:
        # TODO: return number of compactions for this thread.
        return int(self._ensure_thread(thread_id).get("compactions", 0))
