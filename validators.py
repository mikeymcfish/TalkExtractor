import regex as re

SPEAKER_LINE = re.compile(r"^(Speaker\s+\d+):\s")

def validate_output(text: str, min_lines: int = 2, max_speaker_index: int = 9) -> bool:
    if not text:
        return False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < min_lines:
        return False
    if not all(SPEAKER_LINE.match(ln) for ln in lines):
        return False
    for ln in lines:
        try:
            num = int(ln.split(":")[0].split()[1])
            if num > max_speaker_index:
                return False
        except Exception:
            return False
    return True
