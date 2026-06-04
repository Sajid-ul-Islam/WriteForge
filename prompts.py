SYSTEM_PROMPT = "You are a professional editorial writer with expertise in crafting compelling articles across multiple styles and tones."

TONES = ["Persuasive", "Formal", "Op-ed", "Analytical", "Blog"]

LENGTHS = {
    "Short": "around 300 words",
    "Medium": "around 600 words",
    "Long": "around 1000 words",
}


def build_article_prompt(text: str, tone: str, length: str, context: str = "") -> str:
    length_desc = LENGTHS.get(length, LENGTHS["Medium"])
    context_section = f"\n\nReference Context:\n{context}\n" if context else ""
    return f"""You are an elite editorial writer.

Task:
1. Detect the language of the input text
2. Convert it into a high-quality article, utilizing the provided Reference Context if available.

Requirements:
- Strong, attention-grabbing headline
- 3-5 clear subtitles organizing the content
- Well-structured paragraphs with smooth transitions
- {tone} tone throughout
- Target length: {length_desc}
- No filler or fluff

Format your output as:
# [Headline]

{context_section}

[Article body with ## subtitles]

Text:
{text}"""


def build_headlines_prompt(text: str, tone: str) -> str:
    return f"""Generate exactly 3 alternative headline options for an article based on this text.
Each headline should be compelling, {tone.lower()} in tone, and under 15 words.

Format:
1. [headline]
2. [headline]
3. [headline]

Text:
{text}"""


def build_poem_prompt(text: str, tone: str) -> str:
    return f"""Convert the following text into a high-quality poem.

Requirements:
- Capture the core message and emotions of the input
- {tone} tone and style
- Use evocative imagery and rhythmic structure
- Target language: English (translate if the input is not in English)

Text:
{text}"""
