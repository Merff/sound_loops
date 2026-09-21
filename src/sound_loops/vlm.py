"""VLM-описание сцены лупа и перевод его в музыкальный запрос.

Обе задачи используют одну и ту же модель через Ollama: отдельная текстовая
модель для шага B выгружала бы веса шага A из памяти при каждом прогоне
(Ollama держит одну модель за раз при нехватке ОЗУ), а qwen3-vl:4b-instruct
с чисто текстовой задачей справляется нормально.

Настоящая реализация спрятана за Protocol SceneAnalyzer: тесты подставляют детерминированный
fake вместо реального похода в Ollama.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Literal, Protocol

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator

# Часть ключа кеша в video_analyses (см. analysis.py) — менять при любой
# правке текста промптов/схемы ниже, иначе в базе смешаются результаты
# старой и новой формулировки без возможности их различить.
PROMPT_VERSION = "v8"

Motion = Literal["static", "slow", "moderate", "fast", "chaotic"]
Mood = Literal[
    "calm",  # спокойное
    "tense",  # напряжённое
    "joyful",  # радостное
    "melancholic",  # меланхоличное
    "epic",  # эпичное
    "comic",  # комичное
    "eerie",  # жуткое
    "romantic",  # романтичное
    "aggressive",  # агрессивное
    "nostalgic",  # ностальгическое
    "dreamy",  # мечтательное
    "triumphant",  # триумфальное
    "tragic",  # трагичное
    "festive",  # праздничное
]
Setting = Literal[
    "combat",  # бой/сражение
    "horror",  # ужасы
    "domestic",  # быт/дом
    "nature",  # природа
    "scifi_fantasy",  # фантастика/фэнтези
    "sports",  # спорт
    "romance",  # романтика
    "nightlife",  # ночная жизнь/клубы
    "performance",  # выступление/сцена
    "urban",  # город
    "abstract",  # абстрактное/небытовое
]


class SceneObservation(BaseModel):
    """То, что VLM реально в состоянии оценить по отдельным кадрам —
    закрытые категории жанра и настроения. Motion сюда не входит — считается
    алгоритмически (см. motion.py)."""

    setting: Setting = Field(
        description="The single closest genre/setting category for the scene — decides the music's genre."
    )
    mood: list[Mood] = Field(min_length=1, max_length=3, description="One to three DIFFERENT moods that fit the scene.")

    @field_validator("mood")
    @classmethod
    def _dedupe_mood(cls, value: list[Mood]) -> list[Mood]:
        """4B-модель иногда повторяет одно и то же значение (['dreamy', 'dreamy']) —
        схема гарантирует длину 1-3, но не уникальность."""
        return list(dict.fromkeys(value))


class SceneDescription(BaseModel):
    """Полное описание сцены — SceneObservation (VLM) + motion (алгоритм),
    собранное воедино. Это то, что кешируется в video_analyses и что видит
    шаг B (compose_music_query)."""

    setting: Setting
    motion: Motion
    mood: list[Mood]


Vocals = Literal["instrumental", "with_vocals"]


class MusicQuery(BaseModel):
    """короткое текстовое описание музыки для CLAP-поиска.

    CLAP-текстовая башня обучена на описаниях звука, а не сцены — запрос
    вида «a man falls off a skateboard» ищет мусор. Нужно «fast aggressive
    punk rock with distorted guitars, instrumental».
    """

    query: str = Field(
        description=(
            "5 to 20 word English description of MUSIC (not the video): "
            "genre or instruments, tempo, mood. No mentions of the video's "
            "subject, people, animals, artist names or track titles."
        )
    )
    vocals: Vocals = Field(
        description="Whether the ideal track for this scene has vocals — separate from the query text "
        "so it can be used as a search filter, not just a word inside it."
    )

    @field_validator("query")
    @classmethod
    def _word_count(cls, value: str) -> str:
        n = len(value.split())
        if not (5 <= n <= 20):
            raise ValueError(f"запрос должен быть из 5-20 слов, получено {n}: {value!r}")
        return value


class RerankChoice(BaseModel):
    """переранжирование: модель получает пронумерованный
    список кандидатов с их атрибутами и выбирает один, объясняя выбор —
    RAG в чистом виде, контекст собран из базы, а не лежит в весах модели."""

    candidate_index: int = Field(description="1-based number of the chosen candidate from the numbered list.")
    reasoning: str = Field(description="One or two sentences on why this candidate fits the scene best.")


class SceneAnalyzer(Protocol):
    model_id: str
    prompt_version: str

    def describe_scene(self, frames: Sequence[bytes]) -> SceneObservation:
        """Кадры лупа (JPEG-байты) -> обстановка и настроение сцены (без motion)."""

    def compose_music_query(self, scene: SceneDescription, feedback_history: Sequence[str] = ()) -> MusicQuery:
        """Полное описание сцены (+ опционально текст обратной связи пользователя) -> короткий текстовый
        запрос для CLAP-поиска музыки."""

    def rerank(self, scene: SceneDescription, candidate_descriptions: Sequence[str]) -> RerankChoice:
        """Описание сцены + пронумерованные читаемые описания кандидатов -> выбор + объяснение."""

    def bind_tools(self, tools: Sequence[BaseTool]) -> Runnable[LanguageModelInput, object]:
        """Та же модель с привязанными инструментами —
        единственный способ достать вызываемую модель наружу, чтобы вызывающий
        код (agent_planner.py) не был завязан на ChatOllama напрямую."""


_SCENE_SYSTEM_PROMPT = (
    "You are looking at frames from a short silent video loop, sampled in "
    "order. Classify it so someone else can pick fitting background music "
    "without seeing the video.\n\n"
    "setting: closest genre/setting category from the fixed list — decides "
    "the music's genre. Pick only what the frames actually show — don't "
    "fall back to a familiar-sounding category (domestic, combat, urban, "
    "etc.) just because nothing else jumps out. Examples:\n"
    "- A dragon flies over an army on a snowy battlefield -> combat\n"
    "- A masked figure stalks someone in a dark corridor -> horror\n"
    "- People dancing under colored lights in a club -> nightlife\n"
    "- Daytime traffic and pedestrians crossing a city street -> urban\n"
    "- Colorful shapes morph and pulse with no recognizable subject -> abstract\n"
    "- A singer performs on a lit stage in front of a crowd -> performance\n\n"
    "mood: 1-3 moods justified by what's visible (expressions, action, "
    "lighting, color). Pick only what's actually visible — don't default to "
    "a familiar mood (joyful, tense, melancholic, etc.) when the scene "
    "doesn't clearly show it. Examples:\n"
    "- People laughing and dancing at a bright, colorful party -> joyful, triumphant\n"
    "- A soldier crawling through mud under gunfire, gritted teeth -> tense, aggressive\n"
    "- An old man alone on a park bench watching leaves fall -> melancholic, nostalgic\n"
    "- A cat knocks a vase off a table and looks startled -> comic\n"
    "- A hero raises a sword as armies clash beneath a fiery sky -> epic\n"
    "- Confetti falls as people cheer and raise glasses at a celebration -> festive\n"
    "- A single wilted flower lies beside an empty picture frame -> tragic\n\n"
    "English only, fixed categories only."
)
_SCENE_INSTRUCTION = (
    "These frames are evenly sampled from one video loop, in order. "
    "Classify the scene."
)

# Жанровые теги реально проиндексированного FMA_small (см. tracks.genre) —
# без этого шаг B тянется к orchestral/cinematic, которых в библиотеке нет.
# Библиотека сбалансирована точно (по 1000 треков на жанр, см. README) —
# короткие описания ниже призваны отучить модель от дефолта на electronic
# и дать за что зацепиться помимо самого названия жанра.
LIBRARY_GENRES_WITH_CHARACTER = (
    "Electronic (synths, drum machines, digital production), "
    "Rock (electric guitars, live drums, driving energy), "
    "Hip-Hop (rhythmic beat, heavy bass, sampled loops), "
    "Pop (catchy melodic hook, polished production), "
    "Folk (acoustic guitar, strings, warm and organic), "
    "International (regional instruments/styles outside western pop), "
    "Experimental (unconventional structure, abstract sound design), "
    "Instrumental (atmospheric, functional, piano/ambient-leaning, no strong genre identity)"
)

_MUSIC_SYSTEM_PROMPT = (
    "Turn a video scene's setting and mood into a short search query for "
    "background music (CLAP text-to-audio). Describe only the music — "
    "genre, instruments, tempo — never the video's content or vocals "
    "(vocals go in a separate field).\n\n"
    f"The music library only has these genres, evenly represented (about "
    f"100 tracks each) — pick whichever ACTUALLY fits the mood: "
    f"{LIBRARY_GENRES_WITH_CHARACTER}. Never say orchestral, cinematic, "
    "soundtrack, or symphonic — those don't exist here.\n\n"
    "Electronic is not a safe default — it's one option among eight, no "
    "more statistically likely than any other. Before answering, actively "
    "rule out whether rock, folk, hip-hop, pop, international or "
    "instrumental fits this specific mood better than electronic.\n\n"
    "Examples below show the FORMAT only — never reuse their wording, "
    "genre, or instrument list for a new scene; treat each new scene as "
    "its own fresh judgment call, not a lookup by setting:\n"
    "Setting: domestic, mood calm/dreamy, motion slow\n"
    "Query: slow dreamy instrumental with soft piano and warm strings\n"
    "Vocals: instrumental\n\n"
    "Setting: combat, mood tense/aggressive, motion chaotic\n"
    "Query: aggressive rock with distorted electric guitars and pounding drums\n"
    "Vocals: instrumental\n\n"
    "Setting: nature, mood joyful/festive, motion fast\n"
    "Query: upbeat folk with driving acoustic guitar and lively percussion\n"
    "Vocals: instrumental\n\n"
    "Setting: urban, mood comic/joyful, motion moderate\n"
    "Query: warm mid-tempo pop with a catchy hook and bright synths\n"
    "Vocals: with_vocals\n\n"
    "Setting: nightlife, mood romantic/melancholic, motion slow\n"
    "Query: slow moody synth-pop ballad with soft vocals and warm pads\n"
    "Vocals: with_vocals\n\n"
    "Setting: sports, mood aggressive/epic, motion fast\n"
    "Query: fast aggressive hip-hop with heavy bass and sharp percussion\n"
    "Vocals: with_vocals\n\n"
    "No artist names or track titles. English only, 5 to 20 words."
)
_MUSIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _MUSIC_SYSTEM_PROMPT),
        (
            "human",
            "Setting: {setting}, mood {mood}, motion {motion}{feedback}\nQuery:",
        ),
    ]
)

_RERANK_SYSTEM_PROMPT = (
    "You are picking the single best background music track for a silent "
    "video scene from a short list of candidates already retrieved by a "
    "search engine. Each candidate is described by its title, tempo, mood/"
    "genre tags (from a separate zero-shot classifier — approximate, not "
    "ground truth) and vocals, plus its similarity score to the scene's "
    "music query.\n\n"
    "Pick the candidate that best fits the scene's setting and mood — "
    "tempo and vocals should roughly match what the scene calls for, but "
    "similarity to the query matters too. Reply with the candidate's "
    "number and a short reason (1-2 sentences)."
)
_RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _RERANK_SYSTEM_PROMPT),
        (
            "human",
            "Scene: setting {setting}, mood {mood}, motion {motion}\n\nCandidates:\n{candidates}\n\nBest match:",
        ),
    ]
)


def _frame_content(frames: Sequence[bytes]) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": _SCENE_INSTRUCTION}]
    for frame in frames:
        content.append(
            {
                "type": "image",
                "source_type": "base64",
                "data": base64.b64encode(frame).decode("ascii"),
                "mime_type": "image/jpeg",
            }
        )
    return content


def _scene_messages(frames: Sequence[bytes]) -> list:
    return [SystemMessage(_SCENE_SYSTEM_PROMPT), HumanMessage(content=_frame_content(frames))]


class OllamaSceneAnalyzer:
    """Реализация SceneAnalyzer поверх ChatOllama, собранная как LCEL-цепочка."""

    prompt_version = PROMPT_VERSION

    def __init__(
        self, model: str, base_url: str, context_length: int, temperature: float | None = None
    ) -> None:
        self.model_id = model
        chat_kwargs = {} if temperature is None else {"temperature": temperature}
        chat = ChatOllama(model=model, base_url=base_url, num_ctx=context_length, **chat_kwargs)
        self._chat = chat

        self._scene_chain = (
            RunnableLambda(_scene_messages) | chat.with_structured_output(SceneObservation)
        ).with_retry(stop_after_attempt=2)
        self._music_chain = (_MUSIC_PROMPT | chat.with_structured_output(MusicQuery)).with_retry(
            stop_after_attempt=2
        )
        self._rerank_chain = (_RERANK_PROMPT | chat.with_structured_output(RerankChoice)).with_retry(
            stop_after_attempt=2
        )

    def describe_scene(self, frames: Sequence[bytes]) -> SceneObservation:
        return self._scene_chain.invoke(frames)

    def compose_music_query(self, scene: SceneDescription, feedback_history: Sequence[str] = ()) -> MusicQuery:
        feedback = ""
        if feedback_history:
            feedback = "\nUser feedback on previous suggestions (most recent last), take it into account: " + "; ".join(
                feedback_history
            )
        return self._music_chain.invoke(
            {
                "setting": scene.setting,
                "mood": ", ".join(scene.mood),
                "motion": scene.motion,
                "feedback": feedback,
            }
        )

    def rerank(self, scene: SceneDescription, candidate_descriptions: Sequence[str]) -> RerankChoice:
        return self._rerank_chain.invoke(
            {
                "setting": scene.setting,
                "mood": ", ".join(scene.mood),
                "motion": scene.motion,
                "candidates": "\n".join(candidate_descriptions),
            }
        )

    def bind_tools(self, tools: Sequence[BaseTool]) -> Runnable[LanguageModelInput, object]:
        # .with_retry() — тот же принцип, что у остальных цепочек в этом классе:
        # Ollama иногда роняет вызов с tool calling ошибкой разбора ответа
        # (invalid character ... after object key:value pair, status code -1) —
        # транзиентная проблема стрима/клиента.
        return self._chat.bind_tools(tools).with_retry(stop_after_attempt=2)
