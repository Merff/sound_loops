"""VLM-описание сцены лупа и перевод его в музыкальный запрос (шаги A/Б
итерации 2, см. docs/sound_loops-iteration-2.md).

Обе задачи используют одну и ту же модель через Ollama: отдельная текстовая
модель для шага B выгружала бы веса шага A из памяти при каждом прогоне
(Ollama держит одну модель за раз при нехватке ОЗУ), а qwen3-vl:4b-instruct
с чисто текстовой задачей справляется нормально.

Настоящая реализация спрятана за Protocol SceneAnalyzer — по тому же
принципу, что Embedder в embeddings.py: тесты подставляют детерминированный
fake вместо реального похода в Ollama.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Literal, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator

# Часть ключа кеша в video_analyses (см. analysis.py) — менять при любой
# правке текста промптов/схемы ниже, иначе в базе смешаются результаты
# старой и новой формулировки без возможности их различить.
PROMPT_VERSION = "v6"

Motion = Literal["static", "slow", "moderate", "fast", "chaotic"]
Mood = Literal[
    "calm",
    "tense",
    "joyful",
    "melancholic",
    "epic",
    "comic",
    "eerie",
    "romantic",
    "aggressive",
    "nostalgic",
    "dreamy",
    "triumphant",
]
Setting = Literal[
    "combat",
    "horror",
    "domestic",
    "nature",
    "scifi_fantasy",
    "sports",
    "romance",
    "nightlife",
    "performance",
    "abstract",
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
        схема гарантирует длину 1-3, но не уникальность, дедуп молча дешевле
        ретрая на VLM ради чисто косметической проблемы."""
        return list(dict.fromkeys(value))


class SceneDescription(BaseModel):
    """Полное описание сцены — SceneObservation (VLM) + motion (алгоритм),
    собранное воедино. Это то, что кешируется в video_analyses и что видит
    шаг B (compose_music_query)."""

    setting: Setting
    motion: Motion
    mood: list[Mood]


class MusicQuery(BaseModel):
    """Шаг B: короткое текстовое описание музыки для CLAP-поиска.

    CLAP-текстовая башня обучена на описаниях звука, а не сцены — запрос
    вида «a man falls off a skateboard» ищет мусор. Нужно «fast aggressive
    punk rock with distorted guitars, instrumental».
    """

    query: str = Field(
        description=(
            "5 to 20 word English description of MUSIC (not the video): "
            "genre or instruments, tempo, mood, and whether it has vocals "
            "('instrumental' or 'with vocals'). No mentions of the video's "
            "subject, people, animals, artist names or track titles."
        )
    )

    @field_validator("query")
    @classmethod
    def _word_count(cls, value: str) -> str:
        n = len(value.split())
        if not (5 <= n <= 20):
            raise ValueError(f"запрос должен быть из 5-20 слов, получено {n}: {value!r}")
        return value


class SceneAnalyzer(Protocol):
    model_id: str
    prompt_version: str

    def describe_scene(self, frames: Sequence[bytes]) -> SceneObservation:
        """Кадры лупа (JPEG-байты) -> обстановка и настроение сцены (без motion)."""

    def compose_music_query(self, scene: SceneDescription) -> MusicQuery:
        """Полное описание сцены -> короткий текстовый запрос для CLAP-поиска музыки."""


_SCENE_SYSTEM_PROMPT = (
    "You are looking at frames from a short silent video loop, sampled in "
    "order. Classify it so someone else can pick fitting background music "
    "without seeing the video.\n\n"
    "setting: closest genre/setting category from the fixed list — decides "
    "the music's genre. Avoid a generic default (e.g. domestic) if a more "
    "specific one fits. Examples:\n"
    "- A dragon flies over an army on a snowy battlefield -> combat\n"
    "- Soldiers crawl through mud under gunfire -> combat\n"
    "- A masked figure stalks someone in a dark corridor -> horror\n"
    "- People dancing under colored lights in a club -> nightlife\n\n"
    "mood: 1-3 moods justified by what's visible (expressions, action, "
    "lighting, color). Avoid a generic default (e.g. dreamy) if a more "
    "specific one fits. Examples:\n"
    "- People laughing and dancing at a bright, colorful party -> joyful, triumphant\n"
    "- A soldier crawling through mud under gunfire, gritted teeth -> tense, aggressive\n"
    "- An old man alone on a park bench watching leaves fall -> melancholic, nostalgic\n"
    "- A cat knocks a vase off a table and looks startled -> comic\n\n"
    "English only, fixed categories only."
)
_SCENE_INSTRUCTION = (
    "These frames are evenly sampled from one video loop, in order. "
    "Classify the scene."
)

# Жанровые теги реально проиндексированного FMA_small (см. tracks.genre) —
# без этого шаг B тянется к orchestral/cinematic, которых в библиотеке нет.
_LIBRARY_GENRES = "Electronic, Rock, Hip-Hop, Pop, Folk, International, Experimental, Instrumental"

_MUSIC_SYSTEM_PROMPT = (
    "Turn a video scene's setting and mood into a short search query for "
    "background music (CLAP text-to-audio). Describe only the music — "
    "genre, instruments, tempo — never the video's content.\n\n"
    f"The music library only has these genres: {_LIBRARY_GENRES}. Frame the "
    "genre/instruments part of the query using one of them (or close to "
    "it) — never say orchestral, cinematic, soundtrack, or symphonic, "
    "those don't exist here. Translate the mood into the closest available "
    "genre instead (e.g. an intense battle scene -> aggressive industrial "
    "electronic or heavy rock, not orchestral).\n\n"
    "Examples:\n"
    "Setting: nature, mood calm/dreamy, motion slow\n"
    "Query: slow dreamy folk instrumental with soft acoustic guitar and warm pads\n\n"
    "Setting: sports, mood comic/joyful, motion fast\n"
    "Query: upbeat quirky electronic track with playful synth and a bouncy beat, instrumental\n\n"
    "Setting: combat, mood tense/aggressive, motion chaotic\n"
    "Query: aggressive industrial electronic with distorted bass and a pounding beat, instrumental\n\n"
    "Setting: nightlife, mood romantic/melancholic, motion moderate\n"
    "Query: slow moody electronic with warm synth pads and a soft beat, instrumental\n\n"
    "No artist names or track titles. English only, 5 to 20 words."
)
_MUSIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _MUSIC_SYSTEM_PROMPT),
        (
            "human",
            "Setting: {setting}, mood {mood}, motion {motion}\nQuery:",
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

        self._scene_chain = (
            RunnableLambda(_scene_messages) | chat.with_structured_output(SceneObservation)
        ).with_retry(stop_after_attempt=2)
        self._music_chain = (_MUSIC_PROMPT | chat.with_structured_output(MusicQuery)).with_retry(
            stop_after_attempt=2
        )

    def describe_scene(self, frames: Sequence[bytes]) -> SceneObservation:
        return self._scene_chain.invoke(frames)

    def compose_music_query(self, scene: SceneDescription) -> MusicQuery:
        return self._music_chain.invoke(
            {
                "setting": scene.setting,
                "mood": ", ".join(scene.mood),
                "motion": scene.motion,
            }
        )
