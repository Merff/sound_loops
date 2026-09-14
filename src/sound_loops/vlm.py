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

# Версия промпта — часть ключа кеша в video_analyses (см. analysis.py).
# Менять при любой правке текста промптов/схемы ниже, иначе в базе окажется
# смесь результатов со старой и новой формулировкой без возможности их
# различить. v2: motion убран из того, что определяет VLM (маленькая модель
# почти всегда отвечала "static" даже на явно подвижных лупах — не умеет
# сравнивать отдельные кадры между собой), считается алгоритмически через
# разницу кадров (см. motion.py).
PROMPT_VERSION = "v2"

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


class SceneObservation(BaseModel):
    """То, что VLM реально в состоянии оценить по отдельным кадрам: смысл и
    настроение. Motion сюда намеренно не входит — маленькая модель ненадёжно
    сравнивает отдельные картинки между собой, это считается отдельно и
    дёшево через разницу кадров (см. motion.py)."""

    summary: str = Field(description="One sentence describing what happens in the video, in English.")
    mood: list[Mood] = Field(min_length=1, max_length=3, description="One to three DIFFERENT moods that fit the scene.")
    is_comic: bool = Field(description="Whether what's happening is funny or absurd.")

    @field_validator("mood")
    @classmethod
    def _dedupe_mood(cls, value: list[Mood]) -> list[Mood]:
        """4B-модель иногда повторяет одно и то же значение (['dreamy', 'dreamy']) —
        схема гарантирует длину 1-3, но не уникальность, дедуп молча дешевле
        ретрая на VLM ради чисто косметической проблемы."""
        return list(dict.fromkeys(value))


class SceneDescription(BaseModel):
    """Полное описание сцены — SceneObservation (VLM) + motion (алгоритм),
    собранное воедино. Это то, что кешируется в video_analyses и то, что
    видит шаг B (compose_music_query)."""

    summary: str
    motion: Motion
    mood: list[Mood]
    is_comic: bool


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
        """Кадры лупа (JPEG-байты) -> смысл и настроение сцены (без motion)."""

    def compose_music_query(self, scene: SceneDescription) -> MusicQuery:
        """Полное описание сцены -> короткий текстовый запрос для CLAP-поиска музыки."""


_SCENE_SYSTEM_PROMPT = (
    "You are looking at frames from a short silent video loop. Your job is "
    "to describe what happens and its mood, briefly, so that someone else "
    "can later pick fitting background music, without seeing the video "
    "themselves. Answer only in English and only using the fixed categories "
    "given by the schema."
)
_SCENE_INSTRUCTION = (
    "These frames are evenly sampled from one video loop, in order. "
    "Describe the scene."
)

_MUSIC_SYSTEM_PROMPT = (
    "You turn a short video scene description into a search query for "
    "finding background music with a text-to-audio model (CLAP). CLAP was "
    "trained on descriptions of SOUND, not of scenes: a query that "
    "describes what happens in the video (a person, a place, an action) "
    "retrieves irrelevant results. A query that describes the MUSIC itself "
    "(genre or instruments, tempo, mood, vocals) works well.\n\n"
    "Examples:\n"
    "Scene: a cat slowly stretches on a sunlit windowsill, mood calm/dreamy, "
    "motion slow, not comic\n"
    "Query: slow dreamy ambient with soft piano and warm pads, instrumental\n\n"
    "Scene: a skateboarder loses balance and comically tumbles onto grass, "
    "mood comic/joyful, motion fast, comic\n"
    "Query: upbeat quirky circus-style track with playful brass, instrumental\n\n"
    "Scene: waves crash against dark rocks under a storm, mood tense/epic, "
    "motion chaotic, not comic\n"
    "Query: intense cinematic orchestral with pounding drums and low brass, instrumental\n\n"
    "Never mention anything from the video itself — no people, animals, "
    "places, actions, objects, shapes, colors or patterns — describe only "
    "the music, as if you had never seen the video, just a mood brief. Also "
    "no artist names or track titles. Answer only in English, 5 to 20 words."
)
_MUSIC_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _MUSIC_SYSTEM_PROMPT),
        (
            "human",
            "Scene: {summary}, mood {mood}, motion {motion}, {comic_note}\nQuery:",
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

    def __init__(self, model: str, base_url: str, context_length: int) -> None:
        self.model_id = model
        chat = ChatOllama(model=model, base_url=base_url, num_ctx=context_length)

        self._scene_chain = (
            RunnableLambda(_scene_messages) | chat.with_structured_output(SceneObservation)
        ).with_retry(stop_after_attempt=2)
        self._music_chain = (_MUSIC_PROMPT | chat.with_structured_output(MusicQuery)).with_retry(
            stop_after_attempt=2
        )

    def describe_scene(self, frames: Sequence[bytes]) -> SceneObservation:
        return self._scene_chain.invoke(frames)

    def compose_music_query(self, scene: SceneDescription) -> MusicQuery:
        comic_note = "the scene is comic/absurd" if scene.is_comic else "the scene is not comic"
        return self._music_chain.invoke(
            {
                "summary": scene.summary,
                "mood": ", ".join(scene.mood),
                "motion": scene.motion,
                "comic_note": comic_note,
            }
        )
