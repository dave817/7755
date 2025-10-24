"""
Data models for the dating chatbot
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class PersonalityType(str, Enum):
    """Personality types for character generation"""
    GENTLE = "溫柔體貼型"  # Gentle & Caring
    CHEERFUL = "活潑開朗型"  # Cheerful & Lively
    INTELLECTUAL = "知性優雅型"  # Intellectual & Elegant
    CUTE = "可愛天真型"  # Cute & Innocent


class DreamType(BaseModel):
    """User's dream partner type"""
    personality_traits: List[str] = Field(..., description="性格特質")
    physical_description: Optional[str] = Field(None, description="外貌描述")
    age_range: Optional[str] = Field(None, description="年齡範圍")
    interests: List[str] = Field(default_factory=list, description="興趣愛好")
    occupation: Optional[str] = Field(None, description="職業背景")
    talking_style: str = Field(..., description="說話風格")


class CustomMemory(BaseModel):
    """User's custom memory and preferences"""
    likes: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="喜好 (food, activities, topics)"
    )
    dislikes: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="不喜歡"
    )
    habits: Dict[str, str] = Field(
        default_factory=dict,
        description="習慣 (daily_routine, communication_style)"
    )
    personal_background: Dict[str, str] = Field(
        default_factory=dict,
        description="個人背景 (occupation, hobbies, life_goals)"
    )


class UserProfile(BaseModel):
    """Complete user profile for character generation"""
    user_name: str = Field(..., description="用戶名稱")
    user_gender: str = Field(..., description="用戶性別")
    user_preference: str = Field(..., description="用戶喜歡的性別")
    preferred_character_name: Optional[str] = Field(None, description="用戶指定的角色名字")
    appearance_description: Optional[str] = Field(None, description="角色外貌描述（用於圖片生成）")
    dream_type: DreamType
    custom_memory: CustomMemory


class CharacterSettings(BaseModel):
    """Character settings for SenseChat API"""
    name: str = Field(..., max_length=50, description="角色姓名")
    gender: str = Field(..., max_length=50, description="角色性別")
    identity: Optional[str] = Field(None, max_length=200, description="角色身份")
    nickname: Optional[str] = Field(None, max_length=50, description="角色別名")
    detail_setting: Optional[str] = Field(None, max_length=500, description="詳細設定")
    other_setting: Optional[str] = Field(None, max_length=2000, description="其他設定(JSON字串)")
    feeling_toward: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="好感度設定"
    )


class RoleSetting(BaseModel):
    """Role setting for conversation"""
    user_name: str = Field(..., description="用戶角色名稱")
    primary_bot_name: str = Field(..., description="AI角色名稱")


class Message(BaseModel):
    """Chat message"""
    name: str = Field(..., description="說話者名稱")
    content: str = Field(..., description="對話內容")


class ChatRequest(BaseModel):
    """Request to SenseChat Character API"""
    model: str
    character_settings: List[CharacterSettings]
    role_setting: RoleSetting
    messages: List[Message]
    max_new_tokens: int = 1024
    n: int = 1


class ChatResponse(BaseModel):
    """Response from SenseChat Character API"""
    id: str
    reply: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
