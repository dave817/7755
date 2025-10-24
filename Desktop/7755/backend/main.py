"""
FastAPI backend for Dating Chatbot
Phase 1: User Onboarding & Character Generation
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional

from backend.models import UserProfile, DreamType, CustomMemory
from backend.character_generator import CharacterGenerator
from backend.api_client import SenseChatClient
from backend.database import get_db, init_db
from backend.conversation_manager import ConversationManager
from sqlalchemy.orm import Session
from fastapi import Depends

app = FastAPI(title="Dating Chatbot API", version="1.0.0")

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
api_client = SenseChatClient()
character_generator = CharacterGenerator(api_client=api_client)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    init_db()
    print("Database initialized successfully")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - shows welcome page"""
    return """
    <html>
        <head>
            <title>戀愛聊天機器人</title>
            <meta charset="UTF-8">
        </head>
        <body>
            <h1>歡迎使用戀愛聊天機器人</h1>
            <p>API 文檔: <a href="/docs">/docs</a></p>
            <p>前端界面: <a href="/ui">使用界面</a></p>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "dating-chatbot"}


@app.post("/api/generate-character")
async def generate_character(user_profile: UserProfile) -> Dict:
    """
    Generate AI character based on user's dream type and custom memory

    Args:
        user_profile: User's complete profile

    Returns:
        Character settings and initial greeting
    """
    try:
        # Generate character settings
        character_settings = character_generator.generate_character(user_profile)

        # Generate initial message
        initial_message = character_generator.create_initial_message(
            character_settings["name"],
            user_profile,
            character_settings["gender"]
        )

        return {
            "success": True,
            "character": character_settings,
            "initial_message": initial_message,
            "message": "角色生成成功！"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"角色生成失敗: {str(e)}")


@app.post("/api/test-chat")
async def test_chat(
    character_settings: Dict,
    user_name: str,
    user_message: str
) -> Dict:
    """
    Test chat with generated character

    Args:
        character_settings: Generated character settings
        user_name: User's name
        user_message: User's message

    Returns:
        Character's response
    """
    try:
        # Prepare role setting
        role_setting = {
            "user_name": user_name,
            "primary_bot_name": character_settings["name"]
        }

        # Prepare messages
        messages = [
            {
                "name": user_name,
                "content": user_message
            }
        ]

        # Need both user and character in character_settings for API
        user_character = {
            "name": user_name,
            "gender": "男",  # Default, can be customized
            "detail_setting": "普通用戶"
        }

        api_character_settings = [user_character, character_settings]

        # Call API
        response = api_client.create_character_chat(
            character_settings=api_character_settings,
            role_setting=role_setting,
            messages=messages,
            max_new_tokens=1024
        )

        return {
            "success": True,
            "reply": response["data"]["reply"],
            "full_response": response["data"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聊天失敗: {str(e)}")


@app.get("/api/test-connection")
async def test_connection():
    """Test API connection to SenseChat"""
    try:
        is_connected = api_client.test_connection()
        return {
            "success": is_connected,
            "message": "API 連接成功" if is_connected else "API 連接失敗"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"連接測試失敗: {str(e)}")


# ==================== Phase 2: Persistent Conversation Endpoints ====================

@app.post("/api/v2/create-character")
async def create_character_v2(user_profile: UserProfile, db: Session = Depends(get_db)) -> Dict:
    """
    Phase 2: Create character and save to database

    Args:
        user_profile: User's complete profile
        db: Database session

    Returns:
        Character with character_id for persistent conversations
    """
    try:
        # Initialize conversation manager
        conv_manager = ConversationManager(db, api_client)

        # Get or create user
        user = conv_manager.get_or_create_user(user_profile.user_name)

        # Generate character
        character_settings = character_generator.generate_character(user_profile)

        # Save character to database
        character = conv_manager.save_character(user.user_id, character_settings)

        # Store appearance description if provided
        if user_profile.appearance_description:
            character.appearance_description = user_profile.appearance_description
            db.commit()
            db.refresh(character)

        # Generate initial message
        initial_message = character_generator.create_initial_message(
            character_settings["name"],
            user_profile,
            character_settings["gender"]
        )

        # Save initial message
        conv_manager.save_message(
            user_id=user.user_id,
            character_id=character.character_id,
            speaker_name=character.name,
            content=initial_message,
            favorability_level=1
        )

        return {
            "success": True,
            "user_id": user.user_id,
            "character_id": character.character_id,
            "character": {
                "name": character.name,
                "nickname": character.nickname,
                "gender": character.gender,
                "identity": character.identity,
                "detail_setting": character.detail_setting,
                "other_setting": character.other_setting
            },
            "initial_message": initial_message,
            "favorability_level": 1,
            "message": "角色已創建並保存！",
            "has_appearance": bool(user_profile.appearance_description)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"角色創建失敗: {str(e)}")


@app.post("/api/v2/generate-character-image/{character_id}")
async def generate_character_image_endpoint(
    character_id: int,
    db: Session = Depends(get_db)
) -> Dict:
    """
    Generate first character image (face portrait)

    Args:
        character_id: Character ID
        db: Database session

    Returns:
        Image URL and seed
    """
    try:
        from backend.image_generator import ImageGenerator
        from backend.database import GeneratedImage

        # Get character
        character = db.query(Character).filter(
            Character.character_id == character_id
        ).first()

        if not character:
            raise HTTPException(status_code=404, detail="角色不存在")

        if not character.appearance_description:
            raise HTTPException(status_code=400, detail="角色沒有外貌描述，無法生成圖片")

        # Initialize image generator
        img_gen = ImageGenerator()

        # Get available models
        models_response = img_gen.get_models(size=1, mtp="Checkpoint")
        if "error" in models_response or not models_response.get("data"):
            raise HTTPException(status_code=500, detail="無法獲取圖片生成模型")

        model_id = models_response["data"][0]["id"]

        # Generate character face
        result = img_gen.generate_character_face(
            appearance_description=character.appearance_description,
            character_name=character.name,
            model_id=model_id,
            seed=0  # Random seed for first generation
        )

        if not result:
            raise HTTPException(status_code=500, detail="圖片生成失敗")

        # Save to character
        character.base_image_url = result["url"]
        character.image_seed = result["seed"]
        character.image_model_id = model_id

        # Save to generated images table
        generated_img = GeneratedImage(
            character_id=character_id,
            image_url=result["url"],
            image_type="first_message",
            trigger_count=0,
            activity_description=f"Character portrait: {character.name}"
        )
        db.add(generated_img)

        db.commit()
        db.refresh(character)

        return {
            "success": True,
            "image_url": result["url"],
            "seed": result["seed"],
            "model_id": model_id,
            "message": "角色圖片生成成功！"
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error generating image: {error_details}")
        raise HTTPException(status_code=500, detail=f"圖片生成失敗: {str(e)}")


class SendMessageRequest(BaseModel):
    user_id: int
    character_id: int
    message: str


@app.post("/api/v2/send-message")
async def send_message_v2(
    request: SendMessageRequest,
    db: Session = Depends(get_db)
) -> Dict:
    """
    Phase 2: Send message with conversation history and favorability tracking

    Args:
        request: Request body with user_id, character_id, and message
        db: Database session

    Returns:
        Character's response with favorability info
    """
    try:
        # Initialize conversation manager
        conv_manager = ConversationManager(db, api_client)

        # Send message and get response
        result = conv_manager.send_message(
            user_id=request.user_id,
            character_id=request.character_id,
            user_message=request.message
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"發送訊息失敗: {str(e)}")


@app.get("/api/v2/conversation-history/{character_id}")
async def get_conversation_history(
    character_id: int,
    limit: Optional[int] = 50,
    db: Session = Depends(get_db)
) -> Dict:
    """
    Get conversation history for a character

    Args:
        character_id: Character ID
        limit: Maximum number of messages to return
        db: Database session

    Returns:
        List of messages
    """
    try:
        conv_manager = ConversationManager(db, api_client)
        messages = conv_manager.get_conversation_history(character_id, limit)

        return {
            "success": True,
            "character_id": character_id,
            "message_count": len(messages),
            "messages": [
                {
                    "message_id": msg.message_id,
                    "speaker_name": msg.speaker_name,
                    "content": msg.message_content,
                    "timestamp": msg.timestamp.isoformat(),
                    "favorability_level": msg.favorability_level
                }
                for msg in messages
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取歷史失敗: {str(e)}")


@app.get("/api/v2/user-characters/{user_id}")
async def get_user_characters(user_id: int, db: Session = Depends(get_db)) -> Dict:
    """
    Get all characters for a user

    Args:
        user_id: User ID
        db: Database session

    Returns:
        List of characters
    """
    try:
        conv_manager = ConversationManager(db, api_client)
        characters = conv_manager.get_user_characters(user_id)

        return {
            "success": True,
            "user_id": user_id,
            "character_count": len(characters),
            "characters": [
                {
                    "character_id": char.character_id,
                    "name": char.name,
                    "nickname": char.nickname,
                    "created_at": char.created_at.isoformat(),
                    "favorability": conv_manager.get_favorability(char.character_id).current_level
                    if conv_manager.get_favorability(char.character_id) else 1
                }
                for char in characters
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取角色列表失敗: {str(e)}")


@app.get("/api/v2/favorability/{character_id}")
async def get_favorability_status(character_id: int, db: Session = Depends(get_db)) -> Dict:
    """
    Get favorability status for a character

    Args:
        character_id: Character ID
        db: Database session

    Returns:
        Favorability information
    """
    try:
        conv_manager = ConversationManager(db, api_client)
        favorability = conv_manager.get_favorability(character_id)

        if not favorability:
            raise HTTPException(status_code=404, detail="好感度記錄不存在")

        return {
            "success": True,
            "character_id": character_id,
            "current_level": favorability.current_level,
            "message_count": favorability.message_count,
            "last_updated": favorability.last_updated.isoformat(),
            "progress": {
                "level_1_threshold": ConversationManager.LEVEL_1_THRESHOLD,
                "level_2_threshold": ConversationManager.LEVEL_2_THRESHOLD,
                "level_3_threshold": ConversationManager.LEVEL_3_THRESHOLD
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取好感度失敗: {str(e)}")


@app.get("/api/v2/character-profile/{character_id}")
async def get_character_profile(character_id: int, db: Session = Depends(get_db)) -> Dict:
    """
    Get complete character profile with detailed statistics

    Args:
        character_id: Character ID
        db: Database session

    Returns:
        Complete character profile including stats and favorability
    """
    try:
        conv_manager = ConversationManager(db, api_client)

        # Get character
        character = conv_manager.get_character(character_id)
        if not character:
            raise HTTPException(status_code=404, detail="角色未找到")

        # Get favorability
        favorability = conv_manager.get_favorability(character_id)

        # Get conversation statistics
        messages = conv_manager.get_conversation_history(character_id, limit=1000)

        # Calculate statistics
        total_messages = len(messages)
        user_messages = sum(1 for msg in messages if msg.speaker_name != character.name)
        character_messages = total_messages - user_messages

        first_message_date = messages[-1].timestamp.isoformat() if messages else None
        last_message_date = messages[0].timestamp.isoformat() if messages else None

        # Calculate conversation days
        conversation_days = 0
        if messages and len(messages) > 1:
            first_date = messages[-1].timestamp.date()
            last_date = messages[0].timestamp.date()
            conversation_days = (last_date - first_date).days + 1

        # Favorability progress
        if favorability:
            if favorability.current_level == 1:
                progress = min(100, (favorability.message_count / 20) * 100)
                next_level_at = 20
                level_name = "陌生期"
            elif favorability.current_level == 2:
                progress = min(100, ((favorability.message_count - 20) / 30) * 100)
                next_level_at = 50
                level_name = "熟悉期"
            else:
                progress = 100
                next_level_at = None
                level_name = "親密期"
        else:
            progress = 0
            next_level_at = 20
            level_name = "未知"

        # Parse other_setting to get background story
        import json
        other_setting = {}
        try:
            other_setting = json.loads(character.other_setting) if isinstance(character.other_setting, str) else character.other_setting
        except:
            pass

        return {
            "success": True,
            "character": {
                "character_id": character.character_id,
                "name": character.name,
                "nickname": character.nickname,
                "gender": character.gender,
                "identity": character.identity,
                "detail_setting": character.detail_setting,
                "background_story": other_setting.get("background_story", ""),
                "interests": other_setting.get("interests", []),
                "communication_style": other_setting.get("communication_style", ""),
                "created_at": character.created_at.isoformat()
            },
            "favorability": {
                "current_level": favorability.current_level if favorability else 1,
                "level_name": level_name,
                "message_count": favorability.message_count if favorability else 0,
                "progress_percentage": round(progress, 1),
                "next_level_at": next_level_at
            },
            "statistics": {
                "total_messages": total_messages,
                "user_messages": user_messages,
                "character_messages": character_messages,
                "conversation_days": conversation_days,
                "first_message_date": first_message_date,
                "last_message_date": last_message_date,
                "average_messages_per_day": round(total_messages / conversation_days, 1) if conversation_days > 0 else 0
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取角色資料失敗: {str(e)}")


@app.delete("/api/v2/delete-character/{character_id}")
async def delete_character_endpoint(character_id: int, db: Session = Depends(get_db)) -> Dict:
    """
    Delete a character and all associated data

    Args:
        character_id: Character ID to delete
        db: Database session

    Returns:
        Success status
    """
    try:
        conv_manager = ConversationManager(db, api_client)
        success = conv_manager.delete_character(character_id)

        if success:
            return {
                "success": True,
                "message": "角色已成功刪除"
            }
        else:
            raise HTTPException(status_code=404, detail="角色不存在")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刪除角色失敗: {str(e)}")


@app.put("/api/v2/update-character/{character_id}")
async def update_character_endpoint(
    character_id: int,
    character_data: Dict,
    db: Session = Depends(get_db)
) -> Dict:
    """
    Update character settings

    Args:
        character_id: Character ID to update
        character_data: Updated character data
        db: Database session

    Returns:
        Success status with updated character
    """
    try:
        # Get character from database
        character = db.query(Character).filter(
            Character.character_id == character_id
        ).first()

        if not character:
            raise HTTPException(status_code=404, detail="角色不存在")

        # Update allowed fields
        if "name" in character_data:
            character.name = character_data["name"]
        if "gender" in character_data:
            character.gender = character_data["gender"]
        if "identity" in character_data:
            character.identity = character_data["identity"]
        if "nickname" in character_data:
            character.nickname = character_data["nickname"]
        if "detail_setting" in character_data:
            character.detail_setting = character_data["detail_setting"]
        if "other_setting" in character_data:
            # Parse JSON string if provided as string
            if isinstance(character_data["other_setting"], str):
                import json
                character.other_setting = json.loads(character_data["other_setting"])
            else:
                character.other_setting = character_data["other_setting"]

        db.commit()
        db.refresh(character)

        return {
            "success": True,
            "message": "角色設定已更新",
            "character": {
                "character_id": character.character_id,
                "name": character.name,
                "gender": character.gender,
                "identity": character.identity,
                "nickname": character.nickname,
                "detail_setting": character.detail_setting,
                "other_setting": character.other_setting
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error updating character: {error_details}")
        raise HTTPException(status_code=500, detail=f"更新角色失敗: {str(e)}")


@app.post("/api/v2/create-knowledge-base/{character_id}")
async def create_knowledge_base_for_character(
    character_id: int,
    db: Session = Depends(get_db)
) -> Dict:
    """
    Create or update knowledge base for a character
    This will enable memory enhancement for better conversations

    Args:
        character_id: Character ID
        db: Database session

    Returns:
        Success status with knowledge base ID
    """
    try:
        from backend.knowledge_base import KnowledgeBaseManager

        # Get character
        character = db.query(Character).filter(
            Character.character_id == character_id
        ).first()

        if not character:
            raise HTTPException(status_code=404, detail="角色不存在")

        # Get user preferences
        user_prefs = db.query(UserPreference).filter(
            UserPreference.user_id == character.user_id
        ).all()

        # Build preferences dict
        preferences_dict = {}
        for pref in user_prefs:
            if pref.category not in preferences_dict:
                preferences_dict[pref.category] = pref.content

        # Create knowledge base manager
        kb_manager = KnowledgeBaseManager(api_client)

        # Create or update knowledge base
        if character.knowledge_base_id:
            # Update existing
            success = kb_manager.update_character_knowledge(
                knowledge_base_id=character.knowledge_base_id,
                character_name=character.name,
                user_preferences=preferences_dict,
                background_info=character.detail_setting
            )
            if success:
                message = "知識庫已更新"
                kb_id = character.knowledge_base_id
            else:
                return {
                    "success": False,
                    "message": "知識庫更新失敗"
                }
        else:
            # Create new
            kb_id = kb_manager.create_character_knowledge(
                character_name=character.name,
                user_preferences=preferences_dict,
                background_info=character.detail_setting
            )

            if kb_id:
                # Save knowledge base ID to character
                character.knowledge_base_id = kb_id
                db.commit()
                message = "知識庫已建立"
            else:
                return {
                    "success": False,
                    "message": "知識庫建立失敗"
                }

        return {
            "success": True,
            "message": message,
            "knowledge_base_id": kb_id
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error creating knowledge base: {error_details}")
        raise HTTPException(status_code=500, detail=f"知識庫操作失敗: {str(e)}")


@app.get("/api/v2/export-conversation/{character_id}")
async def export_conversation(
    character_id: int,
    format: str = "txt",
    db: Session = Depends(get_db)
):
    """
    Export conversation history in JSON or TXT format

    Args:
        character_id: Character ID
        format: Export format ('json' or 'txt')
        db: Database session

    Returns:
        File download with conversation history
    """
    try:
        from fastapi.responses import Response
        import json
        from datetime import datetime

        conv_manager = ConversationManager(db, api_client)

        # Get character
        character = conv_manager.get_character(character_id)
        if not character:
            raise HTTPException(status_code=404, detail="角色未找到")

        # Get favorability
        favorability = conv_manager.get_favorability(character_id)

        # Get all conversation history
        messages = conv_manager.get_conversation_history(character_id, limit=10000)

        # Calculate statistics
        total_messages = len(messages)
        conversation_days = 0
        if messages and len(messages) > 1:
            first_date = messages[-1].timestamp.date()
            last_date = messages[0].timestamp.date()
            conversation_days = (last_date - first_date).days + 1

        # Parse other_setting for background story
        other_setting = {}
        try:
            other_setting = json.loads(character.other_setting) if isinstance(character.other_setting, str) else character.other_setting
        except:
            pass

        if format.lower() == "json":
            # Export as JSON
            export_data = {
                "export_info": {
                    "export_date": datetime.now().isoformat(),
                    "character_id": character_id,
                    "total_messages": total_messages,
                    "conversation_days": conversation_days
                },
                "character": {
                    "name": character.name,
                    "nickname": character.nickname,
                    "gender": character.gender,
                    "identity": character.identity,
                    "personality": character.detail_setting,
                    "background_story": other_setting.get("background_story", ""),
                    "interests": other_setting.get("interests", [])
                },
                "favorability": {
                    "level": favorability.current_level if favorability else 1,
                    "level_name": "陌生期" if not favorability or favorability.current_level == 1 else ("熟悉期" if favorability.current_level == 2 else "親密期"),
                    "message_count": favorability.message_count if favorability else 0
                },
                "messages": [
                    {
                        "timestamp": msg.timestamp.isoformat(),
                        "speaker": msg.speaker_name,
                        "content": msg.message_content,
                        "favorability_level": msg.favorability_level
                    }
                    for msg in reversed(messages)  # Reverse to chronological order
                ]
            }

            content = json.dumps(export_data, ensure_ascii=False, indent=2)
            filename = f"{character.name}_對話記錄_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            media_type = "application/json"

        else:  # TXT format
            lines = []
            lines.append("=" * 60)
            lines.append(f"💕 {character.name} 的對話記錄")
            lines.append("=" * 60)
            lines.append(f"\n📊 統計資訊：")
            lines.append(f"   總訊息數：{total_messages} 條")
            lines.append(f"   對話天數：{conversation_days} 天")
            lines.append(f"   好感度等級：{favorability.current_level if favorability else 1} - {'陌生期' if not favorability or favorability.current_level == 1 else ('熟悉期' if favorability.current_level == 2 else '親密期')}")
            lines.append(f"\n✨ 角色資訊：")
            lines.append(f"   名字：{character.name} ({character.nickname})")
            lines.append(f"   性別：{character.gender}")
            lines.append(f"   身份：{character.identity}")
            lines.append(f"   性格：{character.detail_setting}")
            if other_setting.get("background_story"):
                lines.append(f"   背景故事：{other_setting['background_story']}")

            lines.append(f"\n" + "=" * 60)
            lines.append("💬 對話內容")
            lines.append("=" * 60 + "\n")

            for msg in reversed(messages):  # Chronological order
                timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                lines.append(f"[{timestamp}] {msg.speaker_name}：")
                lines.append(f"  {msg.message_content}\n")

            lines.append("=" * 60)
            lines.append(f"匯出時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("🤖 Generated with Claude Code")
            lines.append("=" * 60)

            content = "\n".join(lines)
            filename = f"{character.name}_對話記錄_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            media_type = "text/plain; charset=utf-8"

        return Response(
            content=content.encode('utf-8'),
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"匯出失敗: {str(e)}")


@app.get("/api/v2/analytics/{character_id}")
async def get_analytics(
    character_id: int,
    db: Session = Depends(get_db)
):
    """
    Get analytics and statistics for a character's conversations

    Args:
        character_id: Character ID
        db: Database session

    Returns:
        Analytics data including message trends, favorability progression, etc.
    """
    try:
        from datetime import datetime, timedelta
        from collections import defaultdict
        import json

        conv_manager = ConversationManager(db, api_client)

        # Get character
        character = conv_manager.get_character(character_id)
        if not character:
            raise HTTPException(status_code=404, detail="角色未找到")

        # Get favorability
        favorability = conv_manager.get_favorability(character_id)

        # Get all messages
        messages = conv_manager.get_conversation_history(character_id, limit=10000)

        if not messages:
            return {
                "success": True,
                "character_id": character_id,
                "total_messages": 0,
                "analytics": {}
            }

        # Calculate basic statistics
        total_messages = len(messages)
        user_messages = sum(1 for msg in messages if msg.speaker_name != character.name)
        character_messages = total_messages - user_messages

        # Time-based statistics
        first_message_time = messages[-1].timestamp
        last_message_time = messages[0].timestamp
        conversation_days = (last_message_time.date() - first_message_time.date()).days + 1

        # Messages by day
        messages_by_day = defaultdict(int)
        for msg in messages:
            date_key = msg.timestamp.date().isoformat()
            messages_by_day[date_key] += 1

        # Messages by hour of day
        messages_by_hour = defaultdict(int)
        for msg in messages:
            hour = msg.timestamp.hour
            messages_by_hour[hour] += 1

        # Favorability progression
        favorability_progression = []
        current_level = 1
        for i, msg in enumerate(reversed(messages), 1):
            # Simulate favorability level progression based on message count
            if i >= 50:
                level = 3
            elif i >= 20:
                level = 2
            else:
                level = 1

            if level != current_level:
                favorability_progression.append({
                    "message_count": i,
                    "level": level,
                    "timestamp": msg.timestamp.isoformat(),
                    "level_name": "陌生期" if level == 1 else ("熟悉期" if level == 2 else "親密期")
                })
                current_level = level

        # Daily message trends (last 30 days)
        today = datetime.now().date()
        daily_trends = []
        for i in range(29, -1, -1):
            date = today - timedelta(days=i)
            date_key = date.isoformat()
            count = messages_by_day.get(date_key, 0)
            daily_trends.append({
                "date": date_key,
                "message_count": count
            })

        # Most active hours
        top_hours = sorted(messages_by_hour.items(), key=lambda x: x[1], reverse=True)[:5]
        most_active_hours = [
            {
                "hour": hour,
                "message_count": count,
                "time_range": f"{hour}:00-{hour+1}:00"
            }
            for hour, count in top_hours
        ]

        # Average response time (simplified - just average messages per day)
        avg_messages_per_day = total_messages / conversation_days if conversation_days > 0 else 0

        # Longest streak (consecutive days with messages)
        dates_with_messages = sorted(set(msg.timestamp.date() for msg in messages))
        longest_streak = 1
        current_streak = 1
        for i in range(1, len(dates_with_messages)):
            if (dates_with_messages[i] - dates_with_messages[i-1]).days == 1:
                current_streak += 1
                longest_streak = max(longest_streak, current_streak)
            else:
                current_streak = 1

        return {
            "success": True,
            "character_id": character_id,
            "character_name": character.name,
            "total_messages": total_messages,
            "analytics": {
                "overview": {
                    "total_messages": total_messages,
                    "user_messages": user_messages,
                    "character_messages": character_messages,
                    "conversation_days": conversation_days,
                    "first_message": first_message_time.isoformat(),
                    "last_message": last_message_time.isoformat(),
                    "avg_messages_per_day": round(avg_messages_per_day, 1),
                    "longest_streak_days": longest_streak
                },
                "favorability": {
                    "current_level": favorability.current_level if favorability else 1,
                    "current_level_name": "陌生期" if not favorability or favorability.current_level == 1 else ("熟悉期" if favorability.current_level == 2 else "親密期"),
                    "message_count": favorability.message_count if favorability else 0,
                    "progression": favorability_progression
                },
                "trends": {
                    "daily": daily_trends,
                    "most_active_hours": most_active_hours,
                    "messages_by_hour": [
                        {"hour": h, "count": messages_by_hour.get(h, 0)}
                        for h in range(24)
                    ]
                }
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取分析數據失敗: {str(e)}")


@app.get("/ui2")
async def ui2():
    """Phase 2 UI - User input and character generation with full persistence"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
        <meta http-equiv="Pragma" content="no-cache">
        <meta http-equiv="Expires" content="0">
        <title>戀愛聊天機器人 - 建立你的專屬伴侶 v2</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: "Microsoft YaHei", "微軟正黑體", sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 {
                color: #667eea;
                text-align: center;
                margin-bottom: 10px;
                font-size: 32px;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
                font-size: 16px;
            }
            .step {
                display: none;
            }
            .step.active {
                display: block;
                animation: fadeIn 0.5s;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                color: #333;
                font-weight: bold;
            }
            input[type="text"],
            textarea,
            select {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 14px;
                transition: border-color 0.3s;
            }
            input[type="text"]:focus,
            textarea:focus,
            select:focus {
                outline: none;
                border-color: #667eea;
            }
            textarea {
                resize: vertical;
                min-height: 80px;
            }
            .checkbox-group {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }
            .checkbox-item {
                flex: 0 0 calc(50% - 5px);
            }
            .checkbox-item input {
                margin-right: 5px;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 8px;
                font-size: 16px;
                cursor: pointer;
                transition: transform 0.2s;
                margin-top: 10px;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            .button-group {
                display: flex;
                gap: 10px;
                justify-content: space-between;
                margin-top: 20px;
            }
            .character-result {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 12px;
                margin-top: 20px;
            }
            .character-name {
                font-size: 24px;
                color: #667eea;
                margin-bottom: 10px;
            }
            .character-detail {
                margin: 10px 0;
                line-height: 1.6;
            }
            .chat-test {
                margin-top: 20px;
                padding: 20px;
                background: #fff;
                border: 2px solid #e0e0e0;
                border-radius: 12px;
            }
            #chatMessages {
                max-height: 400px;
                overflow-y: auto;
                padding: 10px;
            }
            .message {
                padding: 10px;
                margin: 10px 0;
                border-radius: 8px;
            }
            .message.user {
                background: #e3f2fd;
                text-align: right;
            }
            .message.character {
                background: #f3e5f5;
            }
            .typing-indicator {
                display: flex;
                align-items: center;
                padding: 10px;
                margin: 10px 0;
                background: #f3e5f5;
                border-radius: 8px;
                width: fit-content;
            }
            .typing-indicator span {
                height: 8px;
                width: 8px;
                margin: 0 2px;
                background-color: #9e9e9e;
                display: inline-block;
                border-radius: 50%;
                animation: typing 1.4s infinite;
            }
            .typing-indicator span:nth-child(2) {
                animation-delay: 0.2s;
            }
            .typing-indicator span:nth-child(3) {
                animation-delay: 0.4s;
            }
            @keyframes typing {
                0%, 60%, 100% {
                    transform: translateY(0);
                    opacity: 0.7;
                }
                30% {
                    transform: translateY(-10px);
                    opacity: 1;
                }
            }
            .level-up-notification,
            .special-event-notification {
                padding: 15px;
                margin: 15px 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 12px;
                text-align: center;
                font-weight: bold;
                animation: slideIn 0.5s ease-out, glow 1.5s ease-in-out;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }
            .special-event-notification.milestone {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4);
            }
            .special-event-notification.anniversary {
                background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                box-shadow: 0 4px 15px rgba(250, 112, 154, 0.4);
            }
            .special-event-notification.level-up {
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
            }
            @keyframes slideIn {
                from {
                    transform: translateY(-20px);
                    opacity: 0;
                }
                to {
                    transform: translateY(0);
                    opacity: 1;
                }
            }
            @keyframes glow {
                0%, 100% {
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                }
                50% {
                    box-shadow: 0 4px 25px rgba(102, 126, 234, 0.8), 0 0 30px rgba(255, 255, 255, 0.5);
                }
            }
            .profile-button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                transition: transform 0.2s, box-shadow 0.2s;
                margin-right: 10px;
            }
            .profile-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }
            .loading {
                text-align: center;
                color: #667eea;
                font-size: 18px;
                padding: 20px;
            }

            /* Mobile Responsive Styles */
            @media (max-width: 768px) {
                .container {
                    padding: 15px;
                    margin: 10px auto;
                }
                h1 {
                    font-size: 24px;
                }
                h2 {
                    font-size: 20px;
                }
                .form-group input,
                .form-group textarea,
                .form-group select {
                    font-size: 16px; /* Prevents zoom on iOS */
                }
                .button-group {
                    flex-direction: column;
                }
                .button-group button,
                .profile-button {
                    width: 100%;
                    margin: 5px 0;
                }
                #chatMessages {
                    max-height: 300px;
                }
                .character-result {
                    font-size: 14px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>💕 戀愛聊天機器人 [Phase 2]</h1>
            <p class="subtitle">建立你的專屬AI伴侶 - 完整持久化版本</p>

            <!-- Step 1: Basic Info -->
            <div id="step1" class="step active">
                <h2>第一步：基本資料</h2>
                <div class="form-group">
                    <label>你的名字：</label>
                    <input type="text" id="userName" placeholder="請輸入你的名字">
                </div>
                <div class="form-group">
                    <label>你是男生還是女生？</label>
                    <select id="userGender">
                        <option value="">請選擇</option>
                        <option value="男">男生</option>
                        <option value="女">女生</option>
                        <option value="其他">其他</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>你喜歡男生還是女生？</label>
                    <select id="userPreference" onchange="updateCharacterOptions()">
                        <option value="">請選擇</option>
                        <option value="男">男生</option>
                        <option value="女">女生</option>
                        <option value="都可以">都可以</option>
                    </select>
                </div>
                <div class="button-group">
                    <div></div>
                    <button onclick="nextStep(2)">下一步</button>
                </div>
            </div>

            <!-- Step 2: Dream Type -->
            <div id="step2" class="step">
                <h2>第二步：描述你的理想伴侶</h2>

                <div class="form-group">
                    <label>角色名字：</label>
                    <input type="text" id="characterName" placeholder="例如：雨柔、思涵、嘉欣">
                </div>

                <div class="form-group">
                    <label>說話風格：</label>
                    <select id="talkingStyle">
                        <option value="溫柔體貼">溫柔體貼</option>
                        <option value="活潑開朗">活潑開朗</option>
                        <option value="知性優雅">知性優雅</option>
                        <option value="可愛俏皮">可愛俏皮</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>性格特質（可多選）：</label>
                    <div class="checkbox-group" id="traitsContainer">
                        <div class="checkbox-item">
                            <input type="checkbox" id="trait1" value="溫柔">
                            <label for="trait1" style="display:inline">溫柔</label>
                        </div>
                        <div class="checkbox-item">
                            <input type="checkbox" id="trait2" value="活潑">
                            <label for="trait2" style="display:inline">活潑</label>
                        </div>
                        <div class="checkbox-item">
                            <input type="checkbox" id="trait3" value="體貼">
                            <label for="trait3" style="display:inline">體貼</label>
                        </div>
                        <div class="checkbox-item">
                            <input type="checkbox" id="trait4" value="幽默">
                            <label for="trait4" style="display:inline">幽默</label>
                        </div>
                        <div class="checkbox-item">
                            <input type="checkbox" id="trait5" value="知性">
                            <label for="trait5" style="display:inline">知性</label>
                        </div>
                        <div class="checkbox-item">
                            <input type="checkbox" id="trait6" value="可愛">
                            <label for="trait6" style="display:inline">可愛</label>
                        </div>
                    </div>
                </div>

                <div class="form-group">
                    <label>興趣愛好（用逗號分隔）：</label>
                    <input type="text" id="interests" placeholder="例如：音樂、電影、旅行">
                </div>

                <div class="form-group">
                    <label>年齡範圍：</label>
                    <input type="text" id="ageRange" placeholder="例如：20-25">
                </div>

                <div class="form-group">
                    <label>職業背景：</label>
                    <input type="text" id="occupation" placeholder="例如：學生、上班族">
                </div>

                <div class="form-group">
                    <label>外貌特徵（用於生成角色圖片）：</label>
                    <textarea id="appearance" placeholder="例如：長黑髮、大眼睛、白皙皮膚、甜美笑容、穿著休閒服飾..." rows="3"></textarea>
                    <small style="color: #666;">💡 描述角色的髮色、髮型、眼睛、膚色、身高、穿著風格等外貌特徵，用於生成2D動漫風格的角色圖片</small>
                </div>

                <div class="button-group">
                    <button onclick="prevStep(1)">上一步</button>
                    <button onclick="nextStep(3)">下一步</button>
                </div>
            </div>

            <!-- Step 3: Custom Memory -->
            <div id="step3" class="step">
                <h2>第三步：告訴我關於你自己</h2>

                <div class="form-group">
                    <label>你喜歡的事物：</label>
                    <textarea id="likes" placeholder="例如：喜歡喝咖啡、喜歡看電影、喜歡運動..."></textarea>
                </div>

                <div class="form-group">
                    <label>你不喜歡的事物：</label>
                    <textarea id="dislikes" placeholder="例如：不喜歡吵鬧的環境、不喜歡熬夜..."></textarea>
                </div>

                <div class="form-group">
                    <label>你的生活習慣：</label>
                    <textarea id="habits" placeholder="例如：早睡早起、喜歡規律作息..."></textarea>
                </div>

                <div class="form-group">
                    <label>你的職業/愛好：</label>
                    <textarea id="background" placeholder="例如：我是軟體工程師，平時喜歡寫程式..."></textarea>
                </div>

                <div class="button-group">
                    <button onclick="prevStep(2)">上一步</button>
                    <button onclick="generateCharacter()">生成我的專屬伴侶</button>
                </div>
            </div>

            <!-- Step 4: Character Result -->
            <div id="step4" class="step">
                <h2>你的專屬AI伴侶</h2>
                <div id="characterResult" class="character-result"></div>

                <div class="chat-test">
                    <h3>試著和她聊聊天吧！</h3>
                    <div id="chatMessages"></div>
                    <div class="form-group" style="margin-top: 15px;">
                        <input type="text" id="userMessage" placeholder="輸入你想說的話..." onkeypress="if(event.key==='Enter') sendMessage()">
                        <button onclick="sendMessage()" style="width: 100%; margin-top: 10px;">發送</button>
                    </div>
                </div>

                <div class="button-group" style="margin-top: 20px;">
                    <button class="profile-button" onclick="viewProfile()">📊 查看角色檔案</button>
                    <button class="profile-button" onclick="viewCharacters()">💕 角色管理</button>
                    <button onclick="location.reload()">重新開始</button>
                </div>
            </div>
        </div>

        <script>
            let currentStep = 1;
            let generatedCharacter = null;
            let userId = null;
            let characterId = null;
            let favorabilityLevel = 1;
            let messageCount = 0;

            // Gender-specific options
            const femaleTraits = [
                {value: '溫柔', label: '溫柔'},
                {value: '活潑', label: '活潑'},
                {value: '體貼', label: '體貼'},
                {value: '幽默', label: '幽默'},
                {value: '知性', label: '知性'},
                {value: '可愛', label: '可愛'}
            ];

            const maleTraits = [
                {value: '成熟穩重', label: '成熟穩重'},
                {value: '陽光開朗', label: '陽光開朗'},
                {value: '溫柔體貼', label: '溫柔體貼'},
                {value: '霸氣強勢', label: '霸氣強勢'},
                {value: '幽默風趣', label: '幽默風趣'},
                {value: '斯文知性', label: '斯文知性'}
            ];

            const femaleTalkingStyles = [
                {value: '溫柔體貼', label: '溫柔體貼'},
                {value: '活潑開朗', label: '活潑開朗'},
                {value: '知性優雅', label: '知性優雅'},
                {value: '可愛俏皮', label: '可愛俏皮'}
            ];

            const maleTalkingStyles = [
                {value: '成熟穩重', label: '成熟穩重'},
                {value: '陽光活潑', label: '陽光活潑'},
                {value: '溫柔紳士', label: '溫柔紳士'},
                {value: '霸氣強勢', label: '霸氣強勢'},
                {value: '知性優雅', label: '知性優雅'},
                {value: '幽默風趣', label: '幽默風趣'}
            ];

            function updateCharacterOptions() {
                const preference = document.getElementById('userPreference').value;
                const traitsContainer = document.getElementById('traitsContainer');
                const talkingStyleSelect = document.getElementById('talkingStyle');

                if (!preference || preference === '都可以') {
                    // Default to female options
                    updateTraits(femaleTraits);
                    updateTalkingStyles(femaleTalkingStyles);
                } else if (preference === '男') {
                    // Male character options
                    updateTraits(maleTraits);
                    updateTalkingStyles(maleTalkingStyles);
                } else {
                    // Female character options
                    updateTraits(femaleTraits);
                    updateTalkingStyles(femaleTalkingStyles);
                }
            }

            function updateTraits(traits) {
                const container = document.getElementById('traitsContainer');
                container.innerHTML = '';
                traits.forEach((trait, index) => {
                    const div = document.createElement('div');
                    div.className = 'checkbox-item';
                    div.innerHTML = `
                        <input type="checkbox" id="trait${index + 1}" value="${trait.value}">
                        <label for="trait${index + 1}" style="display:inline">${trait.label}</label>
                    `;
                    container.appendChild(div);
                });
            }

            function updateTalkingStyles(styles) {
                const select = document.getElementById('talkingStyle');
                select.innerHTML = '';
                styles.forEach(style => {
                    const option = document.createElement('option');
                    option.value = style.value;
                    option.textContent = style.label;
                    select.appendChild(option);
                });
            }

            function nextStep(step) {
                // Validate current step
                if (step === 2) {
                    if (!document.getElementById('userName').value) {
                        alert('請輸入你的名字');
                        return;
                    }
                    if (!document.getElementById('userGender').value) {
                        alert('請選擇你的性別');
                        return;
                    }
                    if (!document.getElementById('userPreference').value) {
                        alert('請選擇你喜歡的性別');
                        return;
                    }
                }

                document.getElementById('step' + currentStep).classList.remove('active');
                document.getElementById('step' + step).classList.add('active');
                currentStep = step;
            }

            function prevStep(step) {
                document.getElementById('step' + currentStep).classList.remove('active');
                document.getElementById('step' + step).classList.add('active');
                currentStep = step;
            }

            function getSelectedTraits() {
                const traits = [];
                for (let i = 1; i <= 6; i++) {
                    const checkbox = document.getElementById('trait' + i);
                    if (checkbox.checked) {
                        traits.push(checkbox.value);
                    }
                }
                return traits;
            }

            async function generateCharacter() {
                const userName = document.getElementById('userName').value;
                const userGender = document.getElementById('userGender').value;
                const userPreference = document.getElementById('userPreference').value;
                const characterName = document.getElementById('characterName').value;
                const talkingStyle = document.getElementById('talkingStyle').value;
                const traits = getSelectedTraits();
                const interests = document.getElementById('interests').value.split('、').map(s => s.trim()).filter(s => s);
                const ageRange = document.getElementById('ageRange').value;
                const occupation = document.getElementById('occupation').value;
                const appearance = document.getElementById('appearance').value;
                const likes = document.getElementById('likes').value;
                const dislikes = document.getElementById('dislikes').value;
                const habits = document.getElementById('habits').value;
                const background = document.getElementById('background').value;

                if (traits.length === 0) {
                    alert('請至少選擇一個性格特質');
                    return;
                }

                const userProfile = {
                    user_name: userName,
                    user_gender: userGender,
                    user_preference: userPreference,
                    preferred_character_name: characterName,
                    appearance_description: appearance,
                    dream_type: {
                        personality_traits: traits,
                        physical_description: appearance,
                        age_range: ageRange,
                        interests: interests,
                        occupation: occupation,
                        talking_style: talkingStyle
                    },
                    custom_memory: {
                        likes: { general: likes.split('、').map(s => s.trim()).filter(s => s) },
                        dislikes: { general: dislikes.split('、').map(s => s.trim()).filter(s => s) },
                        habits: { general: habits },
                        personal_background: { general: background }
                    }
                };

                // Show loading
                document.getElementById('characterResult').innerHTML = '<div class="loading">正在生成你的專屬伴侶...</div>';
                nextStep(4);

                try {
                    const response = await fetch('/api/v2/create-character', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(userProfile)
                    });

                    const data = await response.json();

                    if (data.success) {
                        // Save Phase 2 data
                        userId = data.user_id;
                        characterId = data.character_id;
                        generatedCharacter = data.character;
                        favorabilityLevel = data.favorability_level;
                        messageCount = 0;

                        displayCharacter(data.character, data.initial_message);
                    } else {
                        alert('生成失敗：' + data.message);
                    }
                } catch (error) {
                    alert('發生錯誤：' + error.message);
                }
            }

            function displayCharacter(character, initialMessage) {
                // Parse other_setting to get background story
                let backgroundStory = '';
                try {
                    const otherSetting = typeof character.other_setting === 'string'
                        ? JSON.parse(character.other_setting)
                        : character.other_setting;
                    backgroundStory = otherSetting.background_story || '';
                } catch (e) {
                    console.error('Failed to parse other_setting:', e);
                }

                // Favorability level display
                const favorabilityText = favorabilityLevel === 1 ? '陌生期 (Level 1)' :
                                        favorabilityLevel === 2 ? '熟悉期 (Level 2)' :
                                        '親密期 (Level 3)';
                const favorabilityColor = favorabilityLevel === 1 ? '#9e9e9e' :
                                         favorabilityLevel === 2 ? '#ff9800' :
                                         '#e91e63';

                const html = `
                    <div class="character-name">💕 ${character.name} (${character.nickname})</div>
                    <div class="character-detail"><strong>身份：</strong>${character.identity || '神秘'}</div>
                    <div class="character-detail"><strong>性格：</strong>${character.detail_setting}</div>
                    <div class="character-detail" style="background: ${favorabilityColor}15; padding: 10px; border-radius: 8px; border-left: 4px solid ${favorabilityColor};"><strong>💗 好感度：</strong><span style="color: ${favorabilityColor}; font-weight: bold;">${favorabilityText}</span> <span style="font-size: 12px; color: #666;">(訊息數: ${messageCount})</span></div>
                    ${backgroundStory ? `<div class="character-detail" style="background: #fff3e0; padding: 15px; border-radius: 8px; margin-top: 15px;"><strong>✨ 她的故事：</strong><br/><div style="margin-top: 8px; line-height: 1.8;">${backgroundStory}</div></div>` : ''}
                    <div class="character-detail" style="margin-top: 15px;"><strong>初次見面：</strong>${initialMessage}</div>
                `;
                document.getElementById('characterResult').innerHTML = html;

                // Display initial message in chat
                displayMessage(character.name, initialMessage, 'character');
            }

            function displayMessage(sender, content, type) {
                const chatMessages = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message ' + type;
                messageDiv.innerHTML = `<strong>${sender}：</strong>${content}`;
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }

            async function sendMessage() {
                const input = document.getElementById('userMessage');
                const message = input.value.trim();

                if (!message) return;

                const userName = document.getElementById('userName').value;
                displayMessage(userName, message, 'user');
                input.value = '';

                // Show typing indicator
                const loadingDiv = document.createElement('div');
                loadingDiv.id = 'loading-indicator';
                loadingDiv.className = 'typing-indicator';
                loadingDiv.innerHTML = '<span></span><span></span><span></span>';
                document.getElementById('chatMessages').appendChild(loadingDiv);
                document.getElementById('chatMessages').scrollTop = document.getElementById('chatMessages').scrollHeight;

                try {
                    const response = await fetch('/api/v2/send-message', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            user_id: userId,
                            character_id: characterId,
                            message: message
                        })
                    });

                    const data = await response.json();

                    // Remove loading indicator
                    const loading = document.getElementById('loading-indicator');
                    if (loading) loading.remove();

                    if (data.success) {
                        displayMessage(generatedCharacter.name, data.reply, 'character');

                        // Update favorability info
                        favorabilityLevel = data.favorability_level;
                        messageCount = data.message_count;

                        // Show special event messages
                        if (data.special_messages && data.special_messages.length > 0) {
                            data.special_messages.forEach(event => {
                                const notification = document.createElement('div');
                                notification.className = 'special-event-notification';

                                let icon = '';
                                if (event.type === 'milestone') {
                                    icon = '🎊';
                                    notification.classList.add('milestone');
                                } else if (event.type === 'anniversary') {
                                    icon = '🎂';
                                    notification.classList.add('anniversary');
                                } else if (event.type === 'level_up') {
                                    icon = '🎉';
                                    notification.classList.add('level-up');
                                }

                                notification.innerHTML = `${icon} ${event.message}`;
                                document.getElementById('chatMessages').appendChild(notification);
                            });

                            document.getElementById('chatMessages').scrollTop = document.getElementById('chatMessages').scrollHeight;
                        }

                        // Update favorability display
                        updateFavorabilityDisplay();
                    } else {
                        alert('發送失敗');
                    }
                } catch (error) {
                    // Remove loading indicator if error occurs
                    const loading = document.getElementById('loading-indicator');
                    if (loading) loading.remove();
                    alert('發生錯誤：' + error.message);
                }
            }

            function updateFavorabilityDisplay() {
                // Update the favorability display in character result
                const favorabilityText = favorabilityLevel === 1 ? '陌生期 (Level 1)' :
                                        favorabilityLevel === 2 ? '熟悉期 (Level 2)' :
                                        '親密期 (Level 3)';
                const favorabilityColor = favorabilityLevel === 1 ? '#9e9e9e' :
                                         favorabilityLevel === 2 ? '#ff9800' :
                                         '#e91e63';

                // Re-render character with updated favorability
                displayCharacter(generatedCharacter, '');
            }

            function viewProfile() {
                if (characterId) {
                    window.location.href = `/profile?character_id=${characterId}`;
                } else {
                    alert('請先生成角色！');
                }
            }

            function viewCharacters() {
                window.location.href = '/characters';
            }
        </script>
    </body>
    </html>
    """

    return HTMLResponse(
        content=html_content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.get("/profile")
async def character_profile_page():
    """Character Profile View - displays complete character information and statistics"""
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <title>角色檔案 - 戀愛聊天機器人</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: "Microsoft YaHei", "微軟正黑體", sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        .profile-card {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .character-name {
            font-size: 36px;
            color: #667eea;
            margin-bottom: 10px;
        }
        .nickname {
            font-size: 18px;
            color: #666;
            font-style: italic;
        }
        .section {
            margin: 30px 0;
        }
        .section-title {
            font-size: 20px;
            color: #333;
            margin-bottom: 15px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .favorability-container {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
        }
        .favorability-level {
            text-align: center;
            font-size: 24px;
            margin-bottom: 15px;
        }
        .level-1 { color: #9e9e9e; }
        .level-2 { color: #ff9800; }
        .level-3 { color: #e91e63; }
        .progress-bar-container {
            background: #ddd;
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            position: relative;
        }
        .progress-bar {
            height: 100%;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        .progress-bar.level-1 { background: linear-gradient(90deg, #9e9e9e, #bdbdbd); }
        .progress-bar.level-2 { background: linear-gradient(90deg, #ff9800, #ffa726); }
        .progress-bar.level-3 { background: linear-gradient(90deg, #e91e63, #ec407a); }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        .stat-value {
            font-size: 32px;
            color: #667eea;
            font-weight: bold;
        }
        .stat-label {
            font-size: 14px;
            color: #666;
            margin-top: 8px;
        }
        .background-story {
            background: #fff3e0;
            padding: 20px;
            border-radius: 12px;
            line-height: 1.8;
            color: #333;
        }
        .detail-row {
            padding: 12px 0;
            border-bottom: 1px solid #eee;
        }
        .detail-label {
            font-weight: bold;
            color: #667eea;
            margin-right: 10px;
        }
        .button {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border-radius: 8px;
            text-decoration: none;
            margin: 10px 5px;
            transition: transform 0.2s;
        }
        .button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .export-button {
            border: none;
            cursor: pointer;
            font-size: 14px;
            font-family: 'Microsoft JhengHei', 'PingFang TC', sans-serif;
        }
        .export-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .loading {
            text-align: center;
            padding: 60px;
            font-size: 20px;
            color: #667eea;
        }
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="profile-card">
            <div id="loading" class="loading">正在載入角色檔案...</div>
            <div id="content" style="display: none;">
                <div class="header">
                    <div class="character-name" id="characterName"></div>
                    <div class="nickname" id="nickname"></div>
                </div>

                <div class="section">
                    <div class="section-title">💗 好感度</div>
                    <div class="favorability-container">
                        <div class="favorability-level" id="favorabilityLevel"></div>
                        <div class="progress-bar-container">
                            <div class="progress-bar" id="progressBar"></div>
                        </div>
                        <div style="text-align: center; margin-top: 10px; color: #666;" id="progressText"></div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">📊 對話統計</div>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value" id="totalMessages">0</div>
                            <div class="stat-label">總訊息數</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="conversationDays">0</div>
                            <div class="stat-label">對話天數</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="avgMessages">0</div>
                            <div class="stat-label">平均每日訊息</div>
                        </div>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">✨ 角色資訊</div>
                    <div class="detail-row">
                        <span class="detail-label">身份：</span>
                        <span id="identity"></span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">性格：</span>
                        <span id="detailSetting"></span>
                    </div>
                    <div class="detail-row">
                        <span class="detail-label">興趣：</span>
                        <span id="interests"></span>
                    </div>
                </div>

                <div class="section">
                    <div class="section-title">📖 角色背景</div>
                    <div class="background-story" id="backgroundStory"></div>
                </div>

                <div style="text-align: center; margin-top: 30px;">
                    <button class="button" onclick="viewAnalytics()" style="margin-right: 10px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">📊 數據分析</button>
                    <button class="button export-button" onclick="exportConversation('txt')" style="margin-right: 10px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">📥 匯出為TXT</button>
                    <button class="button export-button" onclick="exportConversation('json')" style="margin-right: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">📥 匯出為JSON</button>
                    <a href="/ui2" class="button">返回聊天</a>
                </div>
            </div>
            <div id="error" class="error" style="display: none;"></div>
        </div>
    </div>

    <script>
        async function loadProfile() {
            const urlParams = new URLSearchParams(window.location.search);
            const characterId = urlParams.get('character_id');

            if (!characterId) {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('error').textContent = '錯誤：未提供角色ID。請從聊天頁面訪問。';
                document.getElementById('error').style.display = 'block';
                return;
            }

            try {
                const response = await fetch(`/api/v2/character-profile/${characterId}`);
                const data = await response.json();

                if (!data.success) {
                    throw new Error(data.error || '載入失敗');
                }

                // Hide loading, show content
                document.getElementById('loading').style.display = 'none';
                document.getElementById('content').style.display = 'block';

                // Fill in character info
                document.getElementById('characterName').textContent = `${data.character.name}`;
                document.getElementById('nickname').textContent = `（${data.character.nickname}）`;
                document.getElementById('identity').textContent = data.character.identity;
                document.getElementById('detailSetting').textContent = data.character.detail_setting;
                document.getElementById('interests').textContent = data.character.interests.join('、') || '無';
                document.getElementById('backgroundStory').textContent = data.character.background_story || '暫無背景故事';

                // Favorability
                const fav = data.favorability;
                const favLevel = document.getElementById('favorabilityLevel');
                favLevel.textContent = `${fav.level_name} (Level ${fav.current_level})`;
                favLevel.className = `favorability-level level-${fav.current_level}`;

                const progressBar = document.getElementById('progressBar');
                progressBar.style.width = `${fav.progress_percentage}%`;
                progressBar.className = `progress-bar level-${fav.current_level}`;
                progressBar.textContent = `${fav.progress_percentage}%`;

                const progressText = document.getElementById('progressText');
                if (fav.next_level_at) {
                    progressText.textContent = `已交流 ${fav.message_count} 則訊息，距離下一級還需 ${fav.next_level_at - fav.message_count} 則`;
                } else {
                    progressText.textContent = `已達到最高好感度！共 ${fav.message_count} 則訊息`;
                }

                // Statistics
                document.getElementById('totalMessages').textContent = data.statistics.total_messages;
                document.getElementById('conversationDays').textContent = data.statistics.conversation_days;
                document.getElementById('avgMessages').textContent = data.statistics.average_messages_per_day;

            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('error').textContent = `載入失敗：${error.message}`;
                document.getElementById('error').style.display = 'block';
            }
        }

        function exportConversation(format) {
            const urlParams = new URLSearchParams(window.location.search);
            const characterId = urlParams.get('character_id');

            if (!characterId) {
                alert('找不到角色ID');
                return;
            }

            // Create download link
            const exportUrl = `/api/v2/export-conversation/${characterId}?format=${format}`;
            window.location.href = exportUrl;
        }

        function viewAnalytics() {
            const urlParams = new URLSearchParams(window.location.search);
            const characterId = urlParams.get('character_id');

            if (!characterId) {
                alert('找不到角色ID');
                return;
            }

            window.location.href = `/analytics?character_id=${characterId}`;
        }

        // Load profile on page load
        loadProfile();
    </script>
</body>
</html>
        """,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.get("/characters")
async def character_management():
    """Character Management Page - manage multiple characters"""
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <title>角色管理 - 戀愛聊天機器人</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: "Microsoft YaHei", "微軟正黑體", sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header-card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 30px;
            text-align: center;
        }
        .page-title {
            font-size: 32px;
            color: #667eea;
            margin-bottom: 10px;
        }
        .page-subtitle {
            color: #666;
            font-size: 16px;
        }
        .characters-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .character-card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s, box-shadow 0.3s;
            cursor: pointer;
            position: relative;
        }
        .character-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }
        .character-card.active {
            border: 3px solid #667eea;
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
        }
        .character-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 20px;
        }
        .character-info {
            flex: 1;
        }
        .character-name {
            font-size: 28px;
            color: #333;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .character-nickname {
            color: #666;
            font-size: 16px;
            font-style: italic;
        }
        .character-actions {
            display: flex;
            gap: 10px;
        }
        .icon-button {
            background: none;
            border: none;
            font-size: 20px;
            cursor: pointer;
            padding: 5px;
            transition: transform 0.2s;
        }
        .icon-button:hover {
            transform: scale(1.2);
        }
        .character-stats {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 12px;
            margin: 15px 0;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
        }
        .stat-label {
            color: #666;
        }
        .stat-value {
            font-weight: bold;
            color: #333;
        }
        .favorability-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
            color: white;
        }
        .favorability-badge.level-1 { background: #9e9e9e; }
        .favorability-badge.level-2 { background: #ff9800; }
        .favorability-badge.level-3 { background: #e91e63; }
        .character-buttons {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        .button {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 10px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        .button-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .button-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .button-secondary {
            background: #f8f9fa;
            color: #667eea;
        }
        .button-secondary:hover {
            background: #e9ecef;
        }
        .button-danger {
            background: #dc3545;
            color: white;
        }
        .button-danger:hover {
            background: #c82333;
        }
        .create-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border: 3px dashed #667eea;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 300px;
        }
        .create-card:hover {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        }
        .create-icon {
            font-size: 60px;
            color: #667eea;
            margin-bottom: 20px;
        }
        .create-text {
            font-size: 20px;
            color: #667eea;
            font-weight: bold;
        }
        .empty-state {
            background: white;
            border-radius: 20px;
            padding: 60px 30px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .empty-icon {
            font-size: 80px;
            margin-bottom: 20px;
        }
        .empty-title {
            font-size: 24px;
            color: #333;
            margin-bottom: 10px;
        }
        .empty-text {
            color: #666;
            margin-bottom: 30px;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: white;
            font-size: 24px;
        }
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }
        .back-button {
            display: inline-block;
            padding: 12px 30px;
            background: white;
            color: #667eea;
            text-decoration: none;
            border-radius: 25px;
            font-weight: bold;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: all 0.3s;
            margin-top: 20px;
        }
        .back-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-card">
            <div class="page-title">💕 角色管理</div>
            <div class="page-subtitle">管理你的所有角色，切換或建立新的對話夥伴</div>
        </div>

        <div id="loading" class="loading">
            載入角色中...
        </div>

        <div id="content" style="display: none;">
            <div class="characters-grid" id="charactersGrid"></div>
        </div>

        <div id="error" class="error" style="display: none;"></div>

        <div style="text-align: center;">
            <a href="/ui2" class="back-button">返回首頁</a>
        </div>
    </div>

    <script>
        let userId = null;

        async function loadCharacters() {
            // Get user ID from localStorage
            userId = localStorage.getItem('userId');

            if (!userId) {
                showError('請先建立一個角色！');
                document.getElementById('loading').style.display = 'none';
                return;
            }

            try {
                const response = await fetch(`/api/v2/user-characters/${userId}`);
                const data = await response.json();

                document.getElementById('loading').style.display = 'none';

                if (data.success) {
                    displayCharacters(data.characters);
                } else {
                    showError('載入角色失敗');
                }
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                showError('載入失敗: ' + error.message);
            }
        }

        function displayCharacters(characters) {
            const grid = document.getElementById('charactersGrid');
            const currentCharacterId = parseInt(localStorage.getItem('characterId'));

            if (characters.length === 0) {
                document.getElementById('content').innerHTML = `
                    <div class="empty-state">
                        <div class="empty-icon">🤖</div>
                        <div class="empty-title">還沒有角色</div>
                        <div class="empty-text">快來建立你的第一個專屬角色吧！</div>
                        <a href="/ui2" class="button button-primary" style="display: inline-block; text-decoration: none; padding: 15px 40px;">建立新角色</a>
                    </div>
                `;
                document.getElementById('content').style.display = 'block';
                return;
            }

            grid.innerHTML = characters.map(char => {
                const isActive = char.character_id === currentCharacterId;
                const levelClass = `level-${char.favorability}`;
                const levelText = char.favorability === 1 ? '陌生期' :
                                 char.favorability === 2 ? '熟悉期' : '親密期';
                const createdDate = new Date(char.created_at).toLocaleDateString('zh-TW');

                return `
                    <div class="character-card ${isActive ? 'active' : ''}" onclick="selectCharacter(${char.character_id})">
                        <div class="character-header">
                            <div class="character-info">
                                <div class="character-name">${char.name}</div>
                                ${char.nickname ? `<div class="character-nickname">${char.nickname}</div>` : ''}
                            </div>
                            ${isActive ? '<div style="font-size: 24px;">✨</div>' : ''}
                        </div>

                        <div class="character-stats">
                            <div class="stat-row">
                                <span class="stat-label">好感度</span>
                                <span class="favorability-badge ${levelClass}">${levelText}</span>
                            </div>
                            <div class="stat-row">
                                <span class="stat-label">建立時間</span>
                                <span class="stat-value">${createdDate}</span>
                            </div>
                        </div>

                        <div class="character-buttons">
                            <button class="button button-primary" onclick="event.stopPropagation(); chatWithCharacter(${char.character_id})">
                                💬 開始聊天
                            </button>
                            <button class="button button-secondary" onclick="event.stopPropagation(); viewProfile(${char.character_id})">
                                📋 查看檔案
                            </button>
                            <button class="button button-secondary" onclick="event.stopPropagation(); editCharacter(${char.character_id})">
                                ✏️ 編輯
                            </button>
                            <button class="button button-danger" onclick="event.stopPropagation(); deleteCharacter(${char.character_id}, '${char.name}')">
                                🗑️
                            </button>
                        </div>
                    </div>
                `;
            }).join('');

            // Add create new character card
            grid.innerHTML += `
                <div class="character-card create-card" onclick="createNewCharacter()">
                    <div class="create-icon">➕</div>
                    <div class="create-text">建立新角色</div>
                </div>
            `;

            document.getElementById('content').style.display = 'block';
        }

        function selectCharacter(characterId) {
            localStorage.setItem('characterId', characterId);
            loadCharacters(); // Reload to update active state
        }

        function chatWithCharacter(characterId) {
            localStorage.setItem('characterId', characterId);
            window.location.href = '/ui2';
        }

        function viewProfile(characterId) {
            window.location.href = `/profile?character_id=${characterId}`;
        }

        function editCharacter(characterId) {
            window.location.href = `/edit-character/${characterId}`;
        }

        async function deleteCharacter(characterId, characterName) {
            if (!confirm(`確定要刪除 ${characterName} 嗎？\n\n刪除後將無法恢復所有對話記錄！`)) {
                return;
            }

            try {
                const response = await fetch(`/api/v2/delete-character/${characterId}`, {
                    method: 'DELETE'
                });

                const data = await response.json();

                if (data.success) {
                    alert('角色已刪除');

                    // If deleted current character, clear from localStorage
                    if (parseInt(localStorage.getItem('characterId')) === characterId) {
                        localStorage.removeItem('characterId');
                        localStorage.removeItem('generatedCharacter');
                    }

                    loadCharacters(); // Reload list
                } else {
                    alert('刪除失敗: ' + (data.error || '未知錯誤'));
                }
            } catch (error) {
                alert('刪除失敗: ' + error.message);
            }
        }

        function createNewCharacter() {
            window.location.href = '/ui2';
        }

        function showError(message) {
            document.getElementById('error').textContent = message;
            document.getElementById('error').style.display = 'block';
        }

        // Load characters on page load
        loadCharacters();
    </script>
</body>
</html>
        """,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.get("/edit-character/{character_id}")
async def edit_character_page(character_id: int):
    """Character Editing Page - customize character settings"""
    return HTMLResponse(
        content=f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <title>編輯角色 - 戀愛聊天機器人</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: "Microsoft YaHei", "微軟正黑體", sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
        }}
        .header-card {{
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 30px;
            text-align: center;
        }}
        .page-title {{
            font-size: 32px;
            color: #667eea;
            margin-bottom: 10px;
        }}
        .page-subtitle {{
            color: #666;
            font-size: 16px;
        }}
        .edit-card {{
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        .form-group {{
            margin-bottom: 25px;
        }}
        .form-label {{
            display: block;
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
            font-size: 16px;
        }}
        .form-input {{
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 15px;
            transition: border-color 0.3s;
        }}
        .form-input:focus {{
            outline: none;
            border-color: #667eea;
        }}
        .form-textarea {{
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 15px;
            min-height: 120px;
            font-family: "Microsoft YaHei", "微軟正黑體", sans-serif;
            resize: vertical;
            transition: border-color 0.3s;
        }}
        .form-textarea:focus {{
            outline: none;
            border-color: #667eea;
        }}
        .char-count {{
            text-align: right;
            color: #999;
            font-size: 13px;
            margin-top: 5px;
        }}
        .char-count.warning {{
            color: #ff9800;
        }}
        .char-count.error {{
            color: #f44336;
        }}
        .button-group {{
            display: flex;
            gap: 15px;
            margin-top: 30px;
        }}
        .btn {{
            flex: 1;
            padding: 15px 30px;
            border: none;
            border-radius: 30px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }}
        .btn-primary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }}
        .btn-secondary {{
            background: #f0f0f0;
            color: #666;
        }}
        .btn-secondary:hover {{
            background: #e0e0e0;
        }}
        .loading {{
            display: none;
            text-align: center;
            padding: 20px;
        }}
        .loading.active {{
            display: block;
        }}
        .hint-text {{
            color: #999;
            font-size: 13px;
            margin-top: 5px;
        }}
        .select-input {{
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-size: 15px;
            background: white;
            cursor: pointer;
            transition: border-color 0.3s;
        }}
        .select-input:focus {{
            outline: none;
            border-color: #667eea;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header-card">
            <div class="page-title">✏️ 編輯角色</div>
            <div class="page-subtitle">自訂角色設定，打造你的專屬伴侶</div>
        </div>

        <div class="edit-card">
            <form id="editForm" onsubmit="event.preventDefault(); saveCharacter();">
                <div class="form-group">
                    <label class="form-label">角色名字 *</label>
                    <input type="text" id="name" class="form-input" maxlength="50" required>
                    <div class="char-count" id="nameCount">0 / 50</div>
                </div>

                <div class="form-group">
                    <label class="form-label">性別 *</label>
                    <select id="gender" class="select-input" required>
                        <option value="女">女</option>
                        <option value="男">男</option>
                        <option value="其他">其他</option>
                    </select>
                </div>

                <div class="form-group">
                    <label class="form-label">暱稱</label>
                    <input type="text" id="nickname" class="form-input" maxlength="50">
                    <div class="char-count" id="nicknameCount">0 / 50</div>
                    <div class="hint-text">例如：小雨雨、寶貝</div>
                </div>

                <div class="form-group">
                    <label class="form-label">身份背景</label>
                    <input type="text" id="identity" class="form-input" maxlength="200">
                    <div class="char-count" id="identityCount">0 / 200</div>
                    <div class="hint-text">例如：23歲大學生、25歲上班族</div>
                </div>

                <div class="form-group">
                    <label class="form-label">詳細設定 *</label>
                    <textarea id="detail_setting" class="form-textarea" maxlength="500" required></textarea>
                    <div class="char-count" id="detailCount">0 / 500</div>
                    <div class="hint-text">描述角色的性格特質、說話風格、行為模式等</div>
                </div>

                <div class="form-group">
                    <label class="form-label">興趣愛好</label>
                    <input type="text" id="interests" class="form-input">
                    <div class="hint-text">用逗號分隔，例如：音樂，閱讀，旅遊</div>
                </div>

                <div class="loading" id="loading">
                    <div>⏳ 正在保存...</div>
                </div>

                <div class="button-group">
                    <button type="button" class="btn btn-secondary" onclick="goBack()">取消</button>
                    <button type="submit" class="btn btn-primary">💾 保存變更</button>
                </div>
            </form>
        </div>
    </div>

    <script>
        const characterId = {character_id};

        // Character count tracking
        const fields = [
            {{ id: 'name', max: 50 }},
            {{ id: 'nickname', max: 50 }},
            {{ id: 'identity', max: 200 }},
            {{ id: 'detail_setting', max: 500 }}
        ];

        fields.forEach(field => {{
            const input = document.getElementById(field.id);
            const counter = document.getElementById(field.id + 'Count');

            input.addEventListener('input', () => {{
                const length = input.value.length;
                counter.textContent = `${{length}} / ${{field.max}}`;

                if (length >= field.max * 0.9) {{
                    counter.classList.add('warning');
                }} else {{
                    counter.classList.remove('warning');
                }}

                if (length >= field.max) {{
                    counter.classList.add('error');
                }} else {{
                    counter.classList.remove('error');
                }}
            }});
        }});

        async function loadCharacter() {{
            try {{
                const response = await fetch(`/api/v2/character-profile/${{characterId}}`);
                const data = await response.json();

                if (data.success) {{
                    const char = data.character;
                    document.getElementById('name').value = char.name || '';
                    document.getElementById('gender').value = char.gender || '女';
                    document.getElementById('nickname').value = char.nickname || '';
                    document.getElementById('identity').value = char.identity || '';
                    document.getElementById('detail_setting').value = char.detail_setting || '';

                    // Load interests from other_setting
                    if (char.other_setting && char.other_setting.interests) {{
                        document.getElementById('interests').value = char.other_setting.interests.join('，');
                    }}

                    // Trigger character count updates
                    fields.forEach(field => {{
                        const input = document.getElementById(field.id);
                        input.dispatchEvent(new Event('input'));
                    }});
                }} else {{
                    alert('載入角色資料失敗');
                    goBack();
                }}
            }} catch (error) {{
                console.error('Error loading character:', error);
                alert('載入角色資料時發生錯誤');
                goBack();
            }}
        }}

        async function saveCharacter() {{
            const loading = document.getElementById('loading');
            loading.classList.add('active');

            try {{
                // Parse interests
                const interestsText = document.getElementById('interests').value.trim();
                const interests = interestsText ? interestsText.split(/[，,]/).map(s => s.trim()).filter(s => s) : [];

                const characterData = {{
                    name: document.getElementById('name').value.trim(),
                    gender: document.getElementById('gender').value,
                    nickname: document.getElementById('nickname').value.trim(),
                    identity: document.getElementById('identity').value.trim(),
                    detail_setting: document.getElementById('detail_setting').value.trim(),
                    other_setting: {{
                        interests: interests
                    }}
                }};

                const response = await fetch(`/api/v2/update-character/${{characterId}}`, {{
                    method: 'PUT',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify(characterData)
                }});

                const data = await response.json();

                if (data.success) {{
                    alert('✅ 角色設定已更新！');
                    window.location.href = '/characters';
                }} else {{
                    alert('保存失敗：' + (data.error || '未知錯誤'));
                }}
            }} catch (error) {{
                console.error('Error saving character:', error);
                alert('保存時發生錯誤');
            }} finally {{
                loading.classList.remove('active');
            }}
        }}

        function goBack() {{
            window.location.href = '/characters';
        }}

        // Load character data on page load
        loadCharacter();
    </script>
</body>
</html>
        """,
        headers={{
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }}
    )


@app.get("/analytics")
async def analytics_dashboard():
    """Analytics Dashboard - displays comprehensive conversation analytics"""
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <title>數據分析 - 戀愛聊天機器人</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: "Microsoft YaHei", "微軟正黑體", sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .dashboard-card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .dashboard-title {
            font-size: 32px;
            color: #667eea;
            margin-bottom: 10px;
        }
        .character-name {
            font-size: 20px;
            color: #666;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }
        .stat-card.green {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }
        .stat-card.orange {
            background: linear-gradient(135deg, #ff9800 0%, #ffa726 100%);
        }
        .stat-card.pink {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        .stat-value {
            font-size: 36px;
            font-weight: bold;
            margin: 10px 0;
        }
        .stat-label {
            font-size: 14px;
            opacity: 0.9;
        }
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        .chart-title {
            font-size: 20px;
            color: #333;
            margin-bottom: 20px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .section {
            margin: 30px 0;
        }
        .favorability-progression {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .progression-item {
            display: flex;
            align-items: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .progression-level {
            font-size: 24px;
            margin-right: 15px;
        }
        .progression-details {
            flex: 1;
        }
        .progression-date {
            color: #666;
            font-size: 14px;
        }
        .button-group {
            text-align: center;
            margin-top: 30px;
        }
        .button {
            display: inline-block;
            padding: 12px 30px;
            margin: 5px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 25px;
            font-size: 16px;
            transition: transform 0.2s;
            border: none;
            cursor: pointer;
        }
        .button:hover {
            transform: translateY(-2px);
        }
        .hours-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(60px, 1fr));
            gap: 10px;
            margin: 20px 0;
        }
        .hour-card {
            text-align: center;
            padding: 15px 10px;
            background: #f8f9fa;
            border-radius: 10px;
            transition: all 0.3s;
        }
        .hour-card.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            transform: scale(1.1);
        }
        .hour-label {
            font-size: 12px;
            margin-bottom: 5px;
        }
        .hour-count {
            font-size: 18px;
            font-weight: bold;
        }
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="dashboard-card">
            <div class="header">
                <div class="dashboard-title">📊 數據分析</div>
                <div class="character-name" id="characterName">載入中...</div>
            </div>

            <div id="loading" style="text-align: center; padding: 40px;">
                <div style="font-size: 24px; color: #667eea;">載入數據中...</div>
            </div>

            <div id="content" style="display: none;">
                <!-- Overview Statistics -->
                <div class="stats-grid" id="statsGrid"></div>

                <!-- Daily Trend Chart -->
                <div class="chart-container">
                    <div class="chart-title">📈 每日訊息趨勢 (最近30天)</div>
                    <canvas id="dailyTrendChart"></canvas>
                </div>

                <!-- Hourly Activity Chart -->
                <div class="chart-container">
                    <div class="chart-title">⏰ 時段活躍度</div>
                    <canvas id="hourlyActivityChart"></canvas>
                </div>

                <!-- Most Active Hours -->
                <div class="section">
                    <div class="chart-container">
                        <div class="chart-title">🔥 最活躍的時段</div>
                        <div class="hours-grid" id="activeHoursGrid"></div>
                    </div>
                </div>

                <!-- Favorability Progression -->
                <div class="section">
                    <div class="chart-container">
                        <div class="chart-title">💕 好感度進度</div>
                        <div class="favorability-progression" id="favorabilityProgression"></div>
                    </div>
                </div>
            </div>

            <div id="error" class="error" style="display: none;"></div>

            <div class="button-group">
                <button class="button" onclick="goBack()">返回檔案</button>
                <a href="/ui2" class="button">返回聊天</a>
            </div>
        </div>
    </div>

    <script>
        let dailyChart = null;
        let hourlyChart = null;

        async function loadAnalytics() {
            const urlParams = new URLSearchParams(window.location.search);
            const characterId = urlParams.get('character_id');

            if (!characterId) {
                showError('找不到角色ID');
                return;
            }

            try {
                const response = await fetch(`/api/v2/analytics/${characterId}`);
                const data = await response.json();

                if (data.success) {
                    displayAnalytics(data);
                } else {
                    showError(data.error || '載入數據失敗');
                }
            } catch (error) {
                showError('載入數據失敗: ' + error.message);
            }
        }

        function displayAnalytics(data) {
            // Hide loading, show content
            document.getElementById('loading').style.display = 'none';
            document.getElementById('content').style.display = 'block';

            // Set character name
            document.getElementById('characterName').textContent = data.character_name;

            // Display overview statistics
            displayOverviewStats(data.analytics.overview);

            // Display charts
            displayDailyTrendChart(data.analytics.trends.daily);
            displayHourlyActivityChart(data.analytics.trends.messages_by_hour);

            // Display most active hours
            displayActiveHours(data.analytics.trends.most_active_hours);

            // Display favorability progression
            displayFavorabilityProgression(data.analytics.favorability);
        }

        function displayOverviewStats(overview) {
            const statsGrid = document.getElementById('statsGrid');
            statsGrid.innerHTML = `
                <div class="stat-card">
                    <div class="stat-label">總訊息數</div>
                    <div class="stat-value">${overview.total_messages}</div>
                </div>
                <div class="stat-card green">
                    <div class="stat-label">對話天數</div>
                    <div class="stat-value">${overview.conversation_days}</div>
                </div>
                <div class="stat-card orange">
                    <div class="stat-label">每日平均</div>
                    <div class="stat-value">${overview.avg_messages_per_day}</div>
                </div>
                <div class="stat-card pink">
                    <div class="stat-label">最長連續天數</div>
                    <div class="stat-value">${overview.longest_streak_days}</div>
                </div>
            `;
        }

        function displayDailyTrendChart(dailyData) {
            const ctx = document.getElementById('dailyTrendChart').getContext('2d');

            if (dailyChart) {
                dailyChart.destroy();
            }

            const dates = dailyData.map(d => d.date);
            const counts = dailyData.map(d => d.count);

            dailyChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: dates,
                    datasets: [{
                        label: '訊息數',
                        data: counts,
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 2.5,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                stepSize: 1
                            }
                        }
                    }
                }
            });
        }

        function displayHourlyActivityChart(hourlyData) {
            const ctx = document.getElementById('hourlyActivityChart').getContext('2d');

            if (hourlyChart) {
                hourlyChart.destroy();
            }

            const hours = hourlyData.map(h => `${h.hour}:00`);
            const counts = hourlyData.map(h => h.count);

            hourlyChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: hours,
                    datasets: [{
                        label: '訊息數',
                        data: counts,
                        backgroundColor: 'rgba(102, 126, 234, 0.7)',
                        borderColor: '#667eea',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 2.5,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                stepSize: 1
                            }
                        }
                    }
                }
            });
        }

        function displayActiveHours(activeHours) {
            const grid = document.getElementById('activeHoursGrid');

            if (activeHours.length === 0) {
                grid.innerHTML = '<div style="text-align: center; color: #666;">暫無數據</div>';
                return;
            }

            grid.innerHTML = activeHours.map(item => `
                <div class="hour-card active">
                    <div class="hour-label">${item.hour}:00</div>
                    <div class="hour-count">${item.count}</div>
                </div>
            `).join('');
        }

        function displayFavorabilityProgression(favorability) {
            const container = document.getElementById('favorabilityProgression');

            const levelEmojis = {
                1: '🌱',
                2: '🌸',
                3: '💕'
            };

            const levelNames = {
                1: '陌生期',
                2: '熟悉期',
                3: '親密期'
            };

            let html = `
                <div class="progression-item">
                    <div class="progression-level">${levelEmojis[favorability.current_level]}</div>
                    <div class="progression-details">
                        <div style="font-size: 18px; font-weight: bold; color: #667eea;">
                            目前等級: ${levelNames[favorability.current_level]}
                        </div>
                    </div>
                </div>
            `;

            if (favorability.progression && favorability.progression.length > 0) {
                html += '<div style="margin: 20px 0; color: #666; font-size: 16px;">歷史進度：</div>';
                favorability.progression.forEach(prog => {
                    html += `
                        <div class="progression-item">
                            <div class="progression-level">${levelEmojis[prog.level]}</div>
                            <div class="progression-details">
                                <div style="font-weight: bold;">${levelNames[prog.level]}</div>
                                <div class="progression-date">第 ${prog.message_count} 條訊息時達成</div>
                            </div>
                        </div>
                    `;
                });
            }

            container.innerHTML = html;
        }

        function showError(message) {
            document.getElementById('loading').style.display = 'none';
            document.getElementById('error').textContent = message;
            document.getElementById('error').style.display = 'block';
        }

        function goBack() {
            const urlParams = new URLSearchParams(window.location.search);
            const characterId = urlParams.get('character_id');
            if (characterId) {
                window.location.href = `/profile?character_id=${characterId}`;
            } else {
                window.history.back();
            }
        }

        // Load analytics on page load
        loadAnalytics();
    </script>
</body>
</html>
        """,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
