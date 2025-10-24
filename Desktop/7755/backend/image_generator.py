"""
Image Generation Client for SenseNova Miaohua API
Generates 2D anime character images for the dating chatbot
"""
import time
import jwt
import requests
from typing import Dict, List, Optional
from backend.config import settings


class ImageGenerator:
    """Client for SenseNova Miaohua image generation API"""

    def __init__(self):
        self.access_key_id = settings.SENSENOVA_ACCESS_KEY_ID
        self.secret_access_key = settings.SENSENOVA_SECRET_ACCESS_KEY
        self.base_url = "https://mhapi.sensetime.com"
        self._token = None
        self._token_expiry = 0

    def _generate_jwt_token(self) -> str:
        """Generate JWT token for API authentication"""
        headers = {
            "alg": "HS256",
            "typ": "JWT"
        }
        payload = {
            "iss": self.access_key_id,
            "exp": int(time.time()) + 1800,  # 30 minutes
            "nbf": int(time.time()) - 5  # Valid from 5 seconds ago
        }
        token = jwt.encode(payload, self.secret_access_key, headers=headers)
        return token

    def _get_valid_token(self) -> str:
        """Get a valid JWT token, generating a new one if expired"""
        current_time = int(time.time())
        if not self._token or current_time >= (self._token_expiry - 300):
            self._token = self._generate_jwt_token()
            self._token_expiry = current_time + 1800
        return self._token

    def get_models(self, size: int = 10, mtp: str = "ALL") -> Dict:
        """
        Get available image generation models

        Args:
            size: Maximum number of models to return
            mtp: Model type filter (LORA, Checkpoint, ALL)

        Returns:
            Response dict with model list
        """
        url = f"{self.base_url}/v1/imgenstd/models"
        token = self._get_valid_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        params = {
            "size": size,
            "offset": 0,
            "mtp": mtp
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting models: {e}")
            return {"error": str(e)}

    def generate_image(
        self,
        model_id: str,
        prompt: str,
        neg_prompt: str = "",
        width: int = 960,
        height: int = 960,
        steps: int = 50,
        cfg_scale: float = 8.0,
        seed: int = 0,
        samples: int = 1,
        reference_image_url: Optional[str] = None,
        lora_configs: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate image using text-to-image

        Args:
            model_id: Model ID to use
            prompt: Positive prompt (character description)
            neg_prompt: Negative prompt
            width: Image width (640-6000)
            height: Image height (640-6000)
            steps: Diffusion steps
            cfg_scale: Prompt control strength
            seed: Random seed (0 for random, same seed = consistent character)
            samples: Number of images to generate (1-8)
            reference_image_url: URL of reference image for consistency
            lora_configs: LoRA configurations

        Returns:
            Response dict with task_id
        """
        url = f"{self.base_url}/v1/imgenstd/imgen"
        token = self._get_valid_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        # Build anime-style prompt
        anime_prompt = f"2D anime style, {prompt}, high quality, detailed, beautiful"

        payload = {
            "model_id": model_id,
            "prompt": anime_prompt,
            "neg_prompt": neg_prompt or "3D, realistic, ugly, blurry, low quality, bad anatomy",
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "samples": samples,
            "format": "PNG",
            "force_translation": False,
            "watermark_config": {"type": "MakaLogo"}
        }

        # Add reference image for character consistency
        if reference_image_url:
            payload["controlnet_configs"] = [
                {
                    "type": "reference_only",
                    "img_url": reference_image_url,
                    "weight": 0.8
                }
            ]

        # Add LoRA configurations if provided
        if lora_configs:
            payload["lora_configs"] = lora_configs

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error generating image: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return {"error": str(e)}

    def get_task_result(self, task_id: str, max_wait: int = 60) -> Dict:
        """
        Get image generation task result

        Args:
            task_id: Task ID from generate_image
            max_wait: Maximum seconds to wait for completion

        Returns:
            Response dict with image URLs
        """
        url = f"{self.base_url}/v1/imgenstd/result/{task_id}"
        token = self._get_valid_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                result = response.json()

                state = result.get("state")
                if state == "SUCCESS":
                    return result
                elif state == "FAILED":
                    return {"error": "Image generation failed", "details": result}
                elif state in ["PENDING", "RUNNING"]:
                    time.sleep(2)  # Wait 2 seconds before checking again
                    continue
                else:
                    return {"error": f"Unknown state: {state}"}

            except Exception as e:
                print(f"Error getting task result: {e}")
                return {"error": str(e)}

        return {"error": "Timeout waiting for image generation"}

    def generate_character_face(
        self,
        appearance_description: str,
        character_name: str,
        model_id: str,
        seed: int = 0
    ) -> Optional[Dict]:
        """
        Generate initial character face image

        Args:
            appearance_description: Character's physical appearance
            character_name: Character's name
            model_id: Model ID to use
            seed: Random seed for consistency

        Returns:
            Dict with image URL and seed if successful
        """
        prompt = f"portrait of {character_name}, {appearance_description}, " \
                 f"smiling, looking at viewer, upper body, cute expression"

        # Generate image
        task_response = self.generate_image(
            model_id=model_id,
            prompt=prompt,
            seed=seed,
            width=960,
            height=960,
            samples=1
        )

        if "error" in task_response:
            return None

        task_id = task_response.get("task_id")
        if not task_id:
            return None

        # Wait for result
        result = self.get_task_result(task_id)

        if "error" in result or not result.get("images"):
            return None

        image_data = result["images"][0]
        return {
            "url": image_data.get("raw"),
            "seed": image_data.get("seed"),
            "task_id": task_id
        }

    def generate_character_activity(
        self,
        appearance_description: str,
        character_name: str,
        activity: str,
        model_id: str,
        reference_image_url: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Generate image of character doing an activity

        Args:
            appearance_description: Character's physical appearance
            character_name: Character's name
            activity: Activity description based on conversation
            model_id: Model ID to use
            reference_image_url: Reference image for character consistency
            seed: Seed from original character image

        Returns:
            Dict with image URL if successful
        """
        prompt = f"{character_name}, {appearance_description}, {activity}, " \
                 f"full body, dynamic pose, detailed background"

        task_response = self.generate_image(
            model_id=model_id,
            prompt=prompt,
            seed=seed or 0,
            width=960,
            height=1280,
            samples=1,
            reference_image_url=reference_image_url
        )

        if "error" in task_response:
            return None

        task_id = task_response.get("task_id")
        if not task_id:
            return None

        result = self.get_task_result(task_id)

        if "error" in result or not result.get("images"):
            return None

        return {"url": result["images"][0].get("raw")}

    def generate_character_with_user(
        self,
        appearance_description: str,
        character_name: str,
        user_description: str,
        scene: str,
        model_id: str,
        reference_image_url: Optional[str] = None,
        seed: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Generate image of character together with user

        Args:
            appearance_description: Character's physical appearance
            character_name: Character's name
            user_description: User's appearance (generic)
            scene: Scene description for milestone
            model_id: Model ID to use
            reference_image_url: Reference image for character consistency
            seed: Seed from original character image

        Returns:
            Dict with image URL if successful
        """
        prompt = f"{character_name}, {appearance_description}, and {user_description}, " \
                 f"{scene}, two people, happy together, romantic atmosphere"

        task_response = self.generate_image(
            model_id=model_id,
            prompt=prompt,
            seed=seed or 0,
            width=1280,
            height=960,
            samples=1,
            reference_image_url=reference_image_url
        )

        if "error" in task_response:
            return None

        task_id = task_response.get("task_id")
        if not task_id:
            return None

        result = self.get_task_result(task_id)

        if "error" in result or not result.get("images"):
            return None

        return {"url": result["images"][0].get("raw")}
