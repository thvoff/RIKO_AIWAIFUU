import asyncio
import logging
import time
import cv2
import numpy as np
from typing import Optional, Dict, Any
import os
import sys
import threading

# Import our custom modules
from tts_optimizer import TTSOptimizer
from face_recognition_system import FaceRecognitionSystem
from emotion_analyzer import EmotionAnalyzer

class IntegrationManagerV2:
    """
    Advanced Integration Manager with Face Recognition and Emotion Analysis
    Coordinates all AI waifu features for human-like interaction
    """
    
    def __init__(self, owner_name: str = "Pavan"):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        self.owner_name = owner_name
        
        # Initialize all components
        self.tts_optimizer = TTSOptimizer()
        self.face_recognition = FaceRecognitionSystem(owner_name=owner_name)
        self.emotion_analyzer = EmotionAnalyzer()
        
        # We'll add memory system in next step
        self.memory_system = None
        
        # Performance tracking
        self.total_requests = 0
        self.average_response_time = 0
        self.interaction_stats = {
            'owner_interactions': 0,
            'stranger_interactions': 0,
            'emotional_responses': 0
        }
        
        # Camera for real-time face/emotion detection
        self.camera = None
        self.camera_enabled = False
        
        # Current context for AI responses
        self.current_context = {
            'person_id': 'unknown',
            'emotion': 'neutral',
            'emotion_confidence': 0.0,
            'face_recognition_confidence': 0.0,
            'interaction_count': 0
        }
        
        self.logger.info("Advanced Integration Manager initialized")
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('waifu_advanced.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def initialize_camera(self) -> bool:
        """Initialize camera for face recognition and emotion detection"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.logger.error("Could not open camera")
                return False
            
            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.camera_enabled = True
            self.logger.info("Camera initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing camera: {e}")
            return False
    
    def capture_current_frame(self) -> Optional[np.ndarray]:
        """Capture current frame from camera"""
        try:
            if not self.camera_enabled or not self.camera:
                return None
            
            ret, frame = self.camera.read()
            if ret:
                return frame
            else:
                self.logger.warning("Failed to capture frame")
                return None
                
        except Exception as e:
            self.logger.error(f"Error capturing frame: {e}")
            return None
    
    async def process_user_input_advanced(
        self, 
        user_text: str, 
        audio_file: Optional[str] = None,
        capture_frame: bool = True
    ) -> Dict[str, Any]:
        """
        Advanced processing pipeline with face recognition and emotion analysis
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing advanced user input: {user_text[:50]}...")
            
            # Step 1: Capture current frame for face/emotion analysis
            current_frame = None
            if capture_frame and self.camera_enabled:
                current_frame = self.capture_current_frame()
            
            # Step 2: Run face recognition and emotion analysis in parallel
            face_task = asyncio.create_task(self.analyze_face(current_frame))
            emotion_task = asyncio.create_task(self.analyze_emotions(current_frame, audio_file))
            
            face_result, emotion_result = await asyncio.gather(face_task, emotion_task)
            
            # Step 3: Update context based on recognition results
            self.update_interaction_context(face_result, emotion_result)
            
            # Step 4: Generate context-aware LLM response
            enhanced_prompt = self.create_enhanced_prompt(user_text, face_result, emotion_result)
            llm_response = await self.get_context_aware_llm_response(enhanced_prompt)
            
            # Step 5: Generate optimized TTS response
            audio_response_path = await self.generate_optimized_voice_response(
                llm_response, 
                audio_file or "character_files/main_sample.wav"
            )
            
            # Step 6: Update interaction statistics
            total_time = time.time() - start_time
            self.update_performance_metrics(total_time)
            self.update_interaction_stats(face_result, emotion_result)
            
            result = {
                'text_response': llm_response,
                'audio_response_path': audio_response_path,
                'face_recognition': face_result,
                'emotion_analysis': emotion_result,
                'current_context': self.current_context.copy(),
                'response_time': total_time,
                'success': True,
                'personalized': face_result.get('is_owner', False)
            }
            
            self.logger.info(f"Advanced response generated in {total_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in advanced processing: {e}")
            return {
                'text_response': "I'm having some technical difficulties. Please try again.",
                'audio_response_path': None,
                'face_recognition': None,
                'emotion_analysis': None,
                'current_context': self.current_context.copy(),
                'response_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
    
    async def analyze_face(self, frame: Optional[np.ndarray]) -> Dict[str, Any]:
        """Analyze face recognition"""
        try:
            if frame is None:
                return {
                    'person_id': 'unknown',
                    'is_owner': False,
                    'confidence': 0.0,
                    'message': 'No camera input available'
                }
            
            # Run face recognition in thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.face_recognition.recognize_face_in_frame, 
                frame
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in face analysis: {e}")
            return {
                'person_id': 'error',
                'is_owner': False,
                'confidence': 0.0,
                'message': f'Face recognition error: {e}'
            }
    
    async def analyze_emotions(self, frame: Optional[np.ndarray], audio_file: Optional[str]) -> Dict[str, Any]:
        """Analyze emotions from face and voice"""
        try:
            # Run emotion analysis in thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.emotion_analyzer.analyze_combined_emotion,
                frame,
                audio_file
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in emotion analysis: {e}")
            return {
                'dominant_emotion': 'neutral',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def update_interaction_context(self, face_result: Dict, emotion_result: Dict):
        """Update current interaction context"""
        try:
            # Update face recognition context
            if face_result:
                self.current_context['person_id'] = face_result.get('person_id', 'unknown')
                self.current_context['face_recognition_confidence'] = face_result.get('confidence', 0.0)
            
            # Update emotion context
            if emotion_result:
                self.current_context['emotion'] = emotion_result.get('dominant_emotion', 'neutral')
                self.current_context['emotion_confidence'] = emotion_result.get('confidence', 0.0)
            
            # Increment interaction count
            self.current_context['interaction_count'] += 1
            
            self.logger.info(f"Context updated: {self.current_context}")
            
        except Exception as e:
            self.logger.error(f"Error updating context: {e}")
    
    def create_enhanced_prompt(self, user_text: str, face_result: Dict, emotion_result: Dict) -> str:
        """Create enhanced prompt with face recognition and emotion context"""
        try:
            base_prompt = f"User: {user_text}\n"
            
            # Add face recognition context
            if face_result and face_result.get('is_owner'):
                base_prompt += f"Recognition: This is {self.owner_name}, the owner of this AI system.\n"
                base_prompt += f"Relationship: You should respond as their personal AI companion who knows them well.\n"
            elif face_result and face_result.get('person_id') == 'stranger':
                base_prompt += "Recognition: This is someone you haven't met before.\n"
                base_prompt += "Relationship: Be polite but maintain appropriate boundaries with strangers.\n"
            
            # Add emotion context
            if emotion_result and emotion_result.get('dominant_emotion') != 'neutral':
                emotion = emotion_result['dominant_emotion']
                confidence = emotion_result.get('confidence', 0.0)
                emotion_context = emotion_result.get('emotion_context', '')
                
                base_prompt += f"User's Current Emotion: {emotion} (confidence: {confidence:.2f})\n"
                base_prompt += f"Emotional Context: {emotion_context}\n"
                base_prompt += "Response Style: Adapt your tone and content to be appropriate for their emotional state.\n"
            
            # Add interaction history context
            interaction_count = self.current_context.get('interaction_count', 0)
            if interaction_count > 1:
                base_prompt += f"Conversation Context: This is interaction #{interaction_count} in this session.\n"
            
            # Add personality instructions
            base_prompt += "\nPersonality Instructions:\n"
            base_prompt += "- Be warm, empathetic, and supportive\n"
            base_prompt += "- Show genuine care for the user's wellbeing\n"
            base_prompt += "- Adapt your personality based on who you're talking to\n"
            base_prompt += "- Keep responses conversational and natural\n"
            base_prompt += "- If the user seems distressed, offer comfort and support\n\n"
            
            base_prompt += "AI Assistant: "
            
            return base_prompt
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced prompt: {e}")
            return f"User: {user_text}\nAI Assistant: "
    
    async def get_context_aware_llm_response(self, enhanced_prompt: str) -> str:
        """Get response from Ollama with enhanced context"""
        try:
            import subprocess
            import json
            
            # Truncate prompt if too long to avoid timeout
            if len(enhanced_prompt) > 2000:
                enhanced_prompt = enhanced_prompt[:2000] + "...\nAI Assistant: "
            
            # Call Ollama with enhanced prompt
            cmd = [
                'ollama', 'run', 'llama3.1:8b',
                enhanced_prompt,
                '--temperature', '0.7',
                '--top-p', '0.9',
                '--max-tokens', '150'  # Slightly longer for contextual responses
            ]
            
            # Run with timeout
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=20  # Increased timeout for context processing
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                response = stdout.strip()
                
                # Clean up response
                if "AI Assistant:" in response:
                    response = response.split("AI Assistant:")[-1].strip()
                
                # Ensure response is appropriate length
                sentences = response.split('.')
                if len(sentences) > 3:
                    response = '. '.join(sentences[:3]) + '.'
                
                return response
            else:
                self.logger.error(f"Ollama error: {stderr}")
                return self.get_fallback_response()
                
        except subprocess.TimeoutExpired:
            self.logger.warning("Ollama response timed out")
            return self.get_fallback_response()
        except Exception as e:
            self.logger.error(f"Error getting context-aware LLM response: {e}")
            return self.get_fallback_response()
    
    def get_fallback_response(self) -> str:
        """Get appropriate fallback response based on current context"""
        person_id = self.current_context.get('person_id', 'unknown')
        emotion = self.current_context.get('emotion', 'neutral')
        
        if person_id == 'owner':
            if emotion == 'sad':
                return f"I'm here for you, {self.owner_name}. What's troubling you?"
            elif emotion == 'happy':
                return f"It's wonderful to see you in such a good mood, {self.owner_name}!"
            else:
                return f"Hello {self.owner_name}, I'm glad you're here. How can I help you today?"
        elif person_id == 'stranger':
            return "Hello there! I'm an AI assistant. How can I help you today?"
        else:
            return "I'm having trouble processing your request right now. Could you try again?"
    
    async def generate_optimized_voice_response(
        self, 
        text: str, 
        reference_audio: str
    ) -> str:
        """Generate voice response using optimized TTS"""
        try:
            # Add emotional inflection to text based on context
            emotion = self.current_context.get('emotion', 'neutral')
            modified_text = self.add_emotional_inflection(text, emotion)
            
            # Generate optimized voice response
            audio_path = await self.tts_optimizer.optimize_gpt_sovits_call(
                modified_text, 
                reference_audio
            )
            
            return audio_path
            
        except Exception as e:
            self.logger.error(f"Error generating voice response: {e}")
            return None
    
    def add_emotional_inflection(self, text: str, emotion: str) -> str:
        """Add emotional inflection to text for more natural TTS"""
        try:
            # Simple emotional markers that some TTS systems recognize
            emotion_markers = {
                'happy': '😊 ',
                'sad': '😔 ',
                'excited': '🎉 ',
                'angry': '😠 ',
                'surprised': '😲 ',
                'fearful': '😰 '
            }
            
            marker = emotion_markers.get(emotion, '')
            
            # Adjust punctuation for emotional delivery
            if emotion == 'excited':
                text = text.replace('.', '!')
            elif emotion == 'sad':
                text = text.replace('!', '.')
            
            return marker + text
            
        except Exception as e:
            self.logger.error(f"Error adding emotional inflection: {e}")
            return text
    
    def update_interaction_stats(self, face_result: Dict, emotion_result: Dict):
        """Update interaction statistics"""
        try:
            if face_result and face_result.get('is_owner'):
                self.interaction_stats['owner_interactions'] += 1
            elif face_result and face_result.get('person_id') == 'stranger':
                self.interaction_stats['stranger_interactions'] += 1
            
            if emotion_result and emotion_result.get('dominant_emotion') != 'neutral':
                self.interaction_stats['emotional_responses'] += 1
                
        except Exception as e:
            self.logger.error(f"Error updating interaction stats: {e}")
    
    def update_performance_metrics(self, response_time: float):
        """Update performance tracking"""
        self.total_requests += 1
        self.average_response_time = (
            (self.average_response_time * (self.total_requests - 1) + response_time) 
            / self.total_requests
        )
        
        if self.total_requests % 5 == 0:  # Log every 5 requests
            self.logger.info(
                f"Performance: {self.total_requests} requests, "
                f"avg response time: {self.average_response_time:.2f}s"
            )
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get component statuses
            face_summary = self.face_recognition.get_interaction_summary()
            emotion_summary = self.emotion_analyzer.get_emotion_summary()
            tts_performance = self.tts_optimizer.get_performance_stats()
            
            return {
                'total_requests': self.total_requests,
                'average_response_time': self.average_response_time,
                'current_context': self.current_context,
                'interaction_stats': self.interaction_stats,
                'face_recognition': face_summary,
                'emotion_analysis': emotion_summary,
                'tts_performance': tts_performance,
                'camera_enabled': self.camera_enabled,
                'components_loaded': {
                    'tts_optimizer': True,
                    'face_recognition': True,
                    'emotion_analyzer': True,
                    'memory_system': self.memory_system is not None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting comprehensive status: {e}")
            return {'error': str(e)}
    
    def setup_owner_face_recognition(self, image_path: str = None) -> bool:
        """Setup owner face recognition"""
        try:
            if image_path:
                # Use provided image
                return self.face_recognition.setup_owner_face(image_path)
            else:
                # Capture photo using camera
                from face_recognition_system import FaceRecognitionSetup
                setup_util = FaceRecognitionSetup()
                
                if setup_util.capture_owner_photo():
                    return self.face_recognition.setup_owner_face("face_data/owner_setup.jpg")
                return False
                
        except Exception as e:
            self.logger.error(f"Error setting up owner face recognition: {e}")
            return False
    
    def start_visual_monitoring(self):
        """Start visual monitoring with face recognition and emotion detection"""
        try:
            if not self.initialize_camera():
                self.logger.error("Could not start visual monitoring - camera initialization failed")
                return False
            
            self.logger.info("Visual monitoring started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting visual monitoring: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources when shutting down"""
        self.logger.info("Cleaning up Advanced Integration Manager...")
        
        # Cleanup camera
        if self.camera:
            self.camera.release()
            cv2.destroyAllWindows()
        
        # Cleanup TTS optimizer
        if self.tts_optimizer:
            self.tts_optimizer.cleanup_old_audio_files()
        
        self.logger.info("Advanced Integration Manager cleanup complete")

# Enhanced interface for main_chat.py
class AdvancedWaifuChat:
    """
    Advanced interface for main_chat.py with all features
    """
    
    def __init__(self, owner_name: str = "Pavan"):
        self.manager = IntegrationManagerV2(owner_name=owner_name)
        self.logger = logging.getLogger(__name__)
        self.owner_name = owner_name
    
    async def setup(self, setup_face_recognition: bool = True) -> bool:
        """Setup the advanced system"""
        try:
            success = True
            
            # Start visual monitoring
            if not self.manager.start_visual_monitoring():
                self.logger.warning("Visual monitoring not available - continuing without camera")
                success = False
            
            # Setup face recognition
            if setup_face_recognition:
                print(f"🔧 Setting up face recognition for {self.owner_name}...")
                if not self.manager.setup_owner_face_recognition():
                    self.logger.warning("Face recognition setup failed - continuing without personalization")
                    success = False
                else:
                    print("✅ Face recognition setup complete!")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Setup error: {e}")
            return False
    
    async def chat(
        self, 
        user_input: str, 
        audio_file: str = None,
        enable_visual: bool = True
    ) -> Dict[str, Any]:
        """
        Advanced chat method with all features
        
        Args:
            user_input: The transcribed text from user
            audio_file: Path to user's audio file (for voice emotion analysis)
            enable_visual: Whether to capture visual frame for face/emotion analysis
            
        Returns:
            Dictionary with response text, audio path, and advanced metadata
        """
        return await self.manager.process_user_input_advanced(
            user_input, 
            audio_file,
            capture_frame=enable_visual
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return self.manager.get_comprehensive_status()
    
    def shutdown(self):
        """Shutdown the system cleanly"""
        self.manager.cleanup()

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced AI Waifu System')
    parser.add_argument('--owner', default='Pavan', help='Owner name')
    parser.add_argument('--setup', action='store_true', help='Run initial setup')
    parser.add_argument('--test', action='store_true', help='Run system test')
    
    args = parser.parse_args()
    
    async def main():
        waifu = AdvancedWaifuChat(owner_name=args.owner)
        
        try:
            if args.setup:
                print("🚀 Running Advanced AI Waifu Setup...")
                await waifu.setup(setup_face_recognition=True)
                print("✅ Setup complete!")
                
            elif args.test:
                print("🧪 Running Advanced System Test...")
                
                # Setup without face recognition for quick test
                await waifu.setup(setup_face_recognition=False)
                
                test_phrases = [
                    "Hello, how are you today?",
                    "I'm feeling a bit sad today",
                    "That's amazing news!",
                    "Can you help me with something?"
                ]
                
                for phrase in test_phrases:
                    print(f"\n🧪 Testing: {phrase}")
                    
                    response = await waifu.chat(phrase, enable_visual=True)
                    
                    if response['success']:
                        print(f"✅ Response: {response['text_response']}")
                        print(f"⏱️  Time: {response['response_time']:.2f}s")
                        print(f"👤 Person: {response['face_recognition']['person_id']}")
                        print(f"😊 Emotion: {response['emotion_analysis']['dominant_emotion']}")
                    else:
                        print(f"❌ Error: {response.get('error')}")
                
                # Show system status
                status = waifu.get_status()
                print(f"\n📊 System Status: {status}")
                
            else:
                print("Use --setup for initial setup or --test to run tests")
                
        finally:
            waifu.shutdown()
    
    asyncio.run(main())