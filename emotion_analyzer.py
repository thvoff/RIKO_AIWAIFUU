import cv2
import numpy as np
import librosa
import logging
from typing import Dict, List, Optional, Tuple, Any
import threading
import time
import os
import json
from datetime import datetime
import tempfile

# Try to import emotion detection libraries
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    print("⚠️  FER not installed. Install with: pip install fer")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("⚠️  soundfile not installed. Install with: pip install soundfile")

class EmotionAnalyzer:
    """
    Advanced emotion analysis for AI Waifu
    - Analyzes facial emotions from video frames
    - Analyzes voice emotions from audio
    - Combines multiple emotion signals
    - Tracks emotional patterns over time
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize emotion detectors
        if FER_AVAILABLE:
            self.face_emotion_detector = FER(mtcnn=True)  # More accurate face detection
        else:
            self.face_emotion_detector = None
            self.logger.warning("Facial emotion detection unavailable - FER not installed")
        
        # Emotion mapping
        self.emotion_labels = {
            'angry': 'angry',
            'disgust': 'disgusted',
            'fear': 'fearful',
            'happy': 'happy',
            'sad': 'sad',
            'surprise': 'surprised',
            'neutral': 'neutral'
        }
        
        # Voice emotion thresholds (based on audio features)
        self.voice_emotion_thresholds = {
            'pitch_high': 300,      # Hz - excitement/anger threshold
            'pitch_low': 150,       # Hz - sadness threshold
            'tempo_fast': 150,      # BPM - excitement threshold
            'tempo_slow': 80,       # BPM - sadness threshold
            'energy_high': 0.7,     # Energy level - excitement threshold
            'energy_low': 0.3       # Energy level - calm/sad threshold
        }
        
        # Emotion history for pattern analysis
        self.emotion_history = []
        self.emotion_patterns = {}
        
        # Storage paths
        self.emotion_data_dir = "emotion_data"
        self.emotion_history_file = os.path.join(self.emotion_data_dir, "emotion_history.json")
        
        os.makedirs(self.emotion_data_dir, exist_ok=True)
        
        # Performance optimization
        self.processing_lock = threading.Lock()
        self.last_analysis_time = 0
        self.analysis_interval = 1.0  # Analyze emotions every 1 second max
        
        # Load existing emotion data
        self.load_emotion_history()
        
        self.logger.info("Emotion Analyzer initialized")
    
    def load_emotion_history(self):
        """Load emotion history from storage"""
        try:
            if os.path.exists(self.emotion_history_file):
                with open(self.emotion_history_file, 'r') as f:
                    data = json.load(f)
                    self.emotion_history = data.get('history', [])
                    self.emotion_patterns = data.get('patterns', {})
                
                self.logger.info(f"Loaded {len(self.emotion_history)} emotion records")
            
        except Exception as e:
            self.logger.error(f"Error loading emotion history: {e}")
            self.emotion_history = []
            self.emotion_patterns = {}
    
    def save_emotion_history(self):
        """Save emotion history to storage"""
        try:
            data = {
                'history': self.emotion_history[-500:],  # Keep last 500 records
                'patterns': self.emotion_patterns,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.emotion_history_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving emotion history: {e}")
    
    def analyze_facial_emotion(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze facial emotions from video frame
        """
        try:
            if not self.face_emotion_detector:
                return {
                    'dominant_emotion': 'neutral',
                    'emotions': {'neutral': 1.0},
                    'confidence': 0.0,
                    'face_detected': False,
                    'error': 'Facial emotion detection not available'
                }
            
            # Convert BGR to RGB for FER
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect emotions
            emotions = self.face_emotion_detector.detect_emotions(rgb_frame)
            
            if not emotions:
                return {
                    'dominant_emotion': 'neutral',
                    'emotions': {'neutral': 1.0},
                    'confidence': 0.0,
                    'face_detected': False,
                    'error': 'No face detected'
                }
            
            # Get the first face's emotions (assuming single person)
            face_emotions = emotions[0]['emotions']
            
            # Find dominant emotion
            dominant_emotion = max(face_emotions, key=face_emotions.get)
            confidence = face_emotions[dominant_emotion]
            
            # Get face box for visualization
            face_box = emotions[0]['box']
            
            return {
                'dominant_emotion': dominant_emotion,
                'emotions': face_emotions,
                'confidence': confidence,
                'face_detected': True,
                'face_box': face_box,
                'error': None
            }
            
        except Exception as e:
            self.logger.error(f"Error in facial emotion analysis: {e}")
            return {
                'dominant_emotion': 'neutral',
                'emotions': {'neutral': 1.0},
                'confidence': 0.0,
                'face_detected': False,
                'error': str(e)
            }
    
    def analyze_voice_emotion(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze voice emotions from audio file
        """
        try:
            if not SOUNDFILE_AVAILABLE:
                return {
                    'dominant_emotion': 'neutral',
                    'confidence': 0.0,
                    'audio_features': {},
                    'error': 'Audio analysis not available - soundfile not installed'
                }
            
            if not os.path.exists(audio_path):
                return {
                    'dominant_emotion': 'neutral',
                    'confidence': 0.0,
                    'audio_features': {},
                    'error': f'Audio file not found: {audio_path}'
                }
            
            # Load audio file
            y, sr = librosa.load(audio_path, duration=30)  # Limit to 30 seconds
            
            if len(y) == 0:
                return {
                    'dominant_emotion': 'neutral',
                    'confidence': 0.0,
                    'audio_features': {},
                    'error': 'Empty audio file'
                }
            
            # Extract audio features
            features = self._extract_audio_features(y, sr)
            
            # Classify emotion based on features
            emotion_result = self._classify_voice_emotion(features)
            
            return {
                'dominant_emotion': emotion_result['emotion'],
                'confidence': emotion_result['confidence'],
                'audio_features': features,
                'error': None
            }
            
        except Exception as e:
            self.logger.error(f"Error in voice emotion analysis: {e}")
            return {
                'dominant_emotion': 'neutral',
                'confidence': 0.0,
                'audio_features': {},
                'error': str(e)
            }
    
    def _extract_audio_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract relevant audio features for emotion analysis"""
        try:
            features = {}
            
            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['mean_pitch'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
            else:
                features['mean_pitch'] = 0
                features['pitch_std'] = 0
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = tempo
            
            # Energy/Intensity
            rms = librosa.feature.rms(y=y)[0]
            features['mean_energy'] = np.mean(rms)
            features['energy_std'] = np.std(rms)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid'] = np.mean(spectral_centroids)
            
            # Zero crossing rate (voice quality indicator)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zero_crossing_rate'] = np.mean(zcr)
            
            # MFCC features (first 4 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=4)
            for i, mfcc in enumerate(mfccs):
                features[f'mfcc_{i+1}'] = np.mean(mfcc)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting audio features: {e}")
            return {}
    
    def _classify_voice_emotion(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Classify emotion based on voice features"""
        try:
            if not features:
                return {'emotion': 'neutral', 'confidence': 0.0}
            
            emotion_scores = {
                'happy': 0.0,
                'sad': 0.0,
                'angry': 0.0,
                'excited': 0.0,
                'calm': 0.0,
                'neutral': 0.5  # Default baseline
            }
            
            mean_pitch = features.get('mean_pitch', 0)
            tempo = features.get('tempo', 0)
            mean_energy = features.get('mean_energy', 0)
            
            # High pitch + high energy + fast tempo = excited/happy
            if mean_pitch > self.voice_emotion_thresholds['pitch_high']:
                if mean_energy > self.voice_emotion_thresholds['energy_high']:
                    emotion_scores['excited'] += 0.4
                    emotion_scores['happy'] += 0.3
                else:
                    emotion_scores['angry'] += 0.3
            
            # Low pitch + low energy = sad
            elif mean_pitch < self.voice_emotion_thresholds['pitch_low']:
                if mean_energy < self.voice_emotion_thresholds['energy_low']:
                    emotion_scores['sad'] += 0.5
                else:
                    emotion_scores['angry'] += 0.2
            
            # Tempo-based adjustments
            if tempo > self.voice_emotion_thresholds['tempo_fast']:
                emotion_scores['excited'] += 0.2
                emotion_scores['happy'] += 0.2
            elif tempo < self.voice_emotion_thresholds['tempo_slow']:
                emotion_scores['sad'] += 0.2
                emotion_scores['calm'] += 0.2
            
            # Energy-based adjustments
            if mean_energy > self.voice_emotion_thresholds['energy_high']:
                emotion_scores['excited'] += 0.2
                emotion_scores['angry'] += 0.1
            elif mean_energy < self.voice_emotion_thresholds['energy_low']:
                emotion_scores['sad'] += 0.2
                emotion_scores['calm'] += 0.2
            
            # Find dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[dominant_emotion]
            
            # Normalize confidence to 0-1 range
            confidence = min(1.0, confidence)
            
            return {
                'emotion': dominant_emotion,
                'confidence': confidence,
                'all_scores': emotion_scores
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying voice emotion: {e}")
            return {'emotion': 'neutral', 'confidence': 0.0}
    
    def combine_emotions(self, face_emotion: Dict, voice_emotion: Dict) -> Dict[str, Any]:
        """
        Combine facial and voice emotions for overall emotion assessment
        """
        try:
            # Weight factors (can be adjusted)
            face_weight = 0.6
            voice_weight = 0.4
            
            combined_emotions = {}
            
            # Get all possible emotions
            all_emotions = set()
            if 'emotions' in face_emotion:
                all_emotions.update(face_emotion['emotions'].keys())
            all_emotions.add(voice_emotion.get('dominant_emotion', 'neutral'))
            
            # Calculate combined scores
            for emotion in all_emotions:
                face_score = face_emotion.get('emotions', {}).get(emotion, 0.0)
                voice_score = 1.0 if voice_emotion.get('dominant_emotion') == emotion else 0.0
                voice_score *= voice_emotion.get('confidence', 0.0)
                
                combined_score = (face_score * face_weight) + (voice_score * voice_weight)
                combined_emotions[emotion] = combined_score
            
            # Find dominant emotion
            if combined_emotions:
                dominant_emotion = max(combined_emotions, key=combined_emotions.get)
                confidence = combined_emotions[dominant_emotion]
            else:
                dominant_emotion = 'neutral'
                confidence = 0.5
            
            # Create emotion context for AI response
            emotion_context = self._generate_emotion_context(
                dominant_emotion, 
                confidence, 
                face_emotion, 
                voice_emotion
            )
            
            result = {
                'dominant_emotion': dominant_emotion,
                'confidence': confidence,
                'combined_emotions': combined_emotions,
                'face_emotion': face_emotion.get('dominant_emotion', 'neutral'),
                'voice_emotion': voice_emotion.get('dominant_emotion', 'neutral'),
                'emotion_context': emotion_context,
                'timestamp': datetime.now().isoformat()
            }
            
            # Record this emotion analysis
            self.record_emotion(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error combining emotions: {e}")
            return {
                'dominant_emotion': 'neutral',
                'confidence': 0.5,
                'combined_emotions': {'neutral': 0.5},
                'emotion_context': 'neutral interaction',
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_emotion_context(self, emotion: str, confidence: float, face_data: Dict, voice_data: Dict) -> str:
        """Generate context description for the AI to use in responses"""
        try:
            context_templates = {
                'happy': [
                    "The user seems happy and cheerful",
                    "The user appears to be in a good mood",
                    "The user looks pleased and content"
                ],
                'sad': [
                    "The user seems sad or down",
                    "The user appears to be feeling low",
                    "The user looks like they might need comfort"
                ],
                'angry': [
                    "The user appears frustrated or angry",
                    "The user seems upset about something",
                    "The user looks like they need to vent"
                ],
                'excited': [
                    "The user seems very excited and energetic",
                    "The user appears enthusiastic about something",
                    "The user looks thrilled and animated"
                ],
                'surprised': [
                    "The user looks surprised or amazed",
                    "The user appears shocked by something",
                    "The user seems caught off guard"
                ],
                'fearful': [
                    "The user appears worried or anxious",
                    "The user seems concerned about something",
                    "The user looks like they need reassurance"
                ],
                'neutral': [
                    "The user appears calm and neutral",
                    "The user seems in a normal mood",
                    "The user looks relaxed"
                ]
            }
            
            # Get appropriate template
            templates = context_templates.get(emotion, context_templates['neutral'])
            base_context = templates[0]  # Use first template for consistency
            
            # Add confidence qualifier
            if confidence > 0.8:
                confidence_qualifier = "clearly"
            elif confidence > 0.6:
                confidence_qualifier = "somewhat"
            else:
                confidence_qualifier = "possibly"
            
            # Add additional context based on voice/face combination
            additional_context = ""
            if face_data.get('face_detected') and voice_data.get('dominant_emotion'):
                if face_data.get('dominant_emotion') != voice_data.get('dominant_emotion'):
                    additional_context = " (mixed emotional signals detected)"
            
            return f"{base_context.replace('seems', f'{confidence_qualifier} seems')}{additional_context}"
            
        except Exception as e:
            self.logger.error(f"Error generating emotion context: {e}")
            return "neutral interaction"
    
    def record_emotion(self, emotion_data: Dict):
        """Record emotion analysis for pattern tracking"""
        try:
            # Add to history
            self.emotion_history.append(emotion_data)
            
            # Update patterns
            emotion = emotion_data['dominant_emotion']
            if emotion not in self.emotion_patterns:
                self.emotion_patterns[emotion] = {
                    'count': 0,
                    'total_confidence': 0,
                    'last_seen': None
                }
            
            self.emotion_patterns[emotion]['count'] += 1
            self.emotion_patterns[emotion]['total_confidence'] += emotion_data['confidence']
            self.emotion_patterns[emotion]['last_seen'] = emotion_data['timestamp']
            
            # Keep history manageable
            if len(self.emotion_history) > 1000:
                self.emotion_history = self.emotion_history[-500:]
            
            # Save periodically
            if len(self.emotion_history) % 10 == 0:
                threading.Thread(target=self.save_emotion_history).start()
                
        except Exception as e:
            self.logger.error(f"Error recording emotion: {e}")
    
    def get_emotion_summary(self) -> Dict[str, Any]:
        """Get summary of emotion patterns"""
        try:
            total_analyses = len(self.emotion_history)
            
            if total_analyses == 0:
                return {'total_analyses': 0, 'patterns': {}}
            
            # Calculate emotion frequencies
            emotion_frequencies = {}
            for emotion, data in self.emotion_patterns.items():
                emotion_frequencies[emotion] = {
                    'count': data['count'],
                    'percentage': (data['count'] / total_analyses) * 100,
                    'average_confidence': data['total_confidence'] / data['count'],
                    'last_seen': data['last_seen']
                }
            
            # Recent emotion trend (last 10 analyses)
            recent_emotions = [e['dominant_emotion'] for e in self.emotion_history[-10:]]
            most_recent_emotion = recent_emotions[-1] if recent_emotions else 'unknown'
            
            return {
                'total_analyses': total_analyses,
                'emotion_frequencies': emotion_frequencies,
                'most_recent_emotion': most_recent_emotion,
                'recent_trend': recent_emotions
            }
            
        except Exception as e:
            self.logger.error(f"Error getting emotion summary: {e}")
            return {'total_analyses': 0, 'patterns': {}}
    
    def analyze_combined_emotion(self, frame: np.ndarray = None, audio_path: str = None) -> Dict[str, Any]:
        """
        Main method to analyze emotions from both video and audio
        """
        try:
            current_time = time.time()
            
            # Rate limiting
            if current_time - self.last_analysis_time < self.analysis_interval:
                return {'error': 'Rate limited - too frequent analysis requests'}
            
            self.last_analysis_time = current_time
            
            with self.processing_lock:
                # Analyze facial emotion if frame provided
                face_emotion = {'dominant_emotion': 'neutral', 'confidence': 0.0}
                if frame is not None:
                    face_emotion = self.analyze_facial_emotion(frame)
                
                # Analyze voice emotion if audio provided
                voice_emotion = {'dominant_emotion': 'neutral', 'confidence': 0.0}
                if audio_path is not None:
                    voice_emotion = self.analyze_voice_emotion(audio_path)
                
                # Combine emotions
                combined_result = self.combine_emotions(face_emotion, voice_emotion)
                
                return combined_result
            
        except Exception as e:
            self.logger.error(f"Error in combined emotion analysis: {e}")
            return {
                'dominant_emotion': 'neutral',
                'confidence': 0.5,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def draw_emotion_overlay(self, frame: np.ndarray, emotion_result: Dict) -> np.ndarray:
        """Draw emotion information overlay on video frame"""
        try:
            if not emotion_result:
                return frame
            
            # Colors for different emotions
            emotion_colors = {
                'happy': (0, 255, 0),      # Green
                'sad': (255, 0, 0),        # Blue
                'angry': (0, 0, 255),      # Red
                'excited': (0, 255, 255),  # Yellow
                'surprised': (255, 0, 255), # Magenta
                'fearful': (128, 0, 128),  # Purple
                'neutral': (255, 255, 255) # White
            }
            
            emotion = emotion_result.get('dominant_emotion', 'neutral')
            confidence = emotion_result.get('confidence', 0.0)
            color = emotion_colors.get(emotion, (255, 255, 255))
            
            # Draw emotion info
            cv2.putText(frame, f"Emotion: {emotion.title()}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw face box if available
            face_box = emotion_result.get('face_emotion', {}).get('face_box')
            if face_box:
                x, y, w, h = face_box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw emotion context
            context = emotion_result.get('emotion_context', '')
            if context:
                # Word wrap for long context
                words = context.split(' ')
                line_length = 40
                current_line = ''
                y_offset = 90
                
                for word in words:
                    if len(current_line + word) > line_length:
                        cv2.putText(frame, current_line, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        current_line = word + ' '
                        y_offset += 25
                    else:
                        current_line += word + ' '
                
                if current_line.strip():
                    cv2.putText(frame, current_line.strip(), (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error drawing emotion overlay: {e}")
            return frame

# Testing and utility functions
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Emotion Analyzer System')
    parser.add_argument('--test-face', action='store_true', help='Test facial emotion detection')
    parser.add_argument('--test-voice', type=str, help='Test voice emotion detection on audio file')
    parser.add_argument('--test-combined', action='store_true', help='Test combined emotion detection')
    
    args = parser.parse_args()
    
    emotion_analyzer = EmotionAnalyzer()
    
    if args.test_face:
        print("🎭 Testing Facial Emotion Detection...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze emotions
            result = emotion_analyzer.analyze_facial_emotion(frame)
            
            # Draw overlay
            frame = emotion_analyzer.draw_emotion_overlay(frame, {'face_emotion': result})
            
            cv2.imshow('Facial Emotion Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    elif args.test_voice:
        print(f"🎤 Testing Voice Emotion Detection on: {args.test_voice}")
        result = emotion_analyzer.analyze_voice_emotion(args.test_voice)
        
        print("\n📊 Voice Emotion Analysis Results:")
        print(f"   Dominant Emotion: {result['dominant_emotion']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Audio Features: {result['audio_features']}")
        if result['error']:
            print(f"   Error: {result['error']}")
    
    elif args.test_combined:
        print("🎭🎤 Testing Combined Emotion Detection...")
        print("This will use webcam for face and capture audio for voice analysis")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        # Simple audio capture for testing (you'd integrate this with your existing audio system)
        import sounddevice as sd
        import soundfile as sf
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # For demo purposes, analyze just facial emotion in real-time
            face_result = emotion_analyzer.analyze_facial_emotion(frame)
            
            # Simulate voice emotion (in real implementation, this would be from actual audio)
            voice_result = {'dominant_emotion': 'neutral', 'confidence': 0.5}
            
            # Combine emotions
            combined_result = emotion_analyzer.combine_emotions(face_result, voice_result)
            
            # Draw overlay
            frame = emotion_analyzer.draw_emotion_overlay(frame, combined_result)
            
            cv2.imshow('Combined Emotion Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Show emotion summary
        summary = emotion_analyzer.get_emotion_summary()
        print("\n📈 Emotion Analysis Summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
    
    else:
        print("Use --test-face, --test-voice <file>, or --test-combined")
        print("Example: python emotion_analyzer.py --test-face")