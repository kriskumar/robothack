#!/usr/bin/env python3
"""
Voice-Controlled Chess Analysis with Camera Capture (using imagesnap)
Listens for speech commands and takes camera photos for chess analysis
"""

import speech_recognition as sr
import subprocess
import time
import os
import sys
import platform
from datetime import datetime
from pathlib import Path
import argparse

# Import our chess analyzer
try:
    from chess_board_analyzer import SimpleChessAnalyzer
    CHESS_ANALYZER_AVAILABLE = True
except ImportError:
    print("⚠️  chess_board_analyzer.py not found. Photos will be saved but not analyzed.")
    CHESS_ANALYZER_AVAILABLE = False

class VoiceChessController:
    """
    Voice-controlled chess analyzer that listens for commands and takes camera photos using imagesnap
    """
    
    def __init__(self, 
                 trigger_phrases=None, 
                 photo_dir="chess_photos",
                 enable_robot=False,
                 execute_moves=False,
                 microphone_index=None,
                 camera_device=None):
        """
        Initialize voice chess controller with imagesnap camera
        
        Args:
            trigger_phrases: List of phrases that trigger photo (default: ["next move", "analyze", "chess move"])
            photo_dir: Directory to save photos
            enable_robot: Whether to enable robot control
            execute_moves: Whether to automatically execute moves on robot
            microphone_index: Specific microphone to use (None for default)
            camera_device: Camera device name for imagesnap (None for default)
        """
        
        # Check if we're on macOS
        if platform.system() != "Darwin":
            raise Exception("This version requires macOS and imagesnap. Use --help for other options.")
        
        # Check if imagesnap is available
        if not self.check_imagesnap():
            raise Exception("imagesnap not found. Install with: brew install imagesnap")
        
        # Default trigger phrases
        if trigger_phrases is None:
            self.trigger_phrases = [
                "next move",
                "analyze",
                "chess move", 
                "analyze position",
                "what's the best move",
                "analyze board",
                "take photo",
                "capture position"
            ]
        else:
            self.trigger_phrases = trigger_phrases
        
        # Setup directories
        self.photo_dir = Path(photo_dir)
        self.photo_dir.mkdir(exist_ok=True)
        
        # Robot control settings
        self.enable_robot = enable_robot
        self.execute_moves = execute_moves
        
        # Camera settings
        self.camera_device = camera_device
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(device_index=microphone_index)
        
        # Initialize chess analyzer if available
        self.chess_analyzer = None
        if CHESS_ANALYZER_AVAILABLE:
            try:
                self.chess_analyzer = SimpleChessAnalyzer(enable_robot=enable_robot)
                print(f"✅ Chess analyzer initialized (robot: {'enabled' if enable_robot else 'disabled'})")
            except Exception as e:
                print(f"⚠️  Chess analyzer failed to initialize: {e}")
        
        # Control flags
        self.listening = False
        self.should_stop = False
        
        # Stats
        self.photos_taken = 0
        self.moves_analyzed = 0
        
        print(f"🎤 Voice Chess Controller with imagesnap initialized")
        print(f"📂 Photos will be saved to: {self.photo_dir}")
        print(f"🗣️  Listening for: {', '.join(self.trigger_phrases)}")
    
    def check_imagesnap(self) -> bool:
        """
        Check if imagesnap is available
        
        Returns:
            True if imagesnap is available
        """
        try:
            result = subprocess.run(["imagesnap", "-h"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def list_cameras(self) -> list:
        """
        List available cameras using imagesnap
        
        Returns:
            List of camera device names
        """
        try:
            result = subprocess.run(["imagesnap", "-l"], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                cameras = []
                for line in lines:
                    if line.strip() and not line.startswith('Video Devices:'):
                        # Extract device name from imagesnap output
                        if '=>' in line:
                            device_name = line.split('=>')[0].strip()
                            cameras.append(device_name)
                return cameras
            else:
                print(f"❌ Failed to list cameras: {result.stderr}")
                return []
                
        except Exception as e:
            print(f"❌ Error listing cameras: {e}")
            return []
    
    def take_photo(self) -> str:
        """
        Take a photo with imagesnap and save it with timestamp
        
        Returns:
            Path to the saved photo
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chess_board_{timestamp}.jpg"
        filepath = self.photo_dir / filename
        
        try:
            # Build imagesnap command
            cmd = ["imagesnap"]
            
            # Add camera device if specified
            if self.camera_device:
                cmd.extend(["-d", self.camera_device])
            
            # Add output path
            cmd.append(str(filepath))
            
            print(f"📸 Taking photo with imagesnap...")
            
            # Execute imagesnap
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                if filepath.exists():
                    self.photos_taken += 1
                    print(f"✅ Photo saved: {filepath}")
                    return str(filepath)
                else:
                    print(f"❌ Photo file not created: {filepath}")
                    return None
            else:
                print(f"❌ imagesnap failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"❌ Camera capture timed out")
            return None
        except Exception as e:
            print(f"❌ Failed to take photo: {e}")
            return None
    
    def test_camera(self, show_preview=False) -> bool:
        """
        Test camera by taking a test photo
        
        Args:
            show_preview: Whether to show a preview (imagesnap doesn't support this)
            
        Returns:
            True if camera test successful
        """
        print(f"📷 Testing camera with imagesnap...")
        
        # Take a test photo
        test_path = self.photo_dir / "camera_test.jpg"
        
        try:
            cmd = ["imagesnap"]
            if self.camera_device:
                cmd.extend(["-d", self.camera_device])
            cmd.append(str(test_path))
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and test_path.exists():
                print(f"✅ Camera test successful!")
                print(f"   Test photo saved: {test_path}")
                
                # Clean up test photo
                try:
                    test_path.unlink()
                    print(f"   Test photo cleaned up")
                except:
                    pass
                
                return True
            else:
                print(f"❌ Camera test failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Camera test error: {e}")
            return False
    
    def calibrate_microphone(self):
        """
        Calibrate microphone for ambient noise
        """
        print("🎤 Calibrating microphone for ambient noise...")
        print("   Please be quiet for a moment...")
        
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            
        print(f"✅ Microphone calibrated. Energy threshold: {self.recognizer.energy_threshold}")
    
    def analyze_photo(self, image_path: str):
        """
        Analyze the photo using chess analyzer
        
        Args:
            image_path: Path to the photo
        """
        if not self.chess_analyzer:
            print("❌ Chess analyzer not available")
            return
        
        try:
            print(f"🔍 Analyzing chess position from photo...")
            
            # Analyze the position
            result = self.chess_analyzer.analyze_and_execute(
                image_path, 
                execute_on_robot=self.execute_moves,
                confirm_execution=not self.execute_moves  # Auto-confirm if execute_moves is True
            )
            
            if "error" in result:
                print(f"❌ Analysis error: {result['error']}")
                return
            
            # Report results
            print(f"\n{'='*50}")
            print(f"VOICE CHESS ANALYSIS RESULTS")
            print(f"{'='*50}")
            print(f"🎯 Best Move: {result['best_move']}")
            if result.get('algebraic_move'):
                print(f"♟️  Move Notation: {result['algebraic_move']}")
            if result.get('move_description'):
                print(f"🗣️  Command: '{result['move_description']}'")
            
            # Evaluation
            if result.get('evaluation'):
                eval_info = result['evaluation']
                if eval_info['type'] == 'cp':
                    advantage = eval_info['value'] / 100
                    print(f"📊 Position: {advantage:+.2f} (White advantage)")
                elif eval_info['type'] == 'mate':
                    print(f"🏁 Mate in: {eval_info['value']} moves")
            
            # Robot execution status
            if "robot_execution" in result:
                robot_result = result["robot_execution"]
                if "error" in robot_result:
                    print(f"❌ Robot: {robot_result['error']}")
                elif robot_result.get("cancelled"):
                    print(f"⏸️  Robot: Execution cancelled")
                else:
                    print(f"✅ Robot: Move executed successfully!")
            
            self.moves_analyzed += 1
            print(f"{'='*50}\n")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    
    def process_speech_command(self, command: str):
        """
        Process a recognized speech command
        
        Args:
            command: The recognized speech text
        """
        command_lower = command.lower().strip()
        
        # Check if command contains any trigger phrase
        triggered = False
        for phrase in self.trigger_phrases:
            if phrase.lower() in command_lower:
                triggered = True
                break
        
        if triggered:
            print(f"🗣️  Heard: '{command}' - Taking photo...")
            
            # Take photo
            image_path = self.take_photo()
            
            if image_path:
                # Analyze the photo
                self.analyze_photo(image_path)
            
        else:
            print(f"🗣️  Heard: '{command}' (not a trigger phrase)")
    
    def listen_continuously(self):
        """
        Main listening loop
        """
        print(f"🎤 Starting continuous listening...")
        print(f"💡 Say one of these phrases to trigger photo analysis:")
        for phrase in self.trigger_phrases:
            print(f"   - '{phrase}'")
        print(f"⏹️  Press Ctrl+C to stop")
        
        while not self.should_stop:
            try:
                # Listen for audio
                with self.microphone as source:
                    print("🎧 Listening...")
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                try:
                    # Recognize speech
                    command = self.recognizer.recognize_google(audio)
                    self.process_speech_command(command)
                    
                except sr.UnknownValueError:
                    # Speech was unintelligible
                    pass
                except sr.RequestError as e:
                    print(f"❌ Speech recognition error: {e}")
                    time.sleep(1)
                    
            except sr.WaitTimeoutError:
                # No speech detected, continue listening
                pass
            except KeyboardInterrupt:
                print(f"\n⏹️  Stopping voice controller...")
                self.should_stop = True
                break
            except Exception as e:
                print(f"❌ Listening error: {e}")
                time.sleep(1)
    
    def start(self):
        """
        Start the voice controller
        """
        try:
            # Test camera first
            if not self.test_camera():
                print(f"❌ Cannot start without working camera")
                return
            
            # Calibrate microphone
            self.calibrate_microphone()
            
            # Start listening
            self.listening = True
            self.listen_continuously()
            
        except KeyboardInterrupt:
            print(f"\n⏹️  Voice controller stopped by user")
        except Exception as e:
            print(f"❌ Voice controller error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """
        Stop the voice controller and show stats
        """
        self.listening = False
        self.should_stop = True
        
        print(f"\n📊 SESSION STATISTICS:")
        print(f"   Photos taken: {self.photos_taken}")
        print(f"   Moves analyzed: {self.moves_analyzed}")
        print(f"   Photos saved to: {self.photo_dir}")
        print(f"✅ Voice Chess Controller stopped")

def list_microphones():
    """
    List available microphones
    """
    print("🎤 Available microphones:")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"   {index}: {name}")

def test_microphone(microphone_index=None):
    """
    Test microphone by recording a short phrase
    """
    recognizer = sr.Recognizer()
    microphone = sr.Microphone(device_index=microphone_index)
    
    print("🎤 Testing microphone...")
    print("   Please say something...")
    
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
    
    try:
        text = recognizer.recognize_google(audio)
        print(f"✅ Microphone test successful!")
        print(f"   You said: '{text}'")
        return True
    except sr.UnknownValueError:
        print(f"❌ Could not understand audio")
        return False
    except sr.RequestError as e:
        print(f"❌ Speech recognition error: {e}")
        return False

def check_requirements():
    """
    Check if all requirements are available
    """
    print("🔍 Checking requirements...")
    
    # Check platform
    if platform.system() != "Darwin":
        print("❌ This tool requires macOS")
        return False
    
    # Check imagesnap
    try:
        result = subprocess.run(["imagesnap", "-h"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ imagesnap is available")
        else:
            print("❌ imagesnap not working properly")
            print("   Install with: brew install imagesnap")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ imagesnap not found")
        print("   Install with: brew install imagesnap")
        return False
    
    print("✅ All requirements satisfied")
    return True

def main():
    parser = argparse.ArgumentParser(description="Voice-Controlled Chess Analysis with imagesnap Camera")
    
    # Utility commands
    parser.add_argument("--check-requirements", action="store_true", help="Check if all requirements are installed")
    
    # Voice control options
    parser.add_argument("--list-mics", action="store_true", help="List available microphones")
    parser.add_argument("--test-mic", action="store_true", help="Test microphone")
    parser.add_argument("--microphone", type=int, help="Microphone index to use")
    
    # Camera options
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras")
    parser.add_argument("--test-camera", action="store_true", help="Test camera")
    parser.add_argument("--camera-device", help="Camera device name for imagesnap")
    
    # Photo storage
    parser.add_argument("--photo-dir", default="chess_photos", help="Directory to save photos")
    
    # Trigger phrases
    parser.add_argument("--triggers", nargs="+", help="Custom trigger phrases")
    
    # Robot control
    parser.add_argument("--enable-robot", action="store_true", help="Enable robot control")
    parser.add_argument("--execute-moves", action="store_true", help="Automatically execute moves on robot")
    
    args = parser.parse_args()
    
    try:
        # Handle utility commands
        if args.check_requirements:
            check_requirements()
            return
        
        if args.list_mics:
            list_microphones()
            return
        
        if args.test_mic:
            test_microphone(args.microphone)
            return
        
        if args.list_cameras:
            controller = VoiceChessController()
            cameras = controller.list_cameras()
            print("📷 Available cameras:")
            for i, camera in enumerate(cameras):
                print(f"   {i}: {camera}")
            return
        
        if args.test_camera:
            controller = VoiceChessController(camera_device=args.camera_device)
            controller.test_camera()
            return
        
        # Check requirements before starting
        if not check_requirements():
            print("\n💡 Install missing requirements and try again")
            return
        
        # Create and start voice controller
        controller = VoiceChessController(
            trigger_phrases=args.triggers,
            photo_dir=args.photo_dir,
            enable_robot=args.enable_robot,
            execute_moves=args.execute_moves,
            microphone_index=args.microphone,
            camera_device=args.camera_device
        )
        
        controller.start()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if "imagesnap" in str(e):
            print("\n💡 Install imagesnap with: brew install imagesnap")

if __name__ == "__main__":
    main()
