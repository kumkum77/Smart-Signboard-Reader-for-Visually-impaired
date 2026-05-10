from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import pytesseract
import pyttsx3
import time

class SignboardApp(App):
    def build(self):
        # Camera feed
        self.img = Image()
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.img)

        # Init camera
        self.capture = cv2.VideoCapture(0)

        # Init TTS engine
        self.engine = pyttsx3.init()

        # Store last spoken text to avoid repeating same output
        self.last_spoken = ""
        self.last_time = 0

        # Schedule updates
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 fps
        Clock.schedule_interval(self.ocr_process, 2.0)   # run OCR every 2 seconds
        return layout

    def update(self, dt):
        """Update live camera feed"""
        ret, frame = self.capture.read()
        if ret:
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img.texture = texture
            self.frame = frame

    def ocr_process(self, dt):
        """Run OCR automatically every few seconds"""
        if hasattr(self, "frame"):
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config="--psm 6").strip()

            if text and text != self.last_spoken:
                # GPS guidance rules
                spoken_text = text
                text_upper = text.upper()

                if "LEFT" in text_upper:
                    spoken_text = "Turn left"
                elif "RIGHT" in text_upper:
                    spoken_text = "Turn right"
                elif "STRAIGHT" in text_upper or "FORWARD" in text_upper:
                    spoken_text = "Move forward"

                # Speak detected / guided text
                self.engine.say(spoken_text)
                self.engine.runAndWait()

                # Save last spoken to prevent repetition
                self.last_spoken = text
                self.last_time = time.time()

if __name__== "__main__":
    SignboardApp().run()