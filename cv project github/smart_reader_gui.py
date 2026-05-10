import cv2
import pytesseract
import pyttsx3
import threading
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture


class CameraApp(App):
    def build(self):
        self.img1 = Image()
        self.label = Label(text="Detected text will appear here", size_hint=(1, 0.2))
        
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.img1)
        layout.add_widget(self.label)

        # Setup camera
        self.capture = cv2.VideoCapture(0)

        # Setup TTS
        self.engine = pyttsx3.init()
        self.last_text = ""

        # Schedule updates
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS

        return layout

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Convert image to Kivy texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tobytes()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img1.texture = image_texture

            # OCR every 2 seconds
            if int(Clock.get_time()) % 2 == 0:
                text = pytesseract.image_to_string(frame)
                if text.strip() and text != self.last_text:
                    self.label.text = text
                    self.last_text = text
                    threading.Thread(target=self.speak_text, args=(text,), daemon=True).start()

    def speak_text(self, text):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print("TTS Error:", e)

    def on_stop(self):
        self.capture.release()


if __name__ == '__main__':
    CameraApp().run()