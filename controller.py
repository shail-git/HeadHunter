import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api
import ctypes
import time

class GameController:
    def __init__(self):
        self.SendInput = ctypes.windll.user32.SendInput
        self.game_window = win32gui.FindWindow(None, 'Counter-Strike: Global Offensive - Direct3D 9')
        
    def grab_screen(self, game_resolution=(1024,768)):
        """Capture game screen"""
        if not self.game_window:
            raise Exception("Game window not found")
            
        bar_height = 35
        offset_height_top = 135 
        offset_height_bottom = 135 
        offset_sides = 100 
        width = game_resolution[0] - 2 * offset_sides
        height = game_resolution[1] - offset_height_top - offset_height_bottom

        hwindc = win32gui.GetWindowDC(self.game_window)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)

        memdc.BitBlt((0, 0), (width, height), srcdc, (offset_sides, bar_height + offset_height_top), win32con.SRCCOPY)
        
        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(self.game_window, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def shoot(self, duration=0.1):
        """Simulate mouse click for shooting"""
        self.hold_left_click()
        time.sleep(duration)
        self.release_left_click()

    def reload(self, duration=0.1):
        """Simulate reload action"""
        self.press_key(0x13)  # R key
        time.sleep(duration)
        self.release_key(0x13)

    def hold_left_click(self):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.mi = MouseInput(0, 0, 0, 0x0002, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(0), ii_)
        self.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def release_left_click(self):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.mi = MouseInput(0, 0, 0, 0x0004, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(0), ii_)
        self.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def press_key(self, key):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, key, 0, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        self.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def release_key(self, key):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, key, 0x0002, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        self.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
