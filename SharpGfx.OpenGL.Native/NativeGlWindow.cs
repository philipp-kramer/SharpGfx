using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;

namespace SharpGfx.OpenGL.Native;

public unsafe class NativeGlWindow : Window
{
    private static readonly Dictionary<uint, ConsoleKey> KeyMapping;


    [DllImport(@"x64/OpenGL.dll", EntryPoint = "createGlfWindow", CallingConvention = CallingConvention.StdCall)]
    private static extern void* createGlfWindow(string title, int width, int height);

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "isWindowCloseRequested", CallingConvention = CallingConvention.Cdecl)]
    private static extern bool isWindowCloseRequested(void* glfWindow);

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "swapBuffers", CallingConvention = CallingConvention.Cdecl)]
    private static extern void swapBuffers(void* glfWindow);

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "getNewWidth", CallingConvention = CallingConvention.Cdecl)]
    private static extern uint getNewWidth();

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "getNewHeight", CallingConvention = CallingConvention.Cdecl)]
    private static extern uint getNewHeight();

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "getKey", CallingConvention = CallingConvention.Cdecl)]
    private static extern uint getKey();

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "getMouseButton", CallingConvention = CallingConvention.Cdecl)]
    private static extern uint getMouseButton();

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "getMouseAction", CallingConvention = CallingConvention.Cdecl)]
    private static extern uint getMouseAction();

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "getMousePosX", CallingConvention = CallingConvention.Cdecl)]
    private static extern double getMousePosX();

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "getMousePosY", CallingConvention = CallingConvention.Cdecl)]
    private static extern double getMousePosY();

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "getMouseScrollX", CallingConvention = CallingConvention.Cdecl)]
    private static extern double getMouseScrollX();

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "getMouseScrollY", CallingConvention = CallingConvention.Cdecl)]
    private static extern double getMouseScrollY();

    [DllImport(@"x64/OpenGL.dll", EntryPoint = "terminateGlfw", CallingConvention = CallingConvention.Cdecl)]
    private static extern void terminateGlfw();

    static NativeGlWindow()
    {
        KeyMapping = new Dictionary<uint, ConsoleKey>
        {
            { 256, ConsoleKey.Escape },
            { 257, ConsoleKey.Enter },
            { 262, ConsoleKey.RightArrow },
            { 263, ConsoleKey.LeftArrow },
            { 264, ConsoleKey.DownArrow },
            { 265, ConsoleKey.UpArrow },
            { 266, ConsoleKey.PageUp },
            { 267, ConsoleKey.PageDown },
            { 320, ConsoleKey.NumPad0 },
            { 321, ConsoleKey.NumPad1 },
            { 322, ConsoleKey.NumPad2 },
            { 323, ConsoleKey.NumPad3 },
            { 324, ConsoleKey.NumPad4 },
            { 325, ConsoleKey.NumPad5 },
            { 326, ConsoleKey.NumPad6 },
            { 327, ConsoleKey.NumPad7 },
            { 328, ConsoleKey.NumPad8 },
            { 329, ConsoleKey.NumPad9 },
        };
    }

    private readonly void* _window;
    private readonly InteractiveCamera _camera;
    private readonly int _targetFrameRate;
    private readonly ButtonState _mouseLeftButton = new();
    private readonly ButtonState _mouseRightButton = new();

    public NativeGlWindow(
        string title,
        int width,
        int height,
        InteractiveCamera camera = null,
        int targetFrameRate = 60)
        : base(width, height)
    {
        _window = createGlfWindow(title, width, height);
        _camera = camera;
        _targetFrameRate = targetFrameRate;
    }

    public override void Show(Rendering rendering)
    {
        base.Show(rendering);

        var watch = new Stopwatch();
        var interval = TimeSpan.FromSeconds(1d / _targetFrameRate);

        while (!isWindowCloseRequested(_window))
        {
            uint key = getKey();
            if (key < uint.MaxValue && TryGetConsoleKey(key, out var mappedKey))
            {
                if (mappedKey == ConsoleKey.Escape) break;

                _camera?.OnKeyDown(mappedKey);
            }

            uint width = getNewWidth();
            uint height = getNewHeight();
            if (width < int.MaxValue || height < int.MaxValue)
            {
                Width = (int) width;
                Height = (int) height;
            }

            uint button = getMouseButton();
            uint action = getMouseAction();
            float posX = (float) getMousePosX();
            float posY = (float) getMousePosY();

            if (button == (uint)GlfwMouseButton.Left)
            {
                if (action == (uint)GlfwButtonAction.Press)
                {
                    _camera?.MouseDown(MouseButton.Left, posX, posY);
                    _mouseLeftButton.Down();
                }
                if (action == (uint)GlfwButtonAction.Release) _mouseLeftButton.Up();
            }

            if (button == (uint)GlfwMouseButton.Right)
            {
                if (action == (uint)GlfwButtonAction.Press)
                {
                    _camera?.MouseDown(MouseButton.Right, posX, posY);
                    _mouseRightButton.Down();
                }
                if (action == (uint)GlfwButtonAction.Release) _mouseRightButton.Up();
            }

            if (MouseX != posX || MouseY != posY)
            {
                if (_mouseLeftButton.Pressed)
                {
                    _camera?.MouseDragging(MouseButton.Left, posX, posY);
                }
                if (_mouseRightButton.Pressed)
                {
                    _camera?.MouseDragging(MouseButton.Right, posX, posY);
                }
            }

            MouseX = posX;
            MouseY = posY;

            float wheelX = (float) getMouseScrollX();
            float wheelY = (float) getMouseScrollY();
            if (wheelX != 0 || wheelY != 0)
            {
                _camera?.MouseDragging(MouseButton.Middle, wheelX, wheelY);
            }

            OnUpdateFrame();
            OnRenderFrame();

            var delta = interval - watch.Elapsed;
            if (delta > TimeSpan.Zero)
            {
                Thread.Sleep(delta);
            }
            swapBuffers(_window);
        }
        terminateGlfw();
    }

    private static bool TryGetConsoleKey(uint key, out ConsoleKey mappedKey)
    {
        if (KeyMapping.TryGetValue(key, out mappedKey))
        {
            return true;
        }
        switch (key)
        {
            case <= '0' and <= '9':
                mappedKey = (ConsoleKey) (key - '0' + (uint) ConsoleKey.D0);
                return true;

            case <= 'A' and <= 'Z':
                mappedKey = (ConsoleKey) (key - 'A' + (uint) ConsoleKey.A);
                return true;

            default:
                return false;
        }
    }
}