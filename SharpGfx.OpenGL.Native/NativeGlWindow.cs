using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;

namespace SharpGfx.OpenGL.Native;

public unsafe partial class NativeGlWindow : Window
{
    private static readonly Dictionary<uint, ConsoleKey> KeyMapping;


    [LibraryImport(NativeGlApi.OpenGlLibarary, StringMarshalling = StringMarshalling.Utf8)]
    private static partial void* createGlfWindow(string title, int width, int height);

    [LibraryImport(NativeGlApi.OpenGlLibarary)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool isWindowCloseRequested(void* glfWindow);

    [LibraryImport(NativeGlApi.OpenGlLibarary)]
    private static partial void swapBuffers(void* glfWindow);

    [LibraryImport(NativeGlApi.OpenGlLibarary)]
    private static partial uint getNewWidth();

    [LibraryImport(NativeGlApi.OpenGlLibarary)]
    private static partial uint getNewHeight();

    [LibraryImport(NativeGlApi.OpenGlLibarary)]
    private static partial uint getKey();

    [LibraryImport(NativeGlApi.OpenGlLibarary)]
    private static partial uint getMouseButton();

    [LibraryImport(NativeGlApi.OpenGlLibarary)]
    private static partial uint getMouseAction();

    [LibraryImport(NativeGlApi.OpenGlLibarary)]
    private static partial double getMousePosX();

    [LibraryImport(NativeGlApi.OpenGlLibarary)]
    private static partial double getMousePosY();

    [LibraryImport(NativeGlApi.OpenGlLibarary)]
    private static partial double getMouseScrollX();

    [LibraryImport(NativeGlApi.OpenGlLibarary)]
    private static partial double getMouseScrollY();

    [LibraryImport(NativeGlApi.OpenGlLibarary)]
    private static partial void terminateGlfw();

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
    private readonly InteractiveCamera? _camera;
    private readonly int _targetFrameRate;

    public NativeGlWindow(
        string title,
        int width,
        int height,
        InteractiveCamera? camera = default,
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
            uint width = getNewWidth();
            uint height = getNewHeight();
            if (width < int.MaxValue || height < int.MaxValue)
            {
                Width = (int) width;
                Height = (int) height;
            }

            HandleInput();

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

    private void HandleInput()
    {
        if (_camera == default) return;

        uint key = getKey();
        if (key < uint.MaxValue && TryGetConsoleKey(key, out var mappedKey))
        {
            _camera.KeyDown(mappedKey);
        }

        uint button = getMouseButton();
        uint action = getMouseAction();

        var actionButton = button switch
        {
            (uint)GlfwMouseButton.Left => MouseButton.Left,
            (uint)GlfwMouseButton.Middle => MouseButton.Middle,
            (uint)GlfwMouseButton.Right => MouseButton.Right,
            _ => MouseButton.None
        };
        switch (action)
        {
            case (uint)GlfwButtonAction.Press:
                _camera.MouseDown(actionButton);
                break;

            case (uint)GlfwButtonAction.Release:
                _camera.MouseUp(actionButton);
                break;
        }


        _camera.MousePosition = ((float)getMousePosX(), (float)getMousePosY());
        _camera.MouseScrollX += (float)getMouseScrollX();
        _camera.MouseScrollY += (float)getMouseScrollY();
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