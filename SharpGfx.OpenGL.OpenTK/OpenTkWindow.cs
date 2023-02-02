using System;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using OpenTK.Windowing.GraphicsLibraryFramework;

namespace SharpGfx.OpenGL.OpenTK;

public sealed class OpenTkWindow : Window, IDisposable
{
    private static readonly Dictionary<Keys, ConsoleKey> KeyMapping;

    static OpenTkWindow()
    {
        KeyMapping = new Dictionary<Keys, ConsoleKey>();
        foreach (Keys key in Enum.GetValues(typeof(Keys)))
        {
            string keyName = Enum.GetName(typeof(Keys), key);
            if (Enum.TryParse(typeof(ConsoleKey), keyName, out var value) && value != null)
            {
                KeyMapping.Add(key, (ConsoleKey) value);
            }
            else
            {
                switch (key)
                {
                    case Keys.KeyPad0:
                        KeyMapping.Add(key, ConsoleKey.NumPad0);
                        break;
                    case Keys.KeyPad1:
                        KeyMapping.Add(key, ConsoleKey.NumPad1);
                        break;
                    case Keys.KeyPad2:
                        KeyMapping.Add(key, ConsoleKey.NumPad2);
                        break;
                    case Keys.KeyPad3:
                        KeyMapping.Add(key, ConsoleKey.NumPad3);
                        break;
                    case Keys.KeyPad4:
                        KeyMapping.Add(key, ConsoleKey.NumPad4);
                        break;
                    case Keys.KeyPad5:
                        KeyMapping.Add(key, ConsoleKey.NumPad5);
                        break;
                    case Keys.KeyPad6:
                        KeyMapping.Add(key, ConsoleKey.NumPad6);
                        break;
                    case Keys.KeyPad7:
                        KeyMapping.Add(key, ConsoleKey.NumPad7);
                        break;
                    case Keys.KeyPad8:
                        KeyMapping.Add(key, ConsoleKey.NumPad8);
                        break;
                    case Keys.KeyPad9:
                        KeyMapping.Add(key, ConsoleKey.NumPad9);
                        break;
                    case Keys.Down:
                        KeyMapping.Add(key, ConsoleKey.DownArrow);
                        break;
                    case Keys.Up:
                        KeyMapping.Add(key, ConsoleKey.UpArrow);
                        break;
                    case Keys.Left:
                        KeyMapping.Add(key, ConsoleKey.LeftArrow);
                        break;
                    case Keys.Right:
                        KeyMapping.Add(key, ConsoleKey.RightArrow);
                        break;
                }
            }
        }
    }

    private readonly GameWindow _window;
    private readonly InteractiveCamera _camera;
    private PolygonMode _polygonMode;
    private MouseButton _mouseButton;

    public event Action<KeyDownArgs> KeyDown;

    public OpenTkWindow(
        string title,
        int width,
        int height,
        InteractiveCamera camera = null,
        int targetFrameRate = 60, 
        bool antiAliased = false)
        : base(width, height)
    {
        _camera = camera;
        var settings = new GameWindowSettings
        {
            RenderFrequency = targetFrameRate,
            UpdateFrequency = targetFrameRate
        };
        var nativeSettings = new NativeWindowSettings
        {
            Size = new Vector2i(width, height),
            Title = title,
            RedBits = 8,
            GreenBits = 8,
            BlueBits = 8,
            AlphaBits = 8,
            NumberOfSamples = antiAliased ? 4 : 1
        };
        _window = new GameWindow(settings, nativeSettings);
        if (antiAliased)
        {
            GL.Enable(EnableCap.Multisample);
        }
        _polygonMode = PolygonMode.Fill;

        _window.Load += OnLoad;
        _window.Resize += OnResize;
        _window.UpdateFrame += OnUpdateFrame;
        _window.RenderFrame += OnRenderFrame;
        _window.KeyDown += OnKeyDown;
        _window.MouseMove += OnMouseMove;
        _window.MouseDown += OnMouseDown;
        _window.MouseUp += OnMouseUp;
        _window.MouseWheel += OnMouseWheel;
    }

    public override void Show(Rendering rendering)
    {
        base.Show(rendering);
        _window.Run();
    }

    protected override void OnLoad()
    {
        _window.Title += $" (OpenGL Version: {GL.GetString(StringName.Version)})";
        base.OnLoad();
    }

    private void OnResize(ResizeEventArgs e)
    {
        Width = e.Width;
        Height = e.Height;
    }

    private void OnUpdateFrame(FrameEventArgs e)
    {
        OnUpdateFrame();
    }

    private void OnRenderFrame(FrameEventArgs e)
    {
        OnRenderFrame();
        _window.SwapBuffers();
    }

    private void OnKeyDown(KeyboardKeyEventArgs e)
    {
        if (!KeyMapping.TryGetValue(e.Key, out var key)) return;

        if (KeyDown != null)
        {
            var eventArgs = new KeyDownArgs(key);
            KeyDown.Invoke(eventArgs);
            if (eventArgs.Handled) return;
        }

        _camera?.OnKeyDown(key);

        switch (key)
        {
            case ConsoleKey.G:
                TogglePolygonMode();
                GL.PolygonMode(MaterialFace.FrontAndBack, _polygonMode);
                break;

            case ConsoleKey.Escape:
                _window.Close();
                break;
        }
    }

    private void OnMouseMove(MouseMoveEventArgs e)
    {
        MouseX = e.X;
        MouseY = e.Y;
        _camera?.MouseDragging(_mouseButton, MouseX, MouseY);
    }

    private void OnMouseDown(MouseButtonEventArgs e)
    {
        _mouseButton = e.Button switch
        {
            global::OpenTK.Windowing.GraphicsLibraryFramework.MouseButton.Left => MouseButton.Left,
            global::OpenTK.Windowing.GraphicsLibraryFramework.MouseButton.Middle => MouseButton.Middle,
            global::OpenTK.Windowing.GraphicsLibraryFramework.MouseButton.Right => MouseButton.Right,
            _ => MouseButton.None
        };
        _camera?.MouseDown(_mouseButton, _window.MousePosition.X, _window.MousePosition.Y);
    }

    private void OnMouseUp(MouseButtonEventArgs e)
    {
        Rendering.OnMouseUp(_mouseButton);
        _mouseButton = MouseButton.None;
    }

    private void OnMouseWheel(MouseWheelEventArgs e)
    {
        _camera?.MouseDragging(MouseButton.Middle, e.Offset.X, e.OffsetY);
    }

    private void TogglePolygonMode()
    {
        switch (_polygonMode)
        {
            case PolygonMode.Line:
                _polygonMode = PolygonMode.Fill;
                break;
            case PolygonMode.Fill:
                _polygonMode = PolygonMode.Line;
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(_polygonMode));
        }
    }

    private void Dispose(bool disposing)
    {
        if (disposing)
        {
            _window.Load -= OnLoad;
            _window.Resize -= OnResize;
            _window.UpdateFrame -= OnUpdateFrame;
            _window.RenderFrame -= OnRenderFrame;
            _window.KeyDown -= OnKeyDown;
            _window.MouseMove -= OnMouseMove;
            _window.MouseDown -= OnMouseDown;
            _window.MouseUp -= OnMouseUp;
            _window.MouseWheel -= OnMouseWheel;
            _window.Dispose();
        }
    }

    public void Dispose()
    {
        GC.SuppressFinalize(this);
        Dispose(true);
    }

    ~OpenTkWindow()
    {
        Dispose(false);
    }
}