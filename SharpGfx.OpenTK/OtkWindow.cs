using System;
using System.Collections.Generic;
using OpenTK;
using OpenTK.Graphics;
using OpenTK.Graphics.OpenGL;
using OpenTK.Input;

namespace SharpGfx.OpenTK
{
    public sealed class OtkWindow : Window, IDisposable
    {
        private static readonly Dictionary<Key, ConsoleKey> KeyMapping;

        static OtkWindow()
        {
            KeyMapping = new Dictionary<Key, ConsoleKey>();
            foreach (Key key in Enum.GetValues(typeof(Key)))
            {
                string keyName = Enum.GetName(typeof(Key), key);
                if (Enum.TryParse(typeof(ConsoleKey), keyName, out var value))
                {
                    KeyMapping.Add(key, (ConsoleKey)value);
                }
                else
                {
                    switch (key)
                    {
                        case Key.Keypad0:
                            KeyMapping.Add(key, ConsoleKey.NumPad0);
                            break;
                        case Key.Keypad1:
                            KeyMapping.Add(key, ConsoleKey.NumPad1);
                            break;
                        case Key.Keypad2:
                            KeyMapping.Add(key, ConsoleKey.NumPad2);
                            break;
                        case Key.Keypad3:
                            KeyMapping.Add(key, ConsoleKey.NumPad3);
                            break;
                        case Key.Keypad4:
                            KeyMapping.Add(key, ConsoleKey.NumPad4);
                            break;
                        case Key.Keypad5:
                            KeyMapping.Add(key, ConsoleKey.NumPad5);
                            break;
                        case Key.Keypad6:
                            KeyMapping.Add(key, ConsoleKey.NumPad6);
                            break;
                        case Key.Keypad7:
                            KeyMapping.Add(key, ConsoleKey.NumPad7);
                            break;
                        case Key.Keypad8:
                            KeyMapping.Add(key, ConsoleKey.NumPad8);
                            break;
                        case Key.Keypad9:
                            KeyMapping.Add(key, ConsoleKey.NumPad9);
                            break;
                        case Key.Down:
                            KeyMapping.Add(key, ConsoleKey.DownArrow);
                            break;
                        case Key.Up:
                            KeyMapping.Add(key, ConsoleKey.UpArrow);
                            break;
                        case Key.Left:
                            KeyMapping.Add(key, ConsoleKey.LeftArrow);
                            break;
                        case Key.Right:
                            KeyMapping.Add(key, ConsoleKey.RightArrow);
                            break;
                    };
                }
            }
        }

        private readonly GameWindow _window;
        private PolygonMode _polygonMode;

        public override event Action<KeyDownArgs> KeyDown;

        public OtkWindow(System.Drawing.Size size, string title, bool antiAliased)
            : base(null)
        {
            var graphicsMode = antiAliased
                ? new GraphicsMode(24, 24, 0, 4)
                : GraphicsMode.Default;
            _window = new GameWindow(size.Width, size.Height, graphicsMode, title);
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
        }

        public override void Run(int targetFrameRate)
        {
            _window.Run(targetFrameRate);
        }

        private void OnLoad(object sender, EventArgs e)
        {
            _window.Title += $" (OpenGL Version: {GL.GetString(StringName.Version)})";
            Rendering?.OnLoad();
            var mouse = Mouse.GetState();
            MouseTracking.X = mouse.X;
            MouseTracking.Y = mouse.Y;
        }

        private void OnResize(object sender, EventArgs e)
        {
            var size = new System.Drawing.Size(
                _window.Size.Width,
                _window.Size.Height);
            Rendering?.OnResize(size);
        }

        private void OnUpdateFrame(object sender, FrameEventArgs e)
        {
            Rendering?.OnUpdateFrame();
        }

        private void OnRenderFrame(object sender, FrameEventArgs e)
        {
            var mouse = Mouse.GetState();
            MouseTracking.Update(mouse.X, mouse.Y);
            OnMouseMove();
            Rendering.OnRenderFrame();
            _window.SwapBuffers();
        }

        private void OnKeyDown(object sender, KeyboardKeyEventArgs e)
        {
            if (!KeyMapping.TryGetValue(e.Key, out var key))
            {
                return;
            }

            if (KeyDown != null)
            {
                var eventArgs = new KeyDownArgs(key);
                KeyDown.Invoke(eventArgs);
                if (eventArgs.Handled)
                {
                    return;
                }
            }

            if (Rendering is CameraRendering cameraRendering)
            {
                OnCameraKeyDown(key, cameraRendering);
            }

            switch (key)
            {
                case ConsoleKey.G:
                    TogglePolygonMode();
                    GL.PolygonMode(MaterialFace.FrontAndBack, _polygonMode);
                    break;

                case ConsoleKey.Escape:
                    _window.Exit();
                    break;
            }
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

        public void Dispose()
        {
            GC.SuppressFinalize(this);
            Dispose(true);
        }

        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                _window.Dispose();
            }
        }

        ~OtkWindow()
        {
            Dispose(false);
        }
    }
}