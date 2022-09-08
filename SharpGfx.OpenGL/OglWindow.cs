using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using SharpGfx.Host;
using SharpGfx.Primitives;

namespace SharpGfx.OpenGL
{
    public unsafe class OglWindow : Window
    {
        private static readonly Dictionary<uint, ConsoleKey> KeyMapping;

        static OglWindow()
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
        private IVector2 _size;
        private readonly Camera _camera;
        private readonly int _targetFrameRate;
        private readonly DeltaTracking _mouseMove;
        private readonly StateTracking _mouseLeftButton;
        private readonly StateTracking _mouseRightButton;

        public OglWindow(
            string title,
            IVector2 size,
            Camera camera = null,
            int targetFrameRate = 60)
        {
            _window = GL.CreateWindow(title, (int) size.X, (int) size.Y);
            _size = size;
            _camera = camera;
            _targetFrameRate = targetFrameRate;
            _mouseMove = new DeltaTracking();
            _mouseLeftButton = new StateTracking();
            _mouseRightButton = new StateTracking();
        }

        public override Point2 Position => new(new HostVector2(Screen, _mouseMove.X, _mouseMove.Y));

        public override void Show(Rendering rendering)
        {
            base.Show(rendering);

            rendering.OnResize(_size);

            var watch = new Stopwatch();
            var interval = TimeSpan.FromSeconds(1d / _targetFrameRate);

            while (!GL.IsWindowCloseRequested(_window))
            {
                uint key = GL.getKey();
                if (key < uint.MaxValue && TryGetConsoleKey(key, out var mappedKey))
                {
                    if (mappedKey == ConsoleKey.Escape) break;

                    _camera?.OnKeyDown(mappedKey);
                }

                uint width = GL.getNewWidth();
                uint height = GL.getNewHeight();
                if (width < uint.MaxValue || height < uint.MaxValue)
                {
                    _size = Screen.Vector2(width, height);
                    rendering.OnResize(_size);
                }

                uint button = GL.getMouseButton();
                uint action = GL.getMouseAction();
                if (button == (uint)GlfwMouseButton.Left)
                {
                    if (action == (uint)GlfwButtonAction.Press) _mouseLeftButton.Activate();
                    if (action == (uint)GlfwButtonAction.Release) _mouseLeftButton.Deactivate();
                }

                if (button == (uint)GlfwMouseButton.Right)
                {
                    if (action == (uint)GlfwButtonAction.Press) _mouseRightButton.Activate();
                    if (action == (uint)GlfwButtonAction.Release) _mouseRightButton.Deactivate();
                }

                double posX = GL.getMousePosX();
                double posY = GL.getMousePosY();
                if (0 <= posX && posX < _size.X &&
                    0 <= posY && posY < _size.Y)
                {
                    _mouseMove.X = (int) posX;
                    _mouseMove.Y = (int) posY;
                    int deltaX = _mouseMove.DeltaX;
                    int deltaY = _mouseMove.DeltaY;
                    if (deltaX != 0 || deltaY != 0)
                    {
                        if (_mouseLeftButton.Active)
                        {
                            _camera?.MouseMoving(new HostVector2(Screen, -deltaX, -deltaY), MouseButtons.Left);
                        }
                        if (_mouseRightButton.Active)
                        {
                            _camera?.MouseMoving(new HostVector2(Screen, -deltaX, -deltaY), MouseButtons.Right);
                        }
                    }
                    float wheelX = (float) GL.getMouseScrollX();
                    float wheelY = (float) GL.getMouseScrollY();
                    if (wheelX != 0 || wheelY != 0)
                    {
                        _camera?.MouseMoving(new HostVector2(Screen, -wheelX, -wheelY), MouseButtons.Middle);
                    }
                }

                OnUpdateFrame();
                OnRenderFrame();

                var delta = interval - watch.Elapsed;
                if (delta > TimeSpan.Zero)
                {
                    Thread.Sleep(delta);
                }
                GL.SwapBuffers(_window);
            }
            GL.Terminate();
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
}
