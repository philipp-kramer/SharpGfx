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

        public override void Show(Rendering rendering)
        {
            base.Show(rendering);

            rendering.OnResize(_size);

            var watch = new Stopwatch();
            var interval = TimeSpan.FromSeconds(1d / _targetFrameRate);

            while (!GL.IsWindowCloseRequested(_window))
            {
                var events = new uint[5];
                fixed (uint* keyAndButtonData = events)
                {
                    GL.GetEvents(keyAndButtonData);
                }
                uint key = events[0];
                uint button = events[1];
                uint action = events[2];
                uint width = events[3];
                uint height = events[4];

                if (key < uint.MaxValue && TryGetConsoleKey(key, out var mappedKey))
                {
                    if (mappedKey == ConsoleKey.Escape) break;

                    _camera?.OnKeyDown(mappedKey);
                }

                if (width < uint.MaxValue || height < uint.MaxValue)
                {
                    _size = Screen.Vector2(width, height);
                    rendering.OnResize(_size);
                }

                var mouseInputs = new double[4];
                fixed (double* mouseData = mouseInputs)
                {
                    GL.GetMouseInputs(mouseData);
                }

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

                if (0 <= mouseInputs[0] && mouseInputs[0] < _size.X &&
                    0 <= mouseInputs[1] && mouseInputs[1] < _size.Y)
                {
                    _mouseMove.X = (int) mouseInputs[0];
                    _mouseMove.Y = (int) mouseInputs[1];
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
                    float wheelX = (float) mouseInputs[2];
                    float wheelY = (float) mouseInputs[3];
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
            if ('0' <= key && key <= '9')
            {
                mappedKey = (ConsoleKey) (key - '0' + (uint) ConsoleKey.D0);
                return true;
            }
            if ('a' <= key && key <= 'z')
            {
                mappedKey = (ConsoleKey) (key - 'a' + (uint) ConsoleKey.A);
                return true;
            }

            return false;
        }
    }
}
