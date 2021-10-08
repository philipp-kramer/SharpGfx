using System;

namespace SharpGfx
{
    public abstract class Window
    {
        private const float NavigationSpeed = 0.1f;
        private const float MouseSensitivity = 1/200f;

        protected readonly PositionTracking MouseTracking;

        public Rendering Rendering { get; set; }

        protected Window(Rendering rendering)
        {
            Rendering = rendering;
            MouseTracking = new PositionTracking();
        }

        private static float Limit(float value, float range)
        {
            return Math.Min(Math.Max(value, -range), range);
        }

        public abstract event Action<KeyDownArgs> KeyDown;
        public abstract void Run(int targetFrameRate);

        protected void OnCameraKeyDown(ConsoleKey key, CameraRendering rendering)
        {
            switch (key)
            {
                case ConsoleKey.NumPad5:
                case ConsoleKey.PageUp:
                case ConsoleKey.E:
                    rendering.CameraForward(NavigationSpeed);
                    break;
                case ConsoleKey.NumPad0:
                case ConsoleKey.PageDown:
                case ConsoleKey.Q:
                    rendering.CameraForward(-NavigationSpeed);
                    break;
                case ConsoleKey.NumPad4:
                case ConsoleKey.LeftArrow:
                case ConsoleKey.A:
                    rendering.CameraRight(-NavigationSpeed);
                    break;
                case ConsoleKey.NumPad6:
                case ConsoleKey.RightArrow:
                case ConsoleKey.D:
                    rendering.CameraRight(NavigationSpeed);
                    break;
                case ConsoleKey.NumPad8:
                case ConsoleKey.UpArrow:
                case ConsoleKey.W:
                    rendering.CameraUp(NavigationSpeed);
                    break;
                case ConsoleKey.NumPad2:
                case ConsoleKey.DownArrow:
                case ConsoleKey.S:
                    rendering.CameraUp(-NavigationSpeed);
                    break;
            }
        }

        protected void OnMouseMove()
        {
            if (Rendering is CameraRendering cameraRendering)
            {
                cameraRendering.CameraYaw = Limit(cameraRendering.CameraYaw + MouseTracking.DeltaX * MouseSensitivity, MathF.PI);
                cameraRendering.CameraPitch = Limit(cameraRendering.CameraPitch - MouseTracking.DeltaY * MouseSensitivity, MathF.PI / 2);
            }
        }
    }
}
