using System;
using System.Runtime.InteropServices;

namespace SharpGfx
{
    public abstract class Window
    {
        private const float NavigationSpeed = 0.1f;
        private const float MouseSensitivity = 100f;

        protected readonly PositionTracking MouseTracking;

        public Rendering Rendering { get; set; }

        protected Window(Rendering rendering)
        {
            Rendering = rendering;
            MouseTracking = new PositionTracking();
        }

        [DllImport("user32.dll", EntryPoint = "GetKeyState", SetLastError = true)]
        private static extern int GetKeyState(int nVirtKey);

        private static bool IsNumLocked()
        {
            return ((ushort)GetKeyState(0x90) & 0xffff) != 0;
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
                    rendering.CameraForward(NavigationSpeed);
                    break;
                case ConsoleKey.NumPad0:
                    rendering.CameraForward(-NavigationSpeed);
                    break;
                case ConsoleKey.NumPad4:
                    rendering.CameraRight(-NavigationSpeed);
                    break;
                case ConsoleKey.NumPad6:
                    rendering.CameraRight(NavigationSpeed);
                    break;
                case ConsoleKey.NumPad8:
                    rendering.CameraUp(NavigationSpeed);
                    break;
                case ConsoleKey.NumPad2:
                    rendering.CameraUp(-NavigationSpeed);
                    break;
            }
        }

        protected void OnMouseMove()
        {
            if (Rendering is CameraRendering cameraRendering)
            {
                cameraRendering.Navigable = IsNumLocked();
                cameraRendering.CameraYaw = Limit(cameraRendering.CameraYaw + MouseTracking.DeltaX / MouseSensitivity, MathF.PI);
                cameraRendering.CameraPitch = Limit(cameraRendering.CameraPitch - MouseTracking.DeltaY / MouseSensitivity, MathF.PI / 2);
            }
        }
    }
}
