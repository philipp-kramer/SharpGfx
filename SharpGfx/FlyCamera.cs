using System;
using SharpGfx.Primitives;

namespace SharpGfx
{
    public class FlyCamera : Camera
    {
        private const float FlySpeed = 1f / 10;
        private const float MoveSensitivity = 1/250f;

        public FlyCamera(Space world, Point3 position)
            : base(world, position)
        {
        }

        public override void OnKeyDown(ConsoleKey key)
        {
            switch (key)
            {
                case ConsoleKey.D5:
                case ConsoleKey.NumPad5:
                case ConsoleKey.PageUp:
                case ConsoleKey.E:
                    if (Navigable) Position += FlySpeed * LookAt;
                    break;

                case ConsoleKey.D0:
                case ConsoleKey.NumPad0:
                case ConsoleKey.PageDown:
                case ConsoleKey.Q:
                    if (Navigable) Position -= FlySpeed * LookAt;
                    break;

                case ConsoleKey.D4:
                case ConsoleKey.NumPad4:
                case ConsoleKey.LeftArrow:
                case ConsoleKey.A:
                    if (Navigable) Position -= GetHorizontalDelta();
                    break;

                case ConsoleKey.D6:
                case ConsoleKey.NumPad6:
                case ConsoleKey.RightArrow:
                case ConsoleKey.D:
                    if (Navigable) Position += GetHorizontalDelta();
                    break;

                case ConsoleKey.D8:
                case ConsoleKey.NumPad8:
                case ConsoleKey.UpArrow:
                case ConsoleKey.W:
                    if (Navigable) Position += FlySpeed * World.Unit3Y;
                    break;

                case ConsoleKey.D2:
                case ConsoleKey.NumPad2:
                case ConsoleKey.DownArrow:
                case ConsoleKey.S:
                    if (Navigable) Position -= FlySpeed * World.Unit3Y;
                    break;
            }
        }

        public override void MouseMoving(IVector2 delta, MouseButtons buttonClicked)
        {
            switch (buttonClicked)
            {
                case MouseButtons.Left:
                    Yaw = Limit(Yaw + delta.X * MoveSensitivity, MathF.PI);
                    Pitch = Limit(Pitch - delta.Y * MoveSensitivity, MathF.PI / 2);
                    break;

                case MouseButtons.Middle:
                    Position -= delta.Y * LookAt;
                    break;
            }
        }

        private IVector3 GetHorizontalDelta()
        {
            return FlySpeed * IVector3.Cross(LookAt, World.Unit3Y).Normalized();
        }
    }
}