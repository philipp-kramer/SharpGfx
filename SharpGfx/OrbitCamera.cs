using System;
using SharpGfx.Primitives;

namespace SharpGfx
{
    public class OrbitCamera : Camera
    {
        private const float MoveSensitivity = 1f/250;
        private const float ZoomSensitivity = 100 * MoveSensitivity;

        private float _radius;

        public Point3 Center { get; private set; }

        public OrbitCamera(Space world, Point3 center, float radius)
            : base(world, center - radius * world.Vector3(0, 0, -1))
        {
            Center = center;
            _radius = radius;
        }

        public override void OnKeyDown(ConsoleKey key)
        {
        }

        public override void MouseMoving(IVector2 delta, MouseButtons buttonClicked)
        {
            switch (buttonClicked)
            {
                case MouseButtons.Left:
                    Pitch -= MoveSensitivity * delta.Y;
                    Yaw += MoveSensitivity * delta.X;
                    break;

                case MouseButtons.Middle:
                    _radius += ZoomSensitivity * delta.Y;
                    break;

                case MouseButtons.Right:
                    var lookAtX = LookAt.Cross(World.Unit3Y);
                    var lookAtY = LookAt.Cross(lookAtX);
                    Center -= MoveSensitivity * (delta.X * lookAtX + delta.Y * lookAtY);
                    break;
            }

            UpdatePosition();
        }

        private void UpdatePosition()
        {
            Position = Center - _radius * LookAt;
        }
    }
}