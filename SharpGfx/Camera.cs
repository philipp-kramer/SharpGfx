using System;
using SharpGfx.Primitives;

namespace SharpGfx
{
    public abstract class Camera
    {
        private static readonly float HalfFrustumHeight = MathF.Tan(Rendering.FovY / 2);

        private float _pitch;
        private float _yaw;

        protected Space World { get; }
        protected bool Navigable { get; set; } = true;

        public Point3 Position { get; protected set; }
        public IVector3 LookAt { get; private set; }

        protected Camera(Space world, Point3 position)
        {
            World = world;
            Position = position;
            Yaw = -MathF.PI / 2;
        }

        public abstract void OnKeyDown(ConsoleKey key);
        public abstract void MouseMoving(IVector2 delta, MouseButtons buttonClicked);

        public float Pitch
        {
            get => _pitch;
            protected set
            {
                if (Navigable)
                {
                    _pitch = Limit(value, (0.5f - 1e-6f) * MathF.PI);
                    UpdateLookAt();
                }
            }
        }

        public float Yaw
        {
            get => _yaw;
            protected set
            {
                if (Navigable)
                {
                    _yaw = value;
                    UpdateLookAt();
                }
            }
        }

        private void UpdateLookAt()
        {
            LookAt = World.Vector3(
                    MathF.Cos(_pitch) * MathF.Cos(_yaw),
                    MathF.Sin(_pitch),
                    MathF.Cos(_pitch) * MathF.Sin(_yaw))
                .Normalized();
        }

        protected static float Limit(float value, float range)
        {
            return Math.Min(Math.Max(value, -range), range);
        }

        public (Point3, Point3, Point3, Point3) GetFrustum(IVector3 unitY, float aspect)
        {
            var up = (unitY - unitY.Dot(LookAt) * LookAt).Normalized();
            var right = LookAt.Cross(up);
            var center = Position + LookAt;
            float halfWidth = aspect * HalfFrustumHeight;

            var tl = center + HalfFrustumHeight * up - halfWidth * right;
            var tr = center + HalfFrustumHeight * up + halfWidth * right;
            var bl = center - HalfFrustumHeight * up - halfWidth * right;
            var br = center - HalfFrustumHeight * up + halfWidth * right;

            return (tl, tr, bl, br);
        }
    }
}