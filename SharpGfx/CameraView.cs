using System;
using SharpGfx.Primitives;

namespace SharpGfx
{
    public readonly struct CameraView
    {
        public readonly Point3 Eye;
        public readonly IVector3 LookAt;
        public readonly IVector3 Up;

        public CameraView(Device device, Point3 eye, IVector3 lookAt, IVector3 up)
        {
            if (!eye.Vector.In(device.World)) throw new ArgumentException("needs to be in world-space", nameof(eye));
            if (!lookAt.In(device.World)) throw new ArgumentException("needs to be in world-space", nameof(lookAt));
            if (!up.In(device.World)) throw new ArgumentException("needs to be in world-space", nameof(up));

            Eye = eye;
            LookAt = lookAt;
            Up = up;
        }
    }
}