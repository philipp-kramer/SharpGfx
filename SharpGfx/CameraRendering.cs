using System;
using System.Drawing;
using SharpGfx.Primitives;

namespace SharpGfx
{
    public class CameraRendering : Rendering
    {
        private float _cameraPitch;
        private float _cameraYaw;

        public Point3 CameraPosition { get; set; }
        public Vector3 CameraLookAt { get; private set; }
        public bool Navigable { get; set; } = true;

        protected CameraRendering(Device device, Size size, Color3 ambientColor, Point3 cameraPosition)
            : base(device, size, ambientColor)
        {
            CameraPosition = cameraPosition;
            _cameraYaw = -MathF.PI / 2;
        }

        public CameraView GetView()
        {
            return new CameraView(Device, CameraPosition, CameraLookAt, Device.World.Unit3Y);
        }

        public override void OnUpdateFrame()
        {
            CameraUpdate();
            base.OnUpdateFrame();
        }

        public override void OnRenderFrame()
        {
            CameraUpdate();
            Device.Render(Scene, Size, CameraPosition, AmbientColor.GetColor4(1));
        }

        #region steer

        public float CameraPitch
        {
            get => _cameraPitch;
            set
            {
                if (Navigable)
                {
                    _cameraPitch = value;
                    CameraUpdate();
                }
            }
        }

        public float CameraYaw
        {
            get => _cameraYaw;
            set
            {
                if (Navigable)
                {
                    _cameraYaw = value;
                    CameraUpdate();
                }
            }
        }

        public void CameraForward(float factor)
        {
            if (Navigable)
            {
                CameraPosition += CameraLookAt * factor;
                UpdateView();
            }
        }

        public void CameraRight(float factor)
        {
            if (Navigable)
            {
                CameraPosition += Vector3.Cross(CameraLookAt, Device.World.Unit3Y).Normalized() * factor;
                UpdateView();
            }
        }

        public void CameraUp(float factor)
        {
            if (Navigable)
            {
                CameraPosition += Device.World.Unit3Y * factor;
                UpdateView();
            }
        }

        #endregion

        private void CameraUpdate()
        {
            CameraLookAt = Device.World.Vector3(
                    MathF.Cos(CameraPitch) * MathF.Cos(_cameraYaw),
                    MathF.Sin(CameraPitch),
                    MathF.Cos(CameraPitch) * MathF.Sin(_cameraYaw))
                .Normalized();

            UpdateView();
        }

        private void UpdateView()
        {
            Device.SetCameraView(Scene, GetView());
        }
    }
}
