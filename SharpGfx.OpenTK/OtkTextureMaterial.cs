﻿using SharpGfx.Primitives;

namespace SharpGfx.OpenTK
{
    public class OtkTextureMaterial : OtkShadedMaterial
    {
        private readonly Device _device;
        private readonly TextureHandle _handle;
        private readonly int _unit;


        public OtkTextureMaterial(Device device, string vertexShader, string fragmentShader, TextureHandle handle, int unit)
            : base(vertexShader, fragmentShader, true)
        {
            _device = device;
            _handle = handle;
            _unit = unit;
            Shading.DoInContext(() => Shading.Set("texUnit", _unit));
        }

        protected internal override void Apply(Point3 cameraPosition)
        {
            _handle.ActivateTexture(_unit);
            base.Apply(cameraPosition);
        }

        protected internal override void UnApply()
        {
            base.UnApply();
            _device.ClearTexture(_unit);
        }

        protected override void Dispose(bool disposing)
        {
            ReleaseUnmanagedResources();
            base.Dispose(disposing);
        }

        private void ReleaseUnmanagedResources()
        {
            _handle.DeleteTexture();
        }
    }
}
