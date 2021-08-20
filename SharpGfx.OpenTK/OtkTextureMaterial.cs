using SharpGfx.Primitives;

namespace SharpGfx.OpenTK
{
    public class OtkTextureMaterial : OtkShadedMaterial
    {
        private readonly Device _device;
        private readonly TextureHandle _handle;
        private readonly int _unit;


        public OtkTextureMaterial(Device device, string vertexShader, string fragmentShader, TextureHandle handle, int unit)
            : base(vertexShader, fragmentShader)
        {
            _device = device;
            _handle = handle;
            _unit = unit;
            Shading.DoInContext(() => Shading.Set("texUnit", _unit));
        }

        public override void Apply(Point3 cameraPosition)
        {
            _handle.ActivateTexture(_unit);
        }

        public override void UnApply()
        {
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
