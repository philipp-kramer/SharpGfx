using SharpGfx.Primitives;

namespace SharpGfx.OpenGL.Shading
{
    public class ShadowTextureMaterial : TextureMaterial
    {
        private readonly Device _device;
        private readonly TextureHandle _shadowHandle;
        private readonly int _shadowUnit;

        public ShadowTextureMaterial(
            Device device,
            TextureHandle textureHandle,
            int textureUnit,
            TextureHandle shadowHandle,
            int shadowUnit,
            Color3 ambient)
            : base(
                device,
                Resources.GetSource("shadow_texture.vert"),
                Resources.GetSource("shadow_texture.frag"),
                textureHandle,
                textureUnit)
        {
            _device = device;
            _shadowHandle = shadowHandle;
            _shadowUnit = shadowUnit;

            DoInContext(() =>
            {
                Set("ambient", ambient.Vector);
                Set("shadowUnit", _shadowUnit);
            });
        }

        public void Update(Matrix4 lightViewProjection)
        {
            DoInContext(() =>
            {
                Set("lightViewProjection", lightViewProjection);
            });
        }

        public override void Apply()
        {
            _shadowHandle.ActivateTexture(_shadowUnit);
            base.Apply();
        }

        public override void UnApply()
        {
            base.UnApply();
            _device.ClearTexture(_shadowUnit);
        }

        protected override void Dispose(bool disposing)
        {
            _shadowHandle.DeleteTexture();
            base.Dispose(disposing);
        }
    }
}