using SharpGfx.Primitives;

namespace SharpGfx.OpenTK.Materials
{
    public class ShadowTextureMaterial : OtkTextureMaterial
    {
        private readonly Device _device;
        private readonly TextureHandle _shadowHandle;
        private readonly int _shadowUnit;

        public ShadowTextureMaterial(
            Device device,
            TextureHandle handle,
            int textureUnit,
            TextureHandle shadowHandle,
            int shadowUnit,
            Color3 ambient,
            Matrix4 lightViewProjection)
            : base(
                device,
                Resources.Get<ShadowTextureMaterial>("Shaders.shadow_texture.vert"),
                Resources.Get<ShadowTextureMaterial>("Shaders.shadow_texture.frag"),
                handle,
                textureUnit)
        {
            _device = device;
            _shadowHandle = shadowHandle;
            _shadowUnit = shadowUnit;

            Shading.DoInContext(() =>
            {
                Shading.Set("lightViewProjection", lightViewProjection);
                Shading.Set("ambient", ambient.Vector);
                Shading.Set("shadowUnit", _shadowUnit);
            });
        }

        protected internal override void Apply(Point3 cameraPosition)
        {
            _shadowHandle.ActivateTexture(_shadowUnit);
            base.Apply(cameraPosition);
        }

        protected internal override void UnApply()
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