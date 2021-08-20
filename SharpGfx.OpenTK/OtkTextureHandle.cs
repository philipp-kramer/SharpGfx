namespace SharpGfx.OpenTK
{
    internal class OtkTextureHandle : TextureHandle
    {
        internal readonly int Handle;

        public OtkTextureHandle(int handle)
        {
            Handle = handle;
        }

        public override void ActivateTexture(int unit)
        {
            OtkTextures.ActivateTexture(this, unit);
        }

        public override void DeleteTexture()
        {
            OtkTextures.DeleteTexture(this);
        }
    }
}
