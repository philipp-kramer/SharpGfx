namespace SharpGfx.OpenGL
{
    internal class OglTextureHandle : TextureHandle
    {
        internal readonly uint Handle;

        public OglTextureHandle(uint handle)
        {
            Handle = handle;
        }

        public override void ActivateTexture(int unit)
        {
            GL.ActiveTexture(GlTextureUnit.Texture0 + unit);
            GL.BindTexture(GlTextureTarget.Texture2D, Handle);
        }

        public override void DeleteTexture()
        {
            GL.DeleteTexture(Handle);
        }
    }
}
