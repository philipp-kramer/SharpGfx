using OpenTK.Graphics.OpenGL;

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
            var textureUnit = OtkTextures.GetTextureUnit(unit);
            GL.ActiveTexture(textureUnit);
            GL.BindTexture(TextureTarget.Texture2D, Handle);
        }

        public override void DeleteTexture()
        {
            GL.DeleteTexture(Handle);
        }
    }
}
