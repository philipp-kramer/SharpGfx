namespace SharpGfx
{
    public abstract class TextureHandle
    {
        public abstract void ActivateTexture(int unit);
        public abstract void DeleteTexture();
    }
}