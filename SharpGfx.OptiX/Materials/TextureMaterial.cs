using System;

namespace SharpGfx.OptiX.Materials;

public class TextureMaterial : OptixMaterial
{
    private readonly OptixTextureHandle _textureHandle;

    public TextureMaterial(MaterialPtr handle, TextureHandle texture)
        : base(handle)
    {
        _textureHandle = (OptixTextureHandle) texture;
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _textureHandle.Dispose();
        }
        base.Dispose(disposing);
    }
}