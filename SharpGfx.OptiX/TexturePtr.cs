using System;

namespace SharpGfx.OptiX;

public readonly struct TexturePtr
{
    public readonly IntPtr Texture;

    public TexturePtr(IntPtr texture)
    {
        Texture = texture;
    }
}