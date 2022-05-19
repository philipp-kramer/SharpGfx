using System.Drawing;
using System.IO;

namespace SharpGfx
{
    public readonly struct Image
    {
        public readonly byte[] PixelColors;
        public readonly int Width;
        public readonly int Height;

        public Image(byte[] pixelColors, int width, int height)
        {
            PixelColors = pixelColors;
            Width = width;
            Height = height;
        }

        public Image(Stream stream)
        {
            using var bitmap = new Bitmap(stream);

            Width = bitmap.Width;
            Height = bitmap.Height;
            PixelColors = new byte[Width * Height * 4];
            int index = 0;
            for (int y = 0; y < Height; y++)
            {
                for (int x = 0; x < Width; x++)
                {
                    var pixel = bitmap.GetPixel(x, y);
                    PixelColors[index++] = pixel.R;
                    PixelColors[index++] = pixel.G;
                    PixelColors[index++] = pixel.B;
                    PixelColors[index++] = pixel.A;
                }
            }
        }
    }
}