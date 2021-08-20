namespace SharpGfx.OpenTK
{
    public abstract class OtkShadedMaterial : Material
    {
        protected internal OtkShading Shading { get; }

        protected OtkShadedMaterial(string vertexShader, string fragmentShader)
        {
            Shading = new OtkShading(vertexShader, fragmentShader);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                Shading?.Dispose();
            }
        }
    }
}