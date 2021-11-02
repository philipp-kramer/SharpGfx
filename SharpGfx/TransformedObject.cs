namespace SharpGfx
{
    internal sealed class TransformedObject : RenderObject
    {
        private readonly RenderObject _object;

        internal TransformedObject(RenderObject @object) 
            : base(@object.Space, @object.Material)
        {
            _object = @object;
            Transform = @object.Transform;
        }

        public override void Render()
        {
            var t = _object.Transform;
            _object.Transform = Transform;
            try
            {
                _object.Render();
            }
            finally
            {
                _object.Transform = t;
            }
        }
    }
}