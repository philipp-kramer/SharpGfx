using System;
using SharpGfx.Primitives;

namespace SharpGfx
{
    public abstract class RenderObject : IDisposable
    {
        private Space _space;
        private readonly string _name;

        public Space Space
        {
            get => _space;
            set
            {
                if (Space.Domain <= value.Domain)
                {
                    _space = value;
                }
                else
                {
                    throw new InvalidOperationException("the new space must not be upstream in the pipeline");
                }
            }
        }

        public Material Material { get; }
        public Matrix4 Transform { get; set; }

        protected RenderObject(Space space, string name, Material material)
        {
            _space = space;
            _name = name;
            Material = material;
            Transform = space.Identity4;
        }

        public abstract void Render();

        public RenderObject Copy(string name)
        {
            return new TransformedObject(this, name);
        }

        public RenderObject Scale(float scale)
        {
            Transform *= Space.Scale4(scale);
            return this;
        }

        public RenderObject Scale(IVector3 scale)
        {
            Transform *= scale.Space.Scale4(scale);
            return this;
        }

        public RenderObject Translate(IVector3 delta)
        {
            if (Space.Domain > delta.Space.Domain) throw new ArgumentException("cross space operation");
            Space = delta.Space;
            Transform *= delta.Space.Translation4(delta);
            return this;
        }

        public RenderObject RotateX(float angle)
        {
            Transform *= Space.RotationX4(angle);
            return this;
        }

        public RenderObject RotateY(float angle)
        {
            Transform *= Space.RotationY4(angle);
            return this;
        }

        public RenderObject RotateZ(float angle)
        {
            Transform *= Space.RotationZ4(angle);
            return this;
        }

        public override string ToString()
        {
            return _name;
        }

        public abstract void Dispose();
    }
}