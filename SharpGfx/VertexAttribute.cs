using System;

namespace SharpGfx
{
    public readonly struct VertexAttribute
    {
        public readonly string Channel;
        public readonly Array Values;
        public readonly int Size;
        public readonly int Stride;
        public readonly int Offset;

        public VertexAttribute(string channel, Array values, int size, int stride, int offset)
        {
            Channel = channel;
            Values = values;
            Size = size;
            Stride = stride;
            Offset = offset;
        }

        public VertexAttribute(string channel, Array values, int size)
            : this(channel, values, size, size, 0)
        {
        }
    }
}