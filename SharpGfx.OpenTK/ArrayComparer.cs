using System.Collections.Generic;
using System.Linq;

namespace SharpGfx.OpenTK
{
    public class ArrayComparer<T> : IEqualityComparer<T[]> 
        where T : struct
    {
        public bool Equals(T[] x, T[] y)
        {
            return x == null && y == null || x != null && y != null && x.SequenceEqual(y);
        }

        public int GetHashCode(T[] obj)
        {
            return obj[0].GetHashCode() ^ obj[^1].GetHashCode();
        }
    }
}