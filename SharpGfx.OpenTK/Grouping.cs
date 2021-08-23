using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace SharpGfx.OpenTK
{
    public class Grouping<TKey, TElement> : IGrouping<TKey, TElement>
    {
        private readonly IEnumerable<TElement> _values;

        public Grouping(TKey key, IEnumerable<TElement> values)
        {
            Key = key;
            _values = values ?? throw new ArgumentNullException("values");
        }

        public TKey Key { get; }

        public IEnumerator<TElement> GetEnumerator()
        {
            return _values.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
