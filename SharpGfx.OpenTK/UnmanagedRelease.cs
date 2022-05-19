using System;
using System.Collections.Generic;

namespace SharpGfx.OpenTK
{
    internal static class UnmanagedRelease
    {
        private static readonly Queue<Action> Queue = new Queue<Action>();

        internal static void Add(Action pending)
        {
            Queue.Enqueue(pending);
        }

        internal static void ExecutePending()
        {
            while (Queue.Count > 0)
            {
                Queue.Dequeue()?.Invoke();
            }
        }
    }
}