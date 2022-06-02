namespace SharpGfx.OpenGL
{
    internal class StateTracking
    {
        public bool Active { get; private set; }

        public void Activate()
        {
            Active = true;
        }

        public void Deactivate()
        {
            Active = false;
        }
    }
}