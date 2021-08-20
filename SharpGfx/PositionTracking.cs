namespace SharpGfx
{
    public class PositionTracking
    {
        public int X { get; set; }
        public int Y { get; set; }

        public int DeltaX { get; private set; }
        public int DeltaY { get; private set; }

        public void Update(int x, int y)
        {
            DeltaX = x - X;
            DeltaY = y - Y;
            X = x;
            Y = y;
        }
    }
}
