namespace SharpGfx;

public class ButtonState
{
    public bool Pressed { get; private set; }

    public void Down()
    {
        Pressed = true;
    }

    public void Up()
    {
        Pressed = false;
    }
}