package data;

public enum Face {
	
	E(1, 0),
	SE(1, 1),
	S(0, 1),
	SW(-1, 1),
	W(-1, 0),
	NW(-1, -1),
	N(0, -1),
	NE(1, -1);
	
	public final int X, Y;
	
	private Face(int X, int Y) {
		this.X = X;
		this.Y = Y;
	}
}
