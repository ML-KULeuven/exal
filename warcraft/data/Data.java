package data;

import java.util.Random;

public abstract class Data {
	
	// constructor
	public Data(Random random) {
		this.random = random;
	}
	
	// properties
	protected Random random;
	protected Face[] faces;
	protected int dimension;
	protected int window;
	
	protected int[] costs;
	protected int pixels;
	protected int burn;
	
	public abstract Map getTrain();
	
	public int getDimension() {
		return dimension;
	}
	
	public int[] getCosts() {
		return costs;
	}
	
	public void setDimension(int dimension) {
		this.dimension = dimension;
	}
	
	public void setWindow(int window) {
		this.window = window < dimension ? window : dimension;
	}
	
	public void setFaces(Face[] faces) {
		this.faces = faces;
	}
	
	public void setCosts(int costs) {
		this.costs = new int[costs];
		for(int cost = 0; cost < costs;) this.costs[cost] = ++cost;
	}
	
	public void setCosts(int[] costs) {
		this.costs = costs;
	}
	
	public void setPixels(int pixels) {
		this.pixels = pixels;
	}
	
	public void setBurn(int burn) {
		this.burn = burn;
	}
	
	public Map getTest() {
		return getTrain();
	}
}
