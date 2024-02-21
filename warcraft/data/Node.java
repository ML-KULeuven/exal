package data;

import java.util.Random;

import utility.Queue;

public class Node {
	
	// constructor
	public Node(Map map, int x, int y) {
		this.map = map;
		this.x = x;
		this.y = y;
	}
	
	// properties
	private final Map map;
	private final int x, y;
	private double[] pixel;
	private int tile;
	
	private boolean path;
	private int total;
	private int cost;
	private Node next;
	
	public void generate(Random random, int[] costs, double[][] pixels, double noise) {
		this.tile = random.nextInt(costs.length);
		this.cost = costs[tile];
		this.pixel = new double[pixels[tile].length];
		for(int i = 0; i < pixel.length; ++i) this.pixel[i] = pixels[tile][i] + noise * random.nextGaussian();
	}
	
	public void generate(int[] costs, int pixels, String line) {
		String[] values = line.split(",");
		this.path = values[0].equals("1");
		this.cost = costs[path ? costs.length - 1 : 0];
		this.pixel = new double[pixels];
		for(int i = 0; i < pixels;) this.pixel[i] = Double.parseDouble(values[++i]);
	}
	
	public void reset() {
		this.total = 0;
		this.next = null;
	}
	
	public void update(Node next, Queue queue) {
		if(this.next != null) return;
		this.total = next.total + cost;
		this.next = next;
		queue.add(this);
	}
	
	public void update(Face[] faces, Queue queue) {
		for(Face face : faces) {
			Node node = map.getNode(x + face.X, y + face.Y);
			if(node != null) node.update(this, queue);
		}
	}
	
	public void setCost(int cost) {
		this.cost = cost;
	}
	
	public void setPath(int cost) {
		this.path = true;
		this.cost = cost;
		if(next != this) next.setPath(cost);
	}
	
	public boolean getPath() {
		if(!path) return false;
		if(next == this) return true;
		return next.getPath();
	}
	
	public int getTile() {
		return tile;
	}
	
	public double[] getPixel() {
		return pixel;
	}
	
	public int getCost() {
		return cost;
	}
	
	public int getTotal() {
		return total;
	}
}
