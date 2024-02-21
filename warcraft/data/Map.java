package data;

import java.io.File;
import java.util.Random;
import java.util.Scanner;

import model.Learner;
import model.Model;
import utility.Queue;

public class Map {
	
	// constructor
	public Map(int dimension, int window, Face[] faces) {
		this.faces = faces;
		this.dimension = dimension;
		this.window = window;
		this.nodes = new Node[dimension][dimension];
		for(int x = 0; x < dimension; ++x) for(int y = 0; y < dimension; ++y) nodes[x][y] = new Node(this, x, y);
	}
	
	// properties
	private int dimension;
	private int window;
	private Node[][] nodes;
	private Face[] faces;
	
	public void generate(Random random, int[] costs, double[][] pixels, double noise) {
		for(int x = 0; x < dimension; ++x) for(int y = 0; y < dimension; ++y) nodes[x][y].generate(random, costs, pixels, noise);
		getPath();
		for(int x = 0; x < dimension; ++x) for(int y = 0; y < dimension; ++y) nodes[x][y].setCost(costs[costs.length - 1]);
		nodes[dimension - 1][dimension - 1].setPath(costs[0]);
	}
	
	public void generate(Random random, int[] costs, int pixels, File file) {
		try {
			Scanner scanner = new Scanner(file);
			for(int x = 0; x < dimension; ++x) for(int y = 0; y < dimension; ++y) nodes[x][y].generate(costs, pixels, scanner.nextLine());
			scanner.close();
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public void burn(Random random, int[] costs, int burn) {
		while(0 < burn) if(train(random, null, costs)) --burn;
	}
	
	public boolean train(Random random, Learner learner, int[] costs) {
		Node node = nodes[random.nextInt(window)][random.nextInt(window)];
		int label = random.nextInt(costs.length);
		int cost = node.getCost();
		
		node.setCost(costs[label]);
		if(getPath()) {
			if(learner != null) learner.train(node, label);
			return true;
		} else {
			node.setCost(cost);
			return false;
		}
	}
	
	public boolean test(Model model, int[] costs) {
		for(int x = 0; x < dimension; ++x) for(int y = 0; y < dimension; ++y) nodes[x][y].setCost(costs[model.assign(nodes[x][y])]);
		return getPath();
	}
	
	public boolean getPath() {
		for(int x = 0; x < dimension; ++x) for(int y = 0; y < dimension; ++y) nodes[x][y].reset();
		Queue queue = new Queue(dimension * dimension);
		nodes[0][0].update(nodes[0][0], queue);
		for(Node current = queue.get(); current != null; current = queue.get()) current.update(faces, queue);
		return nodes[dimension - 1][dimension - 1].getPath();
	}
	
	public Node getNode(int x, int y) {
		if(x < 0 || dimension <= x || y < 0 || dimension <= y) return null;
		return nodes[x][y];
	}
	
	public void show() {
		for(int y = 0; y < dimension; ++y) {
			for(int x = 0; x < dimension; ++x) System.out.print(nodes[x][y].getTile() + " ");
			System.out.print("   ");
			for(int x = 0; x < dimension; ++x) System.out.print(nodes[x][y].getPath() ? "x " : ". ");
			System.out.print("\n");
		}
	}
}
