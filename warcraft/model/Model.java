package model;

import java.util.Random;

import data.Node;

public interface Model {
	
	// properties
	public int sample(Random random, Node node);
	public int assign(Node node);
	public void gradient(Node node, int label);
	public void update(double rate);
	public void show(double[][] pixels);
	public String predictions(double[][] pixels);
}
