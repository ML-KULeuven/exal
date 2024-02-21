package model;

import java.util.Random;

import data.Node;

public class Counter implements Model {
	
	public Counter(int size) {
		this.counts = new int[size][size];
		this.total = new int[size];
		for(int tile = 0; tile < size; ++tile) {
			for(int label = 0; label < size; ++label) counts[tile][label] = 1;
			total[tile] = size;
		}
	}
	
	// properties
	private int[][] counts;
	private int[] total;
	
	public int sample(Random random, Node node) {
		int count = random.nextInt(total[node.getTile()]);
		for(int label = 0; label < counts[node.getTile()].length; ++label) {
			count -= counts[node.getTile()][label];
			if(count < 0) return label;
		}
		return counts[node.getTile()].length - 1;
	}
	
	public int assign(Node node) {
		int mode = 0;
		for(int label = 0; label < counts[node.getTile()].length; ++label) {
			if(counts[node.getTile()][mode] < counts[node.getTile()][label]) mode = label;
		}
		return mode;
	}
	
	public void gradient(Node node, int label) {
		++counts[node.getTile()][label];
		++total[node.getTile()];
	}
	
	public void update(double rate) {}
	
	public void show(double[][] pixels) {
		for(int tile = 0; tile < pixels.length; ++tile) {
			System.out.print(tile + "   ->   c:");
			int mode = 0;
			for(int label = 0; label < counts[tile].length; ++label) {
				System.out.print(" " + String.format("%12d", counts[tile][label]));
				if(counts[tile][mode] < counts[tile][label]) mode = label;
			}
			System.out.print("   m: " + mode);
		}
	}
	
	public String predictions(double[][] pixels) {
		String[] predictions = new String[pixels.length];
		for(int tile = 0; tile < predictions.length; ++tile) {
			int mode = 0;
			for(int label = 0; label < counts[tile].length; ++label) if(counts[tile][mode] < counts[tile][label]) mode = label;
			predictions[tile] = Integer.toString(mode);
		}
		return String.join(",", predictions);
	}
}
