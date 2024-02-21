package data;

import java.util.Random;

public class Synthetic extends Data {
	
	// constructor
	public Synthetic(Random random) {
		super(random);
	}
	
	// properties
	private double[][] pixels;
	private double noise;
	
	public Map getTrain() {
		Map map = new Map(dimension, window, faces);
		map.generate(random, costs, pixels, noise);
		map.burn(random, costs, burn);
		return map;
	}
	
	public void setPixels(int pixels) {
		this.pixels = new double[costs.length][pixels];
		for(int i = 0; i < costs.length; ++i) for(int j = 0; j < pixels; ++j) this.pixels[i][j] = random.nextGaussian();
	}
	
	public void setNoise(double noise) {
		this.noise = noise;
	}
}
