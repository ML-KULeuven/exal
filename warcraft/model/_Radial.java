package model;

import java.util.Random;

public class _Radial {
	
	// constructor
	private static final double SCALE = 100;
	private static final double STEP = 0.001;
	
	public _Radial(Random random, int size) {
		tx = random.nextDouble();
		ty = random.nextDouble();
		x = new double[size];
		y = new double[size];
		
		test = new int[size];
		
		for(int index = 0 ; index < size; ++index) {
			x[index] = random.nextDouble();
			y[index] = random.nextDouble();
		}
	}
	
	// properties
	private double tx, ty;
	private double[] x, y;
	
	private int[] test;
	
	public int getSample(Random random) {
		int sample = 0;
		double total = 0;
		for(int index = 0; index < x.length; ++index) {
			double weight = 1 / (1 + SCALE * ((x[index] - tx) * (x[index] - tx) + (y[index] - ty) * (y[index] - ty)));
			total += weight;
			if(total * random.nextDouble() < weight) sample = index;
		}
		return sample;
	}
	
	public void train(int index) {
		test[index]++;
		
		for(int i = 0; i < x.length; ++i) {
			double factor = i == index ? STEP : -STEP / (x.length - 1);
//			factor /= 1 + SCALE * ((x[i] - tx) * (x[i] - tx) + (y[i] - ty) * (y[i] - ty));
			x[index] += factor * (tx - x[index]);
			y[index] += factor * (ty - y[index]);
		}
	}
	
	public void show() {
		int mode = 0;
		double max = 0;
		System.out.print("w:");
		for(int index = 0; index < x.length; ++index) {
			double weight = 1 / (1 + SCALE * ((x[index] - tx) * (x[index] - tx) + (y[index] - ty) * (y[index] - ty)));
			System.out.print(" " + weight);
			if(max < weight) {
				mode = index;
				max = weight;
			}
		}
		System.out.print("     c:");
		for(int index = 0; index < test.length; ++index) System.out.print(" " + test[index]);
		System.out.print("     m: " + mode);
	}
}
