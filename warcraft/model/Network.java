package model;

import java.util.Locale;
import java.util.Random;

import data.Node;

public class Network implements Model {
	
	// constructor
	public Network(int input, int output) {
		y = new double[output];
		w = new double[output][input];
		gw = new double[output][input];
		q = new double[output];
		gq = new double[output];
	}
	
	// properties
	private double[] y;
	private double[][] w, gw;
	private double[] q, gq;
	
	private void evaluate(double[] pixel) {
		double[] a = new double[y.length];
		for(int i = 0; i < a.length; ++i) {
			a[i] = q[i];
			for(int j = 0; j < pixel.length; ++j) a[i] += w[i][j] * pixel[j];
		}
		
		for(int i = 0; i < y.length; ++i) {
			y[i] = 1;
			for(int j = 0; j < a.length; ++j) if(i != j) y[i] += Math.exp(a[j] - a[i]);
			y[i] = 1 / y[i];
		}
	}
	
	public int assign(Node node) {
		int mode = 0;
		double a, ma = 0;
		for(int i = 0; i < y.length; ++i) {
			a = q[i];
			for(int j = 0; j < node.getPixel().length; ++j) a += w[i][j] * node.getPixel()[j];
			if(i == 0 || ma < a) {
				ma = a;
				mode = i;
			}
		}
		return mode;
	}
	
	public int sample(Random random, Node node) {
		evaluate(node.getPixel());
		double total = random.nextDouble();
		for(int label = 0; label < y.length; ++label) {
			total -= y[label];
			if(total < 0) return label;
		}
		return y.length - 1;
	}
	
	public void gradient(Node node, int label) {
		evaluate(node.getPixel());
		for(int i = 0; i < w.length; ++i) {
			double g = y[i] * (i == label ? 1 - y[label] : -y[label]);
			for(int j = 0; j < w[i].length; ++j) gw[i][j] += g * node.getPixel()[j];
			gq[i] += g;
		}
	}
	
	public void update(double rate) {
		for(int i = 0; i < w.length; ++i) {
			for(int j = 0; j < w[i].length; ++j) {
				w[i][j] += rate * gw[i][j];
				gw[i][j] = 0;
			}
			q[i] += rate * gq[i];
			gq[i] = 0;
		}
	}
	
	public void show(double[][] pixels) {
		for(int tile = 0; tile < pixels.length; ++tile) {
			evaluate(pixels[tile]);
			System.out.print(tile + "   ->   p:");
			int mode = 0;
			for(int label = 0; label < y.length; ++label) {
				System.out.print(" " + String.format(Locale.US, "%.12f", y[label]));
				if(y[mode] < y[label]) mode = label;
			}
			System.out.print("   m: " + mode + "\n");
		}
	}
	
	public String predictions(double[][] pixels) {
		String[] predictions = new String[pixels.length];
		for(int tile = 0; tile < predictions.length; ++tile) {
			int mode = 0;
			double a, ma = 0;
			for(int i = 0; i < y.length; ++i) {
				a = q[i];
				for(int j = 0; j < pixels[tile].length; ++j) a += w[i][j] * pixels[tile][j];
				if(i == 0 || ma < a) {
					ma = a;
					mode = i;
				}
			}
			predictions[tile] = Integer.toString(mode);
		}
		return String.join(",", predictions);
	}
}
