package model;

import java.util.Random;

public class _Layer {
	
	public _Layer(Random random, int size) {
		h = new double[size];
		y = new double[size];
		
		hw = new double[size][3];
		for(int i = 0; i < hw.length; ++i) for(int j = 0; j < hw[i].length; ++j) hw[i][j] = random.nextGaussian();
		hq = new double[size];
		for(int i = 0; i < hq.length; ++i) hq[i] = random.nextGaussian();
		yw = new double[size][size];
		for(int i = 0; i < yw.length; ++i) for(int j = 0; j < yw[i].length; ++j) yw[i][j] = random.nextGaussian();
		yq = new double[size];
		for(int i = 0; i < yq.length; ++i) yq[i] = random.nextGaussian();
	}
	
	// properties
	private double[] h;
	private double[] y;
	private double total;
	
	private double[][] hw;
	private double[] hq;
	private double[][] yw;
	private double[] yq;
	
	private void evaluate(double[] pixel) {
		for(int i = 0; i < h.length; ++i) {
			h[i] = hq[i];
			for(int j = 0; j < pixel.length; ++j) h[i] += hw[i][j] * pixel[j];
			h[i] = 1 / (1 + Math.exp(-h[i]));
		}
		
		total = 0;
		for(int i = 0; i < y.length; ++i) {
			y[i] = yq[i];
			for(int j = 0; j < h.length; ++j) y[i] += yw[i][j] * h[j];
			y[i] = Math.exp(y[i]);
			total += y[i];
		}
	}
	
	public int getSample(Random random, int tile, double[] pixel) {
		evaluate(pixel);
		total = total * random.nextDouble();
		for(int label = 0; label < y.length; ++label) {
			total -= y[label];
			if(total < 0) return label;
		}
		return y.length - 1;
	}
	
	public void train(int label, int tile, double[] pixel) {
		evaluate(pixel);
		dir_train(label, tile, pixel);
	}
	
	private void dir_train(int label, int tile, double[] pixel) {
		double lr = 0.001;
		for(int i = 0; i < yw.length; ++i) {
			double g = i == label ? lr : -lr * lr;
			for(int j = 0; j < yw[i].length; ++j) {
				yw[i][j] += g;
			}
			yq[i] += g;
		}
		
		for(int i = 0; i < hw.length; ++i) {
			double g = yw[label][i];
			for(int j = 0; j < hw[i].length; ++j) {
				g -= y[j] * yw[j][i];
			}
			g = 0 <= g ? lr : -lr * lr;
			
			for(int j = 0; j < hw[i].length; ++j) {
				hw[i][j] += g;
			}
			hq[i] += g;
		}
	}
	
//	private void grad_train(int label, int tile, double[] pixel) {
//		double lr = 0.001;
//		
//		for(int i = 0; i < yw.length; ++i) {
//			double g = lr * y[i] * (i == label ? 1 - y[label] : -y[label]);
//			for(int j = 0; j < yw[i].length; ++j) {
//				yw[i][j] += g * h[j];
//			}
//			yq[i] += g;
//		}
//		
//		for(int i = 0; i < hw.length; ++i) {
//			double g = yw[label][i];
//			for(int j = 0; j < hw[i].length; ++j) {
//				g -= y[j] * yw[j][i];
//			}
//			g = lr * y[label] * h[i] * (1 - h[i]) * g;
//			
//			for(int j = 0; j < hw[i].length; ++j) {
//				hw[i][j] += g * pixel[j];
//			}
//			hq[i] += g;
//		}
//	}
	
	public void show(int tile, double[] pixel) {
		evaluate(pixel);
		int mode = 0;
		System.out.print("c:");
		for(int label = 0; label < y.length; ++label) {
			System.out.print(" " + (y[label] / total));
			if(y[mode] < y[label]) mode = label;
		}
		System.out.print("     m: " + mode);
	}
}
