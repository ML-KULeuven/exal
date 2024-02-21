package model;

import data.Node;

public class Learner {
	
	public Learner(Model model) {
		this.model = model;
	}
	
	// properties
	private Model model;
	private int batch, update;
	private double rate;
	
	public void train(Node node, int label) {
		model.gradient(node, label);
		if(batch <= ++update) {
			model.update(rate);
			update = 0;
		}
	}
	
	public void setBatch(int batch) {
		this.batch = batch;
	}
	
	public void setRate(double rate) {
		this.rate = rate;
	}
}
