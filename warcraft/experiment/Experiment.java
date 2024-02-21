package experiment;

import java.io.File;
import java.io.FileWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Locale;
import java.util.Random;

import data.Data;
import data.Face;
import data.Map;
import data.Synthetic;
import model.Learner;
import model.Model;
import model.Network;

public class Experiment {
	
	// constructor
	public Experiment(Random random, Data data, Model model, Learner learner) {
		this.data = data;
		this.model = model;
		this.learner = learner;
		this.random = random;
	}
	
	// properties
	private Data data;
	private Model model;
	private Learner learner;
	private Random random;
	
	private int maps;
	private int train;
	private int test;
	
	private long time;
	private double accuracy;
	
	public void run() {
		this.time = System.currentTimeMillis();
		for(int run = 0; run < maps; ++run) {
			Map map = data.getTrain();
			for(int train = this.train; 0 < train;) if(map.train(random, learner, data.getCosts())) --train;
		}
		this.time = (System.currentTimeMillis() - time) / 1000L;
		
		int accuracy = 0;
		for(int run = 0; run < test; ++run) {
			Map map = data.getTest();
			if(map.test(model, data.getCosts())) ++accuracy;
		}
		this.accuracy = ((double) accuracy) / test;
	}
	
	public void save(File file) {
		try {
			FileWriter writer = new FileWriter(file);
			writer.write("dimension=" + data.getDimension() + "\n");
			writer.write("seconds=" + time + "\n");
			writer.write("accuracy=" + String.format(Locale.US, "%.4f", accuracy) + "\n");
			writer.close();
		} catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public void setModel(Model model) {
		this.model = model;
	}
	
	public void setLearner(Learner learner) {
		this.learner = learner;
	}
	
	public void setRandom(Random random) {
		this.random = random;
	}
	
	public void setMaps(int maps) {
		this.maps = maps;
	}
	
	public void setTrain(int train) {
		this.train = train;
	}
	
	public void setTest(int test) {
		this.test = test;
	}
	
	// main
	public static void main(String[] args) {
		
		for(int run = 5; 0 < run; --run) {
			int costs = 5;
			int pixels = 8 * 8 * 3;
			Face[] faces = {Face.E, Face.S, Face.W, Face.N, Face.SE, Face.SW, Face.NW, Face.NE};
			Random random = new Random();
			
			Data data = new Synthetic(random); // use Loader class to read real data
			data.setDimension(12);
			data.setWindow(5);
			data.setFaces(faces);
			data.setCosts(costs);
			data.setPixels(pixels);
			data.setBurn(100);
			
			Model model = new Network(pixels, costs);
			Learner learner = new Learner(model);
			learner.setBatch(100);
			learner.setRate(0.000001);
			
			Experiment experiment = new Experiment(random, data, model, learner);
			experiment.setMaps(100000);
			experiment.setTrain(300);
			experiment.setTest(10000);
			
			String path = "<path-to-experiments>"; // path to where results should be stored
			path += data.getDimension() + "x" + data.getDimension() + "/";
			path += DateTimeFormatter.ofPattern("yyyy.MM.dd-HH.mm.ss").format(LocalDateTime.now());
			path += ".txt";
			
			experiment.run();
			experiment.save(new File(path));
		}
	}
}
