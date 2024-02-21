package experiment;

import java.io.File;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Scanner;

public class Analysis {
	
	// properties
	private double[] time;
	private double[] accuracy;
	
	public void load(File directory) {
		File[] files = directory.listFiles();
		time = new double[files.length];
		accuracy = new double[files.length];
		
		for(int index = 0; index < files.length; ++index) {
			try {
				Map<String, String> map = new HashMap<String, String>();
				Scanner reader = new Scanner(files[index]);
				while(reader.hasNextLine()) {
					String[] line = reader.nextLine().strip().split("=");
					if(1 < line.length) map.put(line[0], line[1]);
				}
				reader.close();
				
				time[index] = Integer.parseInt(map.get("seconds"));
				accuracy[index] = Double.parseDouble(map.get("accuracy"));
			} catch(Exception e) {
				e.printStackTrace();
			}
		}
	}
	
	private void show(double[] data) {
		double mean = 0;
		for(int index = 0; index < data.length; ++index) mean += data[index];
		mean = mean / data.length;
		double std = 0;
		for(int index = 0; index < data.length; ++index) std += (data[index] - mean) * (data[index] - mean);
		std = Math.sqrt(std / data.length);
		System.out.print(String.format(Locale.US, "%.4f", mean) + " +- " + String.format(Locale.US, "%.4f", std));
	}
	
	public void show() {
		System.out.print("seconds: ");
		show(time);
		System.out.print("\naccuracy: ");
		show(accuracy);
		System.out.print("\n");
	}
	
	// main
	public static void main(String[] args) {
		int dimension = 12;
		String path = "<path-to-experiments>"; // path to where results are stored
		path += dimension + "x" + dimension;
		
		Analysis analysis = new Analysis();
		analysis.load(new File(path));
		analysis.show();
	}
}
