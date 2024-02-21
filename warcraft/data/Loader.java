package data;

import java.io.File;
import java.util.Random;

public class Loader extends Data {
	
	public Loader(Random random) {
		super(random);
	}
	
	// properties
	private File[] files;
	private int index;
	
	public void setPath(String path) {
		this.files = new File(path).listFiles();
		this.index = -1;
	}
	
	public Map getTrain() {
		if(files.length <= ++index) index = 0;
		Map map = new Map(dimension, window, faces);
		map.generate(random, costs, pixels, files[index]);
		map.burn(random, costs, burn);
		return map;
	}
}
