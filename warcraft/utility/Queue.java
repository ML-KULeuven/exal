package utility;

import data.Node;

public class Queue {
	
	// constructor
	public Queue(int size) {
		queue = new Node[size + 1];
	}
	
	// properties
	private Node[] queue;
	private int size;
	
	public void clear() {
		for(int index = 0; index <= size; ++index) queue[index] = null;
		size = 0;
	}
	
	public void add(Node node) {
		queue[++size] = node;
		int child = size;
		int parent = child >> 1;
		while(0 < parent && queue[child].getTotal() < queue[parent].getTotal()) {
			queue[0] = queue[child];
			queue[child] = queue[parent];
			queue[parent] = queue[0];
			child = parent;
			parent = parent >> 1;
		}
	}
	
	public Node get() {
		if(size <= 0) return null;
		Node node = queue[1];
		queue[1] = queue[size];
		
		int parent = 1;
		int child = parent << 1;
		if(child + 1 <= size && queue[child + 1].getTotal() < queue[child].getTotal()) ++child;
		while(child <= size && queue[child].getTotal() < queue[parent].getTotal()) {
			queue[0] = queue[child];
			queue[child] = queue[parent];
			queue[parent] = queue[0];
			parent = child;
			child = child << 1;
			if(child + 1 <= size && queue[child + 1].getTotal() < queue[child].getTotal()) ++child;
		}
		if(--size <= 0) queue[0] = null;
		return node;
	}
}
