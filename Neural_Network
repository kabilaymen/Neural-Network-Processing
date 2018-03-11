int[] structure;
NeuralNetwork brain;
float[] inputs;
float[][] dinputs;
float[][] doutputs;
final int speed = 2000;
boolean train;

void setup() {
  size(800, 500);
  structure = new int[] { 2, 3, 1 };
  dinputs = new float[][] {{0, 0}, {1, 1}, {0, 1}, {1, 0}};
  doutputs = new float[][] {{1}, {1}, {1}, {0}};
  brain = new NeuralNetwork(structure);
  inputs = new float[structure[0]];
  for (int i = 0; i < inputs.length; i++)
    inputs[i] = round(random(1));
  train = false;
}

void draw() {
  background(250);

  brain.show();
  brain.feedforward(inputs);

  if (train) {
    train();
    // show(5);
  }
}

void mousePressed() {
  train = !train;
}

void keyPressed() {
  if (key == ' ') {
    inputs = new float[structure[0]];
    for (int i = 0; i < inputs.length; i++)
      inputs[i] = round(random(1));
  }
}

void show(int resolution) {
  stroke(0);
  strokeWeight(1);
  for (float y = 0; y < (float) height/resolution; y++) {
    for (float x = 0; x < (float) width/resolution; x++) {
      fill(map(brain.guess(new float[] { x*resolution/width, y*resolution/height })[0], 0, 1, 0, 255));
      rect(x*resolution, y*resolution, resolution, resolution);
    }
  }
}

void train() {
  if (train)
    for (int i = 0; i < speed; i++) {
      int b = (int) random(dinputs.length);
      // int b = speed % dinputs.length;
      // int b = 3;
      brain.train(dinputs[b], doutputs[b]);
    }
}
