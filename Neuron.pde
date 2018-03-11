final int dx = 60;
final int dy = 50;
final int s = 60;

final int INPUT = 0;
final int OUTPUT = 1;
final int INPUT_OUTPUT = 2;

class Neuron {
  float x, y, v;
  Layer parent;
  int type;

  Neuron(Layer parent, int type) {
    this.parent = parent;
    this.type = type;
  }

  float b = 0, k = 0, q = 0;
  void show(int index) {
    q = map(v, 0, 1, 255, 0);
    if (q < 255/2) stroke(0);
    else stroke(255);
    fill(map(v, 0, 1, 0, 255));
    ellipseMode(CORNER);
    strokeWeight(2);
    b = parent.neurons.length;
    k = parent.parent.layers.length;
    x = (width-k*s-(k-1)*dx)/2+(s+dx)*index;
    y = (height-((b*s+(b-1)*dy)))/2+(s+dy)*parent.indexOf(this);
    ellipse(x, y, s, s);
    if (q < 255/2) fill(0);
    else fill(255);
    textFont(createFont("Comic Sans MS", 17));
    text(nf(v, 1, 2), x+(s-textWidth(nf(v, 1, 2)))/2, y+s/2+textAscent()/2-2);
  }
}

class Layer {
  NeuralNetwork parent;
  Neuron[] neurons;
  int index;

  Layer(NeuralNetwork parent, int index, int size, int type) {
    this.parent = parent;
    this.index = index;
    neurons = new Neuron[size];
    for (int i = 0; i < size; i++)
      neurons[i] = new Neuron(this, type);
  }

  void show() {
    for (int i = 0; i < neurons.length; i++)
      neurons[i].show(index);
  }

  void connect(Layer l, Matrix weights) {
    /* strokeWeight(map(abs(weights[y][x]), 0, 1, 0, 2));
     if (weights[y][x] > 0) stroke(255);
     else stroke(0);*/
    strokeWeight(2);
    for (int y = 0; y < neurons.length; y++) {
      for (int x = 0; x < l.neurons.length; x++) {
        stroke(map(weights.fromTo(y, x), -1, 1, 255, 0), map(weights.fromTo(y, x), -1, 1, 0, 255), 0);
        line(neurons[y].x+s/2, neurons[y].y+s/2, l.neurons[x].x+s/2, l.neurons[x].y+s/2);
      }
    }
  }

  void setInputs(float[] inputs) {
    for (int i = 0; i < neurons.length; i++)
      neurons[i].v = inputs[i];
  }

  int indexOf(Neuron n) {
    for (int i = 0; i < neurons.length; i++)
      if (neurons[i].equals(n)) return i;
    return -1;
  }
}

class NeuralNetwork {
  Layer[] layers;
  Matrix[] values;
  Matrix[] weights;
  Matrix[] biases;
  final float lr = 0.1;

  NeuralNetwork(int[] layer_s) {
    layers = new Layer[layer_s.length];
    for (int i = 0; i < layers.length; i++) {
      if (i == 0) layers[i] = new Layer(this, i, layer_s[i], INPUT);
      else if (i == layers.length-1) layers[i] = new Layer(this, i, layer_s[i], OUTPUT);
      else layers[i] = new Layer(this, i, layer_s[i], INPUT_OUTPUT);
    }
    weights = new Matrix[layer_s.length-1];
    values = new Matrix[layer_s.length];
    for (int i = 0; i < layer_s.length-1; i++) {
      weights[i] = new Matrix(layer_s[i], layer_s[i+1]);
      weights[i].randomize();
    }
    biases = new Matrix[layer_s.length];
    for (int i = 0; i < biases.length; i++) {
      values[i] = new Matrix(layer_s[i]);
      biases[i] = new Matrix(layer_s[i]);
      biases[i].randomize();
    }
  }

  void show() {
    for (int i = 0; i < layers.length-1; i++)
      layers[i].connect(layers[i+1], weights[i]);
    for (int i = 0; i < layers.length; i++)
      layers[i].show();

    // println(weights[0][0][0], weights[0][1][0]);
    // println(layers[0].bias);
    // println(layers[0].neurons[0].v, layers[0].neurons[1].v);
    // println(sigmoid(weights[0][0][0]*layers[0].neurons[0].v+weights[0][1][0]*layers[0].neurons[1].v+layers[1].bias));
  }

  private void updateValues(float[] inputs) {
    layers[0].setInputs(inputs);
    for (int i = 0; i < layers.length; i++)
      for (int j = 0; j < layers[i].neurons.length; j++)
        values[i].data[j][0] = layers[i].neurons[j].v;
  }

  float[] guess(float[] inputs) {
    Matrix[] data = new Matrix[layers.length];
    data[0] = new Matrix(inputs);
    for (int i = 1; i < layers.length; i++)
      data[i] = Matrix.squash(Matrix.add(Matrix.dotMV(weights[i-1], data[i-1]), biases[i]));
    float[] outputs = new float[layers[layers.length-1].neurons.length];
    for (int i = 0; i < layers[layers.length-1].neurons.length; i++)
      outputs[i] = data[layers.length-1].data[i][0];
    return outputs;
  }

  void feedforward(float[] inputs) {
    for (int i = 1; i < layers.length; i++) {
      updateValues(inputs);
      Matrix output = Matrix.squash(Matrix.add(Matrix.dotMV(weights[i-1], values[i-1]), biases[i]));
      for (int j = 0; j < layers[i].neurons.length; j++) {
        layers[i].neurons[j].v = output.data[j][0];
      }
    }
  }

  void train(float[] dinputs, float[] doutputs) {
    /* The most important line */
    feedforward(dinputs);

    Matrix outputs = new Matrix(guess(dinputs));
    Matrix targets = new Matrix(doutputs);

    Matrix[] errors = new Matrix[layers.length];
    Matrix[] gradients = new Matrix[layers.length];
    Matrix[] deltas = new Matrix[layers.length-1];

    // Matrix.print("Old weights", weights[layers.length-2]);

    for (int i = layers.length-1; i > 0; i--) {
      if (i != layers.length-1) errors[i] = Matrix.dotMV(Matrix.transpose(weights[i]), errors[i+1]);
      else errors[i] = Matrix.substract(targets, outputs);

      gradients[i] = Matrix.multiply(values[i], Matrix.add(Matrix.scale(values[i], -1), 1));
      gradients[i].multiply(errors[i]);
      gradients[i].scale(lr);

      deltas[i-1] = Matrix.dotMV(gradients[i], Matrix.transpose(values[i-1]));
      weights[i-1].add(deltas[i-1]);
      biases[i].add(gradients[i]);

      // Matrix.print("Gradients", gradients[layers.length-1]);
      // Matrix.print("Transposed (layer-1)", Matrix.transpose(values[layers.length-2]));
      // Matrix.print("Deltas of last layer", Matrix.dotMV(gradients[layers.length-1], Matrix.transpose(values[layers.length-2])));
      // Matrix.print("New weights", weights[layers.length-2]);
    }
  }
}

static float sigmoid(float x) {
  return 1/(1+exp(-x));
}
