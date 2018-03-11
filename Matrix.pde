static class Matrix {
  int w, h;
  float[][] data;

  Matrix(int w, int h) {
    this.w = w;
    this.h = h;
    data = new float[h][w];
  }

  Matrix(int s) {
    this.w = 1;
    this.h = s;
    data = new float[h][w];
  }

  Matrix(float[] inputs) {
    this.w = 1;
    this.h = inputs.length;
    data = new float[h][w];
    for (int i = 0; i < h; i++)
      data[i][0] = inputs[i];
  }

  void randomize() {
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
        data[y][x] = (float) Math.random() * 2 - 1;
  }

  float fromTo(int s, int e) {
    return data[e][s];
  }

  static Matrix dotMV(Matrix w, Matrix xm) {
    if (xm.h == w.w) {
      Matrix mat2 = new Matrix(xm.w, w.h);
      for (int i = 0; i < mat2.h; i++) {
        for (int j = 0; j < mat2.w; j++) {
          float sum = 0;
          for (int x = 0; x < xm.h; x++) {
            sum += w.data[i][x] * xm.data[x][j];
          }
          mat2.data[i][j] = sum;
        }
      }
      // print(mat2);
      return mat2;
    }   
    return null;
  }

  static Matrix add(Matrix mat, Matrix mat3) {
    Matrix mat2 = new Matrix(mat.w, mat.h);
    for (int y = 0; y < mat.h; y++)
      for (int x = 0; x < mat.w; x++)
        mat2.data[y][x] = mat.data[y][x] + mat3.data[y][x];
    return mat2;
  }

  void add(Matrix mat) {
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
        data[y][x] += mat.data[y][x];
  }

  void add(float m) {
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
        data[y][x] += m;
  }

  static Matrix add(Matrix mat, float n) {
    Matrix mat2 = new Matrix(mat.w, mat.h);
    for (int y = 0; y < mat.h; y++)
      for (int x = 0; x < mat.w; x++)
        mat2.data[y][x] = mat.data[y][x] + n;
    return mat2;
  }
  
  void multiply(Matrix mat) {
    for (int y = 0; y < mat.h; y++)
      for (int x = 0; x < mat.w; x++)
        data[y][x] *= mat.data[y][x];
  }

  static Matrix multiply(Matrix mat, Matrix mat3) {
    Matrix mat2 = new Matrix(mat.w, mat.h);
    for (int y = 0; y < mat.h; y++)
      for (int x = 0; x < mat.w; x++)
        mat2.data[y][x] = mat.data[y][x] * mat3.data[y][x];
    return mat2;
  }

  static Matrix substract(Matrix mat, Matrix mat3) {
    Matrix mat2 = new Matrix(mat.w, mat.h);
    for (int y = 0; y < mat.h; y++)
      for (int x = 0; x < mat.w; x++)
        mat2.data[y][x] = mat.data[y][x] - mat3.data[y][x];
    return mat2;
  }
  
  void scale(float n) {
    for (int y = 0; y < h; y++)
      for (int x = 0; x < w; x++)
        data[y][x] *= n;
  }


  static Matrix scale(Matrix mat, float n) {
    Matrix mat2 = new Matrix(mat.w, mat.h);
    for (int y = 0; y < mat.h; y++)
      for (int x = 0; x < mat.w; x++)
        mat2.data[y][x] = mat.data[y][x] * n;
    return mat2;
  }

  static Matrix squash(Matrix mat) {
    Matrix mat2 = new Matrix(mat.w, mat.h);
    for (int y = 0; y < mat.h; y++)
      for (int x = 0; x < mat.w; x++)
        mat2.data[y][x] = sigmoid(mat.data[y][x]);
    return mat2;
  }

  static Matrix transpose(Matrix mat) {
    Matrix mat2 = new Matrix(mat.h, mat.w);
    for (int y = 0; y < mat.w; y++)
      for (int x = 0; x < mat.h; x++)
        mat2.data[y][x] = sigmoid(mat.data[x][y]);
    return mat2;
  }

  static void print(String b, Matrix m) {
    println(b+" : ");
    for (int y = 0; y < m.h; y++) {
      System.out.print("[ ");
      for (int x = 0; x < m.w; x++)
        System.out.print(m.data[y][x] + " ");
      println(" ]");
    }
    println();
  }
  
  static void print(Matrix m) {
    for (int y = 0; y < m.h; y++) {
      System.out.print("[ ");
      for (int x = 0; x < m.w; x++)
        System.out.print(m.data[y][x] + " ");
      println(" ]");
    }
    println();
  }
}
