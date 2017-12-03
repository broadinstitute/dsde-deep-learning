/**
 * Convolution
 * by <a href="http://www.shiffman.net">Daniel Shiffman</a>.  
 * adapted by sam@broadinstitute.org
 * 
 * Applys a convolution matrix to a portion of the index.  
 * Move mouse to apply filter to different parts of the image. 
 * A few different 2d convolutional filters to choose from.
 */

PImage img;
int w = 260;
int margin = 10;

// It's possible to convolve the image with 
// many different matrices
float[][] gauss = {  { 1.0/273.0, 4.0/273.0, 7.0/273.0, 4.0/273.0, 1.0/273.0 },
                     { 4.0/273.0, 16.0/273.0, 26.0/273.0, 16.0/273.0, 4.0/273.0 },
                     { 7.0/273.0, 26.0/273.0, 41.0/273.0, 26.0/273.0, 7.0/273.0 },
                     { 4.0/273.0, 16.0/273.0, 26.0/273.0, 16.0/273.0, 4.0/273.0 },
                     { 1.0/273.0, 4.0/273.0, 7.0/273.0, 4.0/273.0, 1.0/273.0 }}; 

float[][] mexican_hat = { { 0, 0, 1, 0, 0 },
                     { 0, 1, 2, 1, 0 },
                     { 1, 2, -16, 2, 1 },
                     { 0, 1, 2, 1, 0 },
                     { 0, 0, 1, 0, 0 }}; 

float[][] sobel_y = {{ 0, 0, 0, 0, 0 },
                     { 0, -1, -2, -1, 0 },
                     { 0, 0, 0, 0, 0 },
                     { 0, 1, 2, 1, 0 },
                     { 0, 0, 0, 0, 0 }}; 

float[][] sobel_x = {{ 0, 0, 0, 0, 0 },
                     { 0, -1, 0, 1, 0 },
                     { 0, -2, 0, 2, 0 },
                     { 0, -1, 0, 1, 0 },
                     { 0, 0, 0, 0, 0 }}; 
                     
float[][] toeplitz = {{0, 1, 0, 0, 0 },
                     { -1, 0, 1, 0, 0 },
                     { 0, -1, 0, 1, 0 },
                     { 0, 0, -1, 0, 1 },
                     { 0, 0, 0, -1, 0 }};                      


float[][] hankel =  {{ 0,  0,  0,  1,  0 },
                     { 0,  0,  1,  0, -1 },
                     { 0,  1,  0, -1,  0 },
                     { 1,  0, -1,  0,  0 },
                     { 0, -1,  0,  0,  0 }};    

float[][] block_avg = {{ 0.04, 0.04, 0.04, 0.04, 0.04 },
                     { 0.04, 0.04, 0.04, 0.04, 0.04 },
                     { 0.04, 0.04, 0.04, 0.04, 0.04 },
                     { 0.04, 0.04, 0.04, 0.04, 0.04 },
                     { 0.04, 0.04, 0.04, 0.04, 0.04 }}; 

float[][] matrix;

void setup() {
  size(2200, 900);
  textSize(30);
  frameRate(30);
  img = loadImage("basenji.jpg");
  matrix = block_avg;
}

void draw() {
  // We're only going to process a portion of the image
  // so let's set the whole image as the background first
  background(80);
  image(img,margin,margin);
  // Where is the small rectangle we will process
  int xstart = constrain(mouseX-w/2,0,img.width);
  int ystart = constrain(mouseY-w/2,0,img.height);
  int xend = constrain(mouseX+w/2,0,img.width);
  int yend = constrain(mouseY+w/2,0,img.height);
  int matrixsize = 5;
  loadPixels();
  // Begin our loop for every pixel
  for (int x = xstart; x < xend; x++) {
    for (int y = ystart; y < yend; y++ ) {
      color c = convolution(x,y,matrix,matrixsize,img);
      int loc = x + y* width;
      pixels[loc] = c;
    }
  }
  updatePixels();
  show_kernel(img.width+50, 50, matrix, matrixsize);
  if (keyPressed) {
    if (key == '1' ) {
      matrix = block_avg;
    } else if (key == '2' ){
        matrix = mexican_hat;
    } else if (key == '3' ){
        matrix = sobel_x;
    } else if (key == '4' ){
        matrix = sobel_y;
    } else if (key == '5' ){
        matrix = toeplitz;
    } else if (key == '6' ){
        matrix = hankel;
    } else if (key == '7' ){
        matrix = gauss;
    }
    
    if (key == 'a'){
      img = loadImage("sunflower.jpg");
    } else if (key == 'd'){
      img = loadImage("end.jpg");
    } else if (key == 'b'){
      img = loadImage("basenji.jpg");
    } else if (key == 'p'){
      img = loadImage("palantir.png");
    } else if (key == 'c'){
      img = loadImage("sunflowerfield.jpg");
    }
  }
}

color convolution(int x, int y, float[][] matrix,int matrixsize, PImage img)
{
  float rtotal = 0.0;
  float gtotal = 0.0;
  float btotal = 0.0;
  int offset = matrixsize / 2;
  for (int i = 0; i < matrixsize; i++){
    for (int j= 0; j < matrixsize; j++){
      // What pixel are we testing
      int xloc = x+i-offset;
      int yloc = y+j-offset;
      int loc = xloc + img.width*yloc;
      // Make sure we haven't walked off our image, we could do better here
      loc = constrain(loc,0,img.pixels.length-1);
      // Calculate the convolution
      rtotal += (red(img.pixels[loc]) * matrix[i][j]);
      gtotal += (green(img.pixels[loc]) * matrix[i][j]);
      btotal += (blue(img.pixels[loc]) * matrix[i][j]);
    }
  }
  // Make sure RGB is within range
  rtotal = constrain(rtotal,0,255);
  gtotal = constrain(gtotal,0,255);
  btotal = constrain(btotal,0,255);
  // Return the resulting color
  return color(rtotal,gtotal,btotal);
}

void show_kernel(float x, float y, float[][] matrix,int matrixsize){
  float textx = 160;
  float texty = 90;
  for (int i = 0; i < matrixsize; i++){
    for (int j= 0; j < matrixsize; j++){
      // What pixel are we testing
      float xloc = x+i*textx;
      float yloc = y+j*texty;
      if(matrix[i][j] != 0.0) text(nf(matrix[i][j],0,0), xloc, yloc);
    }
  }
}