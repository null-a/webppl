'use strict';

var fs = require('fs');
var assert = require('assert');
var _ = require('underscore');

// http://yann.lecun.com/exdb/mnist/

var baseDir = '/Users/paul/Downloads';

var loadImages = function(fn) {
  var buff = fs.readFileSync(fn);

  assert.strictEqual(buff.readUInt32BE(0), 2051);
  assert.strictEqual(buff.readUInt32BE(4), 60000);
  assert.strictEqual(buff.readUInt32BE(8), 28);
  assert.strictEqual(buff.readUInt32BE(12), 28);

  var offset = 16;
  var images = [];

  for (var i = 0; i < 60000; i++) {
    var image = [];
    for (var j = 0; j < 784; j++) {
      var pixelIntensity = buff[offset + (i * 784) + j];
      // Make binary.
      image.push(pixelIntensity < 128 ? 0 : 1);
    }
    images.push(image);
  }

  return images;
};

var loadLabels = function(fn) {
  var buff = fs.readFileSync(fn);

  assert.strictEqual(buff.readUInt32BE(0), 2049);
  assert.strictEqual(buff.readUInt32BE(4), 60000);

  var offset = 8;
  var labels = [];

  for (var i = 0; i < 60000; i++) {
    labels.push(buff[offset + i]);
  }

  return labels;
};

var showImage = function(image) {
  for (var k = 0; k < 28; k++) {
    console.log(image.slice(k * 28, (k + 1) * 28)
                .join('')
                .replace(/0/g, ' ')
                .replace(/1/g, '*'));
  }
};

var filterImages = function(images, labels, digit) {
  var filteredImages = [];
  for (var i = 0; i < 60000; i++) {
    if (labels[i] === digit) {
      filteredImages.push(images[i]);
    }
  }
  return filteredImages;
};


var images = loadImages(baseDir + '/train-images-idx3-ubyte');
//fs.writeFileSync('mnist_inputs.json', JSON.stringify(images));

var labels = loadLabels(baseDir + '/train-labels-idx1-ubyte');
//fs.writeFileSync('mnist_labels.json', JSON.stringify(labels));

// Extract data set of zeros and fives.

var zeros = filterImages(images, labels, 0).slice(0, 2000);
var fives = filterImages(images, labels, 5).slice(0, 2000);

var zerolabels = _.times(2000, _.constant(0));
var fivelabels = _.times(2000, _.constant(5));

var data = _.chain(_.zip(zeros.concat(fives), zerolabels.concat(fivelabels)))
      .shuffle()
      .unzip()
      .value();

var images2 = data[0];
var labels2 = data[1];

fs.writeFileSync('mnist_05_images.json', JSON.stringify(images2));
fs.writeFileSync('mnist_05_labels.json', JSON.stringify(labels2));
