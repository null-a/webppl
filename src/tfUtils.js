'use strict';

var assert = require('assert');
var tf = require('@tensorflow/tfjs-core');

function toNumber(x) {
  if (typeof(x) === 'number') {
    return x;
  }
  else if (x instanceof tf.Tensor) {
    assert.ok(x.rank === 0); // is this check sufficient?
    return x.dataSync()[0]; // is this tolerable? (perhaps it's unavoidable?)
  }
  else {
    throw 'Unpexected argument type';
  }
}

module.exports = {
  toNumber
};
