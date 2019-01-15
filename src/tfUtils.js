'use strict';

var assert = require('assert');
var tf = require('./tf');

// TODO: This probably ought to be thought of as a way of getting a
// scalar off of the e.g. gpu in order to perform cpu computation with
// regular JS math ops. This probably shouldn't be used to stop
// gradients flowing through a sub-computation -- use a dedicated
// method for that.
function toNumber(x) {
  if (typeof(x) === 'number') {
    return x;
  }
  else if (x instanceof tf.Tensor) {
    assert.ok(x.rank === 0); // is this check sufficient?
    return x.dataSync()[0];
  }
  else {
    throw 'Unexpected argument type';
  }
}

// TODO: Figure out a sensible way to do this.

// Ideally we want to be able to avoid backprop altogether (e.g.
// through samplers) rather than just propagating zeros.

function stopGrad(t) {
  // This is probably really horrible, especially on the gpu where I
  // guess we copy the entire buffer from the gpu to main memory and
  // then copy it back again.
  return t.buffer().toTensor();
};

module.exports = {
  toNumber,
  stopGrad
};
