'use strict';
var assert = require('assert');
var Tensor = require('adnn/tensor');
var ad = require('adnn/ad');
var nn = require('adnn/nn');

function puts(x) {
  console.log(x);
  return x;
};

Tensor.prototype.dot = function(t) {

  var a = this, b = t;
  assert.ok(a.rank === 2 && b.rank === 2);
  assert.ok(a.dims[1] === b.dims[0]);

  var l = a.dims[1];
  var h = a.dims[0], w = b.dims[1];
  var y = new Tensor([h, w]);

  for (var r = 0; r < h; r++) {
    for (var c = 0; c < w; c++) {
      var z = 0;
      for (var i = 0; i < l; i++) {
        z += a.data[r * l + i] * b.data[w * i + c];
      }
      y.data[r * w + c] = z;
    }
  }
  return y;
};

ad.tensor.dot = ad.newBinaryFunction({
  OutputType: Tensor,
  name: 'dot',
  forward: function(a, b) {
    return a.dot(b);
  },
  backward1: function(A, B) {

    var Ap = ad.value(A);
    var Bp = ad.value(B);

    var Ah = Ap.dims[0];
    var Aw = Ap.dims[1];
    var Bh = Bp.dims[0];
    var Bw = Bp.dims[1];
    var hout = Ah;
    var wout = Bw;

    for (var l = 0; l < Ah; l++) {
      for (var m = 0; m < Aw; m++) {
        var z = 0;
        for (var j = 0; j < wout; j++) {
          z += this.dx.data[l * wout + j] * Bp.data[m * Bw + j];
        }
        A.dx.data[l * Aw + m] += z;
      }
    }
  },
  backward2: function(A, B) {

    var Ap = ad.value(A);
    var Bp = ad.value(B);

    var Ah = Ap.dims[0];
    var Aw = Ap.dims[1];
    var Bh = Bp.dims[0];
    var Bw = Bp.dims[1];
    var hout = Ah;
    var wout = Bw;

    for (var l = 0; l < Bh; l++) {
      for (var m = 0; m < Bw; m++) {
        var z = 0;
        for (var i = 0; i < Ah; i++) {
          z += this.dx.data[i * wout + m] * Ap.data[i * Aw + l];
        }
        B.dx.data[l * Bw + m] += z;
      }
    }

  }
});

// Test 1.
// var f = function(x) {
//   // x^2 + x
//   return ad.tensor.add(ad.tensor.dot(x, x), x);
// };

// var x = ad.lift(new Tensor([1, 1]).fromArray([[2]]));
// var y = f(x);
// y.backprop(); // 2x + 1
// puts(y);
// puts(x);

// Test 2.
var a = ad.lift(new Tensor([1, 2]).fromArray([[1, 2]]));
var b = ad.lift(new Tensor([2, 1]).fromArray([[3], [4]]));
var y = ad.tensor.dot(a, b);
y.backprop();
puts(y);
puts(a);
puts(b);
