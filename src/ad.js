'use strict';

var ad = require('adnn/ad');
var Tensor = require('./tensor');

ad.tensor.transpose = ad.newUnaryFunction({
  OutputType: Tensor,
  name: 'transpose',
  forward: function(a) {
    return a.T();
  },
  backward: function(a) {
    var h = this.x.dims[0];
    var w = this.x.dims[1];
    for (var i = 0; i < h; i++) {
      for (var j = 0; j < w; j++) {
        a.dx.data[j * h + i] += this.dx.data[i * w + j];
      }
    }
  }
});

ad.tensor.diag = ad.newUnaryFunction({
  OutputType: Tensor,
  name: 'diag',
  forward: function(a) {
    return a.diag();
  },
  backward: function(a) {
    var n = a.dx.dims[0];
    for (var i = 0; i < n; i++) {
      a.dx.data[i] += this.dx.data[i * (n + 1)];
    }
  }
});

ad.tensor.inv = ad.newUnaryFunction({
  OutputType: Tensor,
  name: 'inverse',
  forward: function(A) {
    return A.inv();
  },
  backward: function(A) {
    var xT = this.x.T();
    A.dx = A.dx.add(xT.dot(this.dx).dot(xT).neg());
  }
});

ad.tensor.det = ad.newUnaryFunction({
  OutputType: Number,
  name: 'determinant',
  forward: function(A) {
    return A.det();
  },
  backward: function(A) {
    // A is square matrix.
    // Assume A is invertable.
    var n = A.x.dims[0];
    var invA = A.x.inv();
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        A.dx.data[i * n + j] += this.x * this.dx * invA.data[j * n + i];
      }
    }
  }
});

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
    var Bw = Bp.dims[1];
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

module.exports = ad;
